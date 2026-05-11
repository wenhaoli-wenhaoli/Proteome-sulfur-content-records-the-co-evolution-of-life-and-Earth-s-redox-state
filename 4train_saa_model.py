#!/usr/bin/env python3
"""
train_saa_model.py

Train machine learning model to predict sulfur-containing amino acid (SAA) frequency
with feature gradient evaluation (multiple feature sets based on permutation importance)

Prerequisites:
- parse_eggnog_results.py must be completed to generate genome_cog_matrix.tsv
- The file contains: genome IDs, COG family presence/absence (0/1), target variable s_aa_freq

Input: genome_cog_matrix.tsv (or similarly named file with COG columns)
Output: Model files, feature sets, performance evaluation results, SHAP analysis results,
        robustness/sparsity test results

前置要求:
- 已完成 parse_eggnog_results.py，生成 genome_cog_matrix.tsv
- 该文件包含: 基因组ID、各COG家族的存在/缺失值(0/1)、目标变量 s_aa_freq

输入: genome_cog_matrix.tsv
输出: 模型文件、特征集、性能评估结果、SHAP分析结果、稀疏性测试结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
from copy import deepcopy
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore", category=UserWarning)

# Set font embedding for EPS/PDF output
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('results/saved_models', exist_ok=True)
os.makedirs('results/feature_sets', exist_ok=True)

# ==========================================
# 1. Load data and binary conversion
# ==========================================
print("=" * 70)
print("Step 1/9: Loading COG binary matrix")
print("=" * 70)

df = pd.read_csv('genome_cog_matrix.tsv', sep='\t', low_memory=False)
X_counts = df.filter(regex='^COG')
print(f"Raw COG features: {X_counts.shape[1]}")

# Convert to binary (presence/absence)
X_binary = (X_counts > 0).astype(np.int8)
print("Converted to binary matrix (0=absent, 1=present)")

# Initial variance filtering
print("\nPerforming low variance filtering...")
selector = VarianceThreshold(threshold=0.998 * (1 - 0.998))
X_filtered = selector.fit_transform(X_binary)
kept_columns = X_binary.columns[selector.get_support()].tolist()
print(f"Features after low variance filtering: {len(kept_columns)}")

X = pd.DataFrame(X_filtered, columns=kept_columns)
target = 's_aa_freq'

# Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, df[target], test_size=0.2, random_state=42
)
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ==========================================
# 2. Permutation importance for feature selection
# ==========================================
print("\n" + "=" * 70)
print("Step 2/9: Computing permutation importance")
print("=" * 70)

scaler_temp = StandardScaler()
X_train_scaled_temp = scaler_temp.fit_transform(X_train)
X_test_scaled_temp = scaler_temp.transform(X_test)

print("Training temporary Random Forest for feature importance evaluation...")
rf_temp = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=4, random_state=42)
rf_temp.fit(X_train_scaled_temp, y_train)

print("Computing permutation importance...")
perm_result = permutation_importance(rf_temp, X_test_scaled_temp, y_test,
                                     n_repeats=5, random_state=42, n_jobs=4)

perm_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance_mean': perm_result.importances_mean
}).sort_values('importance_mean', ascending=False)

# Calculate cumulative importance
cumsum = perm_df['importance_mean'].cumsum() / perm_df['importance_mean'].sum()

# Define feature gradient thresholds
thresholds = {
    '99%': 0.99,
    '95%': 0.95,
    '90%': 0.90,
    '80%': 0.80,
    '70%': 0.70,
    '60%': 0.60,
    '50%': 0.50
}
feature_sets = {'full': list(X_train.columns)}

print("\nFeature set sizes by cumulative importance:")
print(f"  - full: {len(feature_sets['full'])} features")

for name, thresh in thresholds.items():
    n_select = (cumsum <= thresh).sum()
    n_select = max(1, n_select)
    selected = perm_df['feature'].head(n_select).tolist()
    feature_sets[name] = selected
    print(f"  - {name}: {n_select} features (cumulative importance: {cumsum.iloc[n_select-1]:.4f})")

# ==========================================
# 3. Save feature set COG lists
# ==========================================
print("\n" + "=" * 70)
print("Step 3/9: Saving feature set COG lists")
print("=" * 70)

for set_name, features in feature_sets.items():
    safe_name = set_name.replace('%', 'percent')
    output_file = f'results/feature_sets/{safe_name}_COG_list.tsv'
    pd.DataFrame({'COG': features}).to_csv(output_file, sep='\t', index=False)
    print(f"Saved: {output_file} ({len(features)} COGs)")

# Save full importance ranking
perm_df['cumulative_importance'] = cumsum
perm_df.to_csv('results/feature_importance_ranking.tsv', sep='\t', index=False)

# Save 90% feature set gene list
features_90 = feature_sets['90%']
genes_90_df = pd.DataFrame({
    'gene_family': features_90,
    'importance_rank': range(1, len(features_90) + 1),
    'importance_score': perm_df[perm_df['feature'].isin(features_90)]['importance_mean'].values
})
genes_90_df.to_csv('results/90percent_features_gene_list.tsv', sep='\t', index=False)
print("\n90% feature set gene list saved: results/90percent_features_gene_list.tsv")

# ==========================================
# 4. Define five regression models
# ==========================================
print("\n" + "=" * 70)
print("Step 4/9: Defining five regression models")
print("=" * 70)

models = {
    "LassoCV": LassoCV(cv=5, max_iter=5000, n_jobs=4, random_state=42, tol=1e-3),
    "RidgeCV": RidgeCV(cv=5, alphas=np.logspace(-6, 6, 13)),
    "RandomForest": RandomForestRegressor(
        n_estimators=600, min_samples_split=4, min_samples_leaf=2,
        max_features=0.2, max_depth=None, bootstrap=False,
        random_state=42, n_jobs=4
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=400, max_features=0.3, max_depth=None,
        random_state=42, n_jobs=4
    ),
    "XGBoost": XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
        random_state=42, n_jobs=4, tree_method='hist'
    )
}
print(f"Defined {len(models)} models: {', '.join(models.keys())}")

# ==========================================
# 5. Evaluate models across all feature sets
# ==========================================
print("\n" + "=" * 70)
print("Step 5/9: Evaluating models across feature gradients")
print("=" * 70)

all_performance_results = []
total_combinations = len(feature_sets) * len(models)
pbar = tqdm(total=total_combinations, desc="Performance evaluation", unit="combination")

for set_name, features in feature_sets.items():
    X_train_sub = X_train[features]
    X_test_sub = X_test[features]

    scaler_sub = StandardScaler()
    X_train_sub_scaled = scaler_sub.fit_transform(X_train_sub)
    X_test_sub_scaled = scaler_sub.transform(X_test_sub)

    for model_name, model in models.items():
        try:
            model_copy = deepcopy(model)
            model_copy.fit(X_train_sub_scaled, y_train)
            y_pred = model_copy.predict(X_test_sub_scaled)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            cv_scores = cross_val_score(model_copy, X_train_sub_scaled, y_train, cv=5, scoring='r2')

            all_performance_results.append({
                'Feature_Set': set_name,
                'Model': model_name,
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae,
                'CV_Mean_R²': cv_scores.mean()
            })
        except Exception as e:
            all_performance_results.append({
                'Feature_Set': set_name,
                'Model': model_name,
                'R²': np.nan,
                'RMSE': np.nan,
                'MAE': np.nan,
                'CV_Mean_R²': np.nan
            })
        pbar.update(1)

pbar.close()

df_performance = pd.DataFrame(all_performance_results)
df_performance.to_csv('results/all_gradients_performance.tsv', sep='\t', index=False)
print("\nFull performance comparison saved: results/all_gradients_performance.tsv")

# ==========================================
# 6. Generate model performance comparison plots
# ==========================================
print("\n" + "=" * 70)
print("Step 6/9: Generating performance comparison plots")
print("=" * 70)

feature_set_order = ['full', '99%', '95%', '90%', '80%', '70%', '60%', '50%']
df_performance['Feature_Set'] = pd.Categorical(df_performance['Feature_Set'],
                                               categories=feature_set_order, ordered=True)
df_performance_plot = df_performance.dropna(subset=['R²', 'RMSE'])

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

sns.barplot(data=df_performance_plot, x='Model', y='R²', hue='Feature_Set', ax=axes[0], palette='viridis')
axes[0].set_title('Model Performance Comparison - R²', fontsize=14)
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('R²', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].legend(title='Feature Set', bbox_to_anchor=(1.05, 1), loc='upper left')

sns.barplot(data=df_performance_plot, x='Model', y='RMSE', hue='Feature_Set', ax=axes[1], palette='viridis')
axes[1].set_title('Model Performance Comparison - RMSE', fontsize=14)
axes[1].set_xlabel('Model', fontsize=12)
axes[1].set_ylabel('RMSE', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].legend(title='Feature Set', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('results/model_performance_gradients_bar.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('results/model_performance_gradients_bar.pdf', format='pdf', dpi=400, bbox_inches='tight')
plt.close()

# Feature count vs R² relationship
summary_df = df_performance.dropna().groupby('Feature_Set').agg({
    'R²': ['mean', 'std'],
    'Model': lambda x: x[df_performance.loc[x.index, 'R²'].idxmax()]
}).reset_index()
summary_df.columns = ['Feature_Set', 'R²_mean', 'R²_std', 'Best_Model']
feature_counts = {name: len(features) for name, features in feature_sets.items()}
summary_df['Feature_Count'] = summary_df['Feature_Set'].map(feature_counts)
summary_df = summary_df.sort_values('Feature_Count', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(summary_df['Feature_Count'], summary_df['R²_mean'], yerr=summary_df['R²_std'],
            fmt='o-', capsize=5, capthick=2, markersize=8, color='steelblue', ecolor='gray')
ax.set_xlabel('Number of Features', fontsize=12)
ax.set_ylabel('R² (mean ± std)', fontsize=12)
ax.set_title('Feature Count vs Prediction Performance', fontsize=14)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
for _, row in summary_df.iterrows():
    ax.annotate(row['Best_Model'], (row['Feature_Count'], row['R²_mean']),
                textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)
plt.tight_layout()
plt.savefig('results/feature_count_vs_performance.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('results/feature_count_vs_performance.pdf', format='pdf', dpi=400, bbox_inches='tight')
plt.close()
print("Performance plots saved to: results/")

# ==========================================
# 7. XGBoost 90% feature set detailed evaluation
# ==========================================
print("\n" + "=" * 70)
print("Step 7/9: XGBoost 90% feature set detailed analysis")
print("=" * 70)

X_train_90 = X_train[features_90]
X_test_90 = X_test[features_90]

scaler_90 = StandardScaler()
X_train_90_scaled = scaler_90.fit_transform(X_train_90)
X_test_90_scaled = scaler_90.transform(X_test_90)

print("Training XGBoost on 90% feature set...")
xgb_model = XGBRegressor(
    n_estimators=600, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
    random_state=42, n_jobs=4, tree_method='hist'
)
xgb_model.fit(X_train_90_scaled, y_train)

y_pred_90 = xgb_model.predict(X_test_90_scaled)
r2_90 = r2_score(y_test, y_pred_90)
rmse_90 = np.sqrt(mean_squared_error(y_test, y_pred_90))
mae_90 = mean_absolute_error(y_test, y_pred_90)

print(f"XGBoost 90% feature set performance:")
print(f"  R² = {r2_90:.4f}")
print(f"  RMSE = {rmse_90:.4f}")
print(f"  MAE = {mae_90:.4f}")

# Save model and related files
joblib.dump(xgb_model, 'results/saved_models/XGBoost_90percent.pkl')
joblib.dump(scaler_90, 'results/saved_models/XGBoost_90percent_scaler.pkl')
joblib.dump(features_90, 'results/saved_models/XGBoost_90percent_features.pkl')
print("Model saved to: results/saved_models/")

# ==========================================
# 8. Sparsity test on XGBoost 90% feature set (dual-axis plot)
# ==========================================
print("\n" + "=" * 70)
print("Step 8/9: XGBoost 90% feature set sparsity test")
print("=" * 70)

sparsity_levels = [0, 10, 20, 30, 40, 50]
n_iterations = 10
sparsity_results = []

total_sparsity_tests = len(sparsity_levels) * n_iterations
pbar = tqdm(total=total_sparsity_tests, desc="Sparsity test", unit="iteration")

for sparsity_level in sparsity_levels:
    for iteration in range(n_iterations):
        if sparsity_level == 0:
            kept_features = features_90.copy()
        else:
            keep_ratio = 1 - sparsity_level / 100
            n_keep = max(1, int(len(features_90) * keep_ratio))
            np.random.seed(iteration * 100 + sparsity_level)
            kept_features = np.random.choice(features_90, size=n_keep, replace=False).tolist()

        X_train_sparse = X_train_90[kept_features]
        X_test_sparse = X_test_90[kept_features]

        scaler_sparse = StandardScaler()
        X_train_sparse_scaled = scaler_sparse.fit_transform(X_train_sparse)
        X_test_sparse_scaled = scaler_sparse.transform(X_test_sparse)

        model = XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
            random_state=42, n_jobs=4, tree_method='hist'
        )
        model.fit(X_train_sparse_scaled, y_train)
        y_pred = model.predict(X_test_sparse_scaled)

        sparsity_results.append({
            'Sparsity_Level': sparsity_level,
            'Iteration': iteration,
            'n_features_kept': len(kept_features),
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        })
        pbar.update(1)

pbar.close()

df_sparsity = pd.DataFrame(sparsity_results)
df_sparsity.to_csv('results/xgboost_90_sparsity_results.tsv', sep='\t', index=False)

sparsity_summary = df_sparsity.groupby('Sparsity_Level').agg({
    'R²': ['mean', 'std'],
    'RMSE': ['mean', 'std'],
    'n_features_kept': 'mean'
}).reset_index()
sparsity_summary.columns = ['Sparsity_Level', 'R²_mean', 'R²_std', 'RMSE_mean', 'RMSE_std', 'n_features_kept_mean']
sparsity_summary.to_csv('results/xgboost_90_sparsity_summary.tsv', sep='\t', index=False)

# Generate dual-axis sparsity plot (left: R², right: RMSE)
print("Generating dual-axis sparsity plot...")

fig, ax1 = plt.subplots(figsize=(10, 6))

# Left y-axis: R²
summary_r2 = df_sparsity.groupby('Sparsity_Level')['R²'].agg(['mean', 'std']).reset_index()
ax1.plot(summary_r2['Sparsity_Level'], summary_r2['mean'], 'o-',
         color='steelblue', linewidth=2, markersize=8, label='R²')
ax1.fill_between(summary_r2['Sparsity_Level'],
                 summary_r2['mean'] - summary_r2['std'],
                 summary_r2['mean'] + summary_r2['std'],
                 alpha=0.2, color='steelblue')
ax1.set_xlabel('Random Feature Removal (%)', fontsize=12)
ax1.set_ylabel('R²', fontsize=12, color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_xticks(sparsity_levels)
ax1.grid(True, alpha=0.3)

# Right y-axis: RMSE
ax2 = ax1.twinx()
summary_rmse = df_sparsity.groupby('Sparsity_Level')['RMSE'].agg(['mean', 'std']).reset_index()
ax2.plot(summary_rmse['Sparsity_Level'], summary_rmse['mean'], 's-',
         color='coral', linewidth=2, markersize=8, label='RMSE')
ax2.fill_between(summary_rmse['Sparsity_Level'],
                 summary_rmse['mean'] - summary_rmse['std'],
                 summary_rmse['mean'] + summary_rmse['std'],
                 alpha=0.2, color='coral')
ax2.set_ylabel('RMSE', fontsize=12, color='coral')
ax2.tick_params(axis='y', labelcolor='coral')

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

ax1.set_title('Sparsity Robustness of XGBoost on 90% Feature Set', fontsize=14)

plt.tight_layout()
plt.savefig('results/xgboost_90_sparsity_robustness.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('results/xgboost_90_sparsity_robustness.pdf', format='pdf', dpi=400, bbox_inches='tight')
plt.close()
print("Sparsity test completed. Dual-axis plot saved: results/xgboost_90_sparsity_robustness.eps/.pdf")

# ==========================================
# 9. SHAP analysis (interpretability)
# ==========================================
print("\n" + "=" * 70)
print("Step 9/9: SHAP analysis")
print("=" * 70)

print("Computing SHAP values...")
n_shap_samples = min(500, X_train_90_scaled.shape[0])
X_train_shap = X_train_90_scaled[:n_shap_samples]

print(f"Using {n_shap_samples} samples for SHAP calculation...")
explainer = shap.Explainer(xgb_model, X_train_shap, feature_names=features_90)
shap_values = explainer(X_train_shap)

print("Generating SHAP summary plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_train_shap, feature_names=features_90, show=False)
plt.tight_layout()
plt.savefig('results/shap_summary_plot.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('results/shap_summary_plot.pdf', format='pdf', dpi=400, bbox_inches='tight')
plt.close()

print("Generating SHAP bar plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_train_shap, feature_names=features_90, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig('results/shap_bar_plot.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('results/shap_bar_plot.pdf', format='pdf', dpi=400, bbox_inches='tight')
plt.close()

print("Generating Top 20 SHAP importance plot...")
shap_mean_abs = np.abs(shap_values.values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    'feature': features_90,
    'mean_abs_shap': shap_mean_abs
}).sort_values('mean_abs_shap', ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(shap_importance_df['feature'][::-1], shap_importance_df['mean_abs_shap'][::-1], color='steelblue')
plt.xlabel('Mean |SHAP value|', fontsize=12)
plt.ylabel('Gene Family', fontsize=12)
plt.title('Top 20 Gene Families by SHAP Importance', fontsize=14)
plt.tight_layout()
plt.savefig('results/shap_top20_bar.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('results/shap_top20_bar.pdf', format='pdf', dpi=400, bbox_inches='tight')
plt.close()

print("Generating SHAP consistency check plot...")
shap_sum = shap_values.values.sum(axis=1) + shap_values.base_values
plt.figure(figsize=(8, 6))
plt.scatter(shap_sum, xgb_model.predict(X_train_shap), alpha=0.5, s=20, color='steelblue')
plt.plot([shap_sum.min(), shap_sum.max()], [shap_sum.min(), shap_sum.max()], 'r--', lw=2)
plt.xlabel('SHAP sum + base value', fontsize=12)
plt.ylabel('Model Prediction', fontsize=12)
plt.title('SHAP Consistency Check', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/shap_consistency.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('results/shap_consistency.pdf', format='pdf', dpi=400, bbox_inches='tight')
plt.close()

# ==========================================
# Final summary
# ==========================================
print("\n" + "=" * 70)
print("Analysis completed - Summary")
print("=" * 70)
print("""
Output files (results/ directory):
  [Feature Set COG Lists]
    - feature_sets/                     (%50, %60, %70, %80, %90, %95, %99, full)
  [Feature Selection Results]
    - feature_importance_ranking.tsv
    - 90percent_features_gene_list.tsv
  [Model Performance Evaluation]
    - all_gradients_performance.tsv
    - model_performance_gradients_bar.eps/.pdf
    - feature_count_vs_performance.eps/.pdf
  [XGBoost 90% Model]
    - saved_models/XGBoost_90percent.pkl
    - saved_models/XGBoost_90percent_scaler.pkl
    - saved_models/XGBoost_90percent_features.pkl
  [Sparsity Test]
    - xgboost_90_sparsity_results.tsv
    - xgboost_90_sparsity_summary.tsv
    - xgboost_90_sparsity_robustness.eps/.pdf (dual-axis: left R², right RMSE)
  [SHAP Analysis]
    - shap_summary_plot.eps/.pdf
    - shap_bar_plot.eps/.pdf
    - shap_top20_bar.eps/.pdf
    - shap_consistency.eps/.pdf
""")

print("\nAll steps completed successfully!")