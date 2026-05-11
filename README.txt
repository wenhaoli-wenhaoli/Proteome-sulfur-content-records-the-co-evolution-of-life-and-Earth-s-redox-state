# Planetary Redox Evolution Couples Sulfur Sourcing to Proteome Desulfurization

## Repository Overview

This repository contains the main computational workflow for the manuscript "Planetary Redox Evolution Couples Sulfur Sourcing to Proteome Desulfurization". The analysis integrates phylogenomics, machine learning, molecular dating, and reconciliations to investigate the evolutionary trajectory of sulfur-containing amino acids (SAAs) in prokaryotic proteomes.

## Workflow Summary

GTDB r226 Genomes
↓
[1select_genomes.R] → Quality filtering & representative selection
↓
[2run_eggnog.sh] → Functional annotation (COG gene families)
↓
[3parse_eggnog_results.py] → COG binary matrix construction
↓
[4train_saa_model.py] → Machine learning model training (XGBoost) & SHAP analysis
↓
[5species-tree_inference.sh] → Phylogenetic tree of life (GToTree + IQ-TREE)
↓
[6gene-tree_inference.sh] → Gene trees for 226 COG families & 11 enzymes
↓
[7chronogram_construction.sh] → Molecular clock dating (PhyloBayes)
↓
[8reconciliations.sh] → Gene tree-species tree reconciliation (AleRax)
↓
[9collect_all_presence.py] → Extract presence probabilities from reconciliations
↓
[10predict_ancestral_saa.py] → Ancestral SAA frequency prediction


## Script Descriptions

### 1. `1select_genomes.R`

**Purpose:** Quality filtering and representative genome selection from GTDB release 226.

**Key steps:**
- Load bacterial (bac120_metadata_r226.tsv) and archaeal (ar53_metadata_r226.tsv) metadata
- CheckM2 quality filtering (completeness ≥ 50%, contamination < 10%)
- Archaeal superphylum classification (DPANN, Asgard, TACK, Euryarchaeota)
- GUNC chimera removal
- Phylum-level filtering (≥2 classes per phylum, ≥25 genomes per phylum)
- Select representative genomes at order level (1,967 genomes for ML training)
- Select representative genomes at genus level (for HMM profile construction)

### 2. `2run_eggnog.sh`

**Purpose:** Functional annotation of representative genomes using eggNOG-mapper v2.

**Parameters:**
- Search method: DIAMOND
- Query cover: 50%
- E-value threshold: 1e-7

### 3. `3parse_eggnog_results.py`

**Purpose:** Parse eggNOG annotations and construct COG presence/absence binary matrix.

**Key steps:**
- Extract COG families from eggNOG output
- Build binary matrix (8,676 COGs × 1,967 genomes)
- Generate mapping files for downstream analyses

### 4. `4train_saa_model.py`

**Purpose:** Train machine learning models to predict proteomic SAA frequencies.

**Key steps:**
- Low-variance feature filtering - Remove COGs with near-constant presence/absence
- Feature selection via permutation importance (Random Forest) - Generate multiple feature sets at cumulative importance thresholds: 99%, 95%, 90%, 80%, 70%, 60%, 50%
- Train and compare 5 regression models across all feature sets:  LassoCV, RidgeCV, RandomForest, ExtraTrees, XGBoost
- Robustness evaluation (sparsity test) - Random removal of 10-50% COG features from the 90% feature set (best) with XGBoost (10 iterations per level)
- SHAP analysis for model interpretability - Identify top contributing gene families
- Save XGBoost model trained on 90% feature set for ancestral prediction

### 5. `5species-tree_inference.sh`

**Purpose:** Reconstruct the tree of life using 698 representative genomes.

**Key steps:**
- Extract 16 highly conserved single-copy ribosomal marker genes using HMMER
- Concatenate and trim alignments with MUSCLE and TrimAl (implemented in GToTree)
- Partitioned analysis with IQ-TREE v3.0.1
- Model selection: ModelFinder with `-m MFP+MERGE`
- Branch support: 1,000 ultrafast bootstrap replicates + SH-aLRT
- Three independent runs, select tree with highest likelihood

### 6. `6gene-tree_inference.sh`

**Purpose:** Reconstruct phylogenetic trees for 226 key COG gene families and 11 sulfonate-sulfite interconversion enzymes.

**COG Gene Families (226 key COGs):**
- Identified via machine learning feature selection (permutation importance, 95% cumulative importance)
- Represent core functional categories driving SAA composition patterns

**Sulfonate-Sulfite Interconversion Enzymes (11 enzymes):**
| Enzyme | Function |
|--------|----------|
| sqdB | UDP-sulfoquinovose synthase (biosynthesis) |
| comA | Phosphosulfolactate synthase (biosynthesis) |
| cs | Cysteate synthase (biosynthesis) |
| smoC | Sulfoquinovose degradation |
| sqoD | Sulfoquinovose degradation |
| xsc | Sulfolactaldehyde degradation |
| cuyA | Cysteate sulfo-lyase |
| ssuD | Alkanesulfonate monooxygenase |
| suyB | Sulfolactate sulfo-lyase |
| tauD | Taurine dioxygenase |
| hpsG/iseG | (2S)-3-sulfopropanediol / Isethionate sulfolyase |

**Key steps:**
- Extract sequences for each COG/enzyme from genome proteomes
- Multiple sequence alignment: MAFFT (`--maxiterate 1000 --localpair`)
- Alignment trimming: TrimAl (`-automated1`)
- Model selection: IQ-TREE with expanded search space
  - Rate heterogeneity: `-mrate E,I,G,I+G,R`
  - Mixture models: `-madd C10-C60, EX2, EX3, EHO, UL2, UL3, EX_EHO, LG4M, LG4X, CF4, LG+C10-LG+C60`
- Branch support: 1,000 ultrafast bootstrap replicates

### 7. `7chronogram_construction.sh`

**Purpose:** Time-calibrated phylogenetic analysis using PhyloBayes v4.1.

**Calibration Points:**

| Calibration | Minimum age (Ga) | Maximum age (Ga) |
|-------------|------------------|------------------|
| LUCA (Bacteria vs Archaea) | 3.5 | 4.4 |
| TACK + Euryarchaeota MRCA | 3.42 | - |
| Thylakoid cyanobacteria MRCA | 1.75 | - |
| Heterocyst cyanobacteria MRCA | 1.0 | - |
| Eukaryote divergence | 1.64 | - |
| Chromatiaceae divergence | 1.64 | - |

**Molecular Clock Models:**
- Autocorrelated lognormal (LN)
- Autocorrelated Cox-Ingersoll-Ross (CIR)
- Uncorrelated gamma multipliers (UGAM)

**Settings:**
- Substitution model: C20
- Chains: 2 independent runs per model
- Cycles: >50,000
- Burn-in: 25%

**Convergence Criteria:**
- Maximum discrepancy between chains < 0.3
- Minimum effective sample size > 50

### 8. `8reconciliations.sh`

**Purpose:** Gene tree-species tree reconciliation using AleRax.

**Key steps:**
- Reconcile 226 COG gene trees with time-calibrated species tree
- Generate node-specific presence probabilities
- Account for phylogenetic uncertainty (1,000 gene tree samples)
- For sulfonate metabolism enzymes: presence probability >0.5 considered present

### 9. `9collect_all_presence.py`

**Purpose:** Extract presence probabilities from AleRax reconciliation outputs.

**Key steps:**
- Parse `perspecies_eventcount.txt` files from each COG/enzyme directory
- Extract node-specific presence probabilities (column 6)
- Build comprehensive presence matrix (nodes × families)
- Output TSV file for downstream prediction

### 10. `10predict_ancestral_saa.py`

**Purpose:** Predict ancestral SAA frequencies using trained XGBoost model.

**Key steps:**
- Load trained model, scaler, and selected features
- Process three molecular clock models (CIR, LN, UGAM)
- Data augmentation (100 random samples per ancestral node based on presence probabilities)
- Predict SAA frequencies for each internal node
- Generate separate outputs for supplementary materials
- Generate merged output for main text figure



### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| R | ≥4.2 | Data processing, statistics |
| Python | ≥3.10 | Machine learning, predictions |
| GTDB-Tk | r226 | Genome database |
| eggNOG-mapper | v2 | Functional annotation |
| DIAMOND | ≥2.0 | Sequence search |
| GToTree | v1.8.10 | Marker gene extraction |
| IQ-TREE | v3.0.1 | Phylogenetic inference |
| MAFFT | v7.525 | Sequence alignment |
| TrimAl | v1.4.1 | Alignment trimming |
| HMMER | ≥3.0 | HMM search |
| PhyloBayes | v4.1 | Molecular clock dating |
| AleRax | v1.4.0 | Gene tree reconciliation |