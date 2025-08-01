# CLARA: Controllable Learning for Anomaly Recognition in Astrophysics

The software implementation of this independent research project is under development and available at https://github.com/googleboy-byte/clara-cli

## Overview

**CLARA** is a research pipeline and toolkit for unsupervised and weakly supervised discovery of astrophysical phenomena in TESS light curves. It is designed to systematically explore how the design of synthetic training sets and feature engineering can steer the behavior of Unsupervised Random Forests (URFs) and clustering algorithms, enabling the identification and interpretation of anomalies—such as exoplanet transits, eclipsing binaries, and other variable stars—directly from large-scale survey data.

CLARA is built for:
- **Astrophysicists** seeking to discover and interpret new phenomena in TESS data
- **Machine learning researchers** interested in controllable unsupervised learning
- **Data scientists** working on time-series anomaly detection at scale

The project is inspired by and extends the work of Crake & Martínez-Galarza (2023), introducing a novel, systematic approach to synthetic set design and feature-to-metric mapping for anomaly detection.

---

## Key Features & Innovations
- **Synthetic Set Design as a Control Lever:** Systematically vary transit count, duration, cadence, and noise to steer URF behavior
- **Feature Engineering:** Combine flux and periodogram features for robust anomaly detection
- **URF Model Variants:** Multiple URF architectures (URF1–4) and 36+ subvariants for parameter sweeps
- **Clustering & Morphological Analysis:** Multi-stage clustering (DBSCAN, GMM, DPMM, DTW) and t-SNE visualization for physical interpretation
- **Morphological Classification via Cosine Similarity:** Classify anomalies by comparing their feature vectors to SIMBAD-labeled TOIs, enabling weakly supervised, physically interpretable anomaly grouping
- **Astrometric & Astrophysical Interpretation:** Cross-match anomalies with GAIA DR3 and analyze WRSS-normalized scores against stellar parameters (Teff, RUWE, vtan, etc.) for physical insight into anomaly populations
- **Cross-Sector & Cross-Parameter Validation:** Test robustness and generalization across TESS sectors and synthetic regimes
- **Physical Interpretation:** Integrate SIMBAD/GAIA cross-matching for astrophysical labeling
- **Reproducibility & Transparency:** Extensive Q&A, self-criticism, and open methodology

---

## Directory Structure

```
./
├── clara/                # Core library: feature extraction, URF, clustering, utilities
│   ├── astrodata_helpers/    # Astrophysical helpers (phase folding, SIMBAD, etc.)
│   ├── test_helpers/         # Test set and importance score utilities
│   └── ...                   # Main pipeline modules (see below)
├── notebooks/            # Research notebooks (Part 1, 2A, 2B, Q&A)
├── models/               # Saved URF models (URF1–4, subvariants)
├── data/                 # Intermediate data, synthetic light curves, test sets
├── figures/              # Key plots and clustering visualizations
├── results/              # Evaluation CSVs, metrics, and summary tables
├── logs/                 # Download and processing logs
├── catalogues/           # TOI catalogues and TESS download scripts
├── test/                 # Evaluation/test subset results
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── LICENSE               # MIT License
```

---

## Core Methodology

### 1. Synthetic Set Design
- Generate synthetic light curves using `batman` and custom routines
- Systematically vary:
  - **Transit count** (n = 100, 200, 300)
  - **Duration** (d = 13, 27 hours)
  - **Cadence** (c = 2, 10 min)
  - **Noise** (50, 100, 200 ppm)
- 36+ synthetic variants for controlled experiments

### 2. Feature Extraction
- Extract 3000 normalized flux points and 1000 Lomb-Scargle periodogram powers per light curve
- Parallelized for large-scale processing

### 3. URF Training & Scoring
- Train URF models (URF1–4) using real vs synthetic sets
- Hyperparameter search and model selection
- Score all light curves for anomaly detection

### 4. Clustering & Morphological Analysis
- Multi-stage clustering: DBSCAN, GMM, DPMM, DTW
- t-SNE and PCA for visualization
- Cross-sector and cross-parameter robustness testing

### 5. Physical Interpretation & Validation
- Cross-match anomalies with SIMBAD/GAIA for astrophysical labeling
- Analyze TOI recall, anomaly rate, and importance metrics
- Q&A notebook for transparency and self-criticism

### 6. Morphological and Physical Interpretation
- **Cosine Similarity Matching:**
  - Compute cosine similarity between anomaly feature vectors and those of SIMBAD-labeled TOIs (planets, binaries, etc.)
  - Assign morphological classes to anomalies based on their closest match (e.g., "planet-like", "binary-like")
- **GAIA Cross-Matching & WRSS Analysis:**
  - Cross-match high-anomaly light curves with GAIA DR3 to obtain astrometric and astrophysical parameters (Teff, RUWE, vtan, etc.)
  - Correlate WRSS-normalized anomaly scores with these properties to interpret the nature of detected anomalies and validate URF-4 scoring behaviour

---

## Notebooks

- **Part 1.ipynb**: End-to-end pipeline for synthetic set design, feature extraction, URF training, and initial evaluation
- **Part 2A - Feature to Metric Mapping for Model Selection.ipynb**: Systematic parameter sweeps, model selection, and feature-to-metric mapping
- **Part 2B.ipynb**: Cross-sector/parameter generalization, clustering, physical interpretation, and robustness analysis (including cosine similarity classification and GAIA cross-matching)
- **qa_part1.ipynb**: Extensive Q&A covering dataset choices, methodology, limitations, and future work

Each notebook is heavily commented and includes markdown explanations for all major steps, design decisions, and findings.

---

## clara/ Library Modules

- **clara_generate_synth_lc.py**: Synthetic light curve generation
- **clara_feature_extraction_parallel.py**: Parallelized feature extraction
- **clara_urf_predictor.py**: URF model training, scoring, and evaluation
- **clara_urf4_test_suite.py**: Automated test suite for URF-4 subvariants
- **clara_urf4_subvariant_analysis.py**: Analysis of URF-4 subvariant results
- **clara_utils.py**: Utilities for file handling, logging, and TESS downloads
- **clara_viz.py**: Visualization helpers (flux, periodogram, t-SNE)
- **astrodata_helpers/**: Phase folding, SIMBAD/GAIA cross-matching, astrophysical utilities
- **test_helpers/**: Test set generation, TOI importance scoring

---

## Installation & Requirements

- Python 3.9+
- Linux (tested), should work on Mac/Windows with minor tweaks
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Key libraries: numpy, pandas, scipy, scikit-learn, astropy, lightkurve, batman-package, matplotlib, seaborn, plotly, tqdm, joblib

---

## Usage Guide

1. **Download TESS light curves** using provided scripts in `catalogues/` or via the notebook
2. **Generate synthetic light curves** with `clara_generate_synth_lc.py` or via the notebook
3. **Extract features** using `clara_feature_extraction_parallel.py`
4. **Train and score URF models** using `clara_urf_predictor.py` or the notebooks
5. **Run clustering and analysis** via Part 2B notebook or `clara_urf4_test_suite.py`
6. **Visualize results** in `figures/` and analyze metrics in `results/`

All steps are reproducible via the notebooks, which include code, comments, and markdown explanations.

---

## Results & Figures

- **figures/**: Key plots (performance, clustering, t-SNE, feature importance, cosine similarity classification, GAIA overlays)
- **results/**: CSVs with anomaly scores, recall, importance, summary tables, and physical interpretation results
- **models/**: Pre-trained URF models for all variants
- **data/**: Synthetic and real light curves, intermediate features

---

## Documentation, Q&A, and Transparency

- **qa_part1.ipynb**: Answers to all major methodological, scientific, and technical questions
- **In-notebook markdown**: Detailed explanations, limitations, and future directions
- **Code comments**: Every major function and step is documented

---

## Citation & Acknowledgements

- Builds on Crake & Martínez-Galarza (2023) for URF methodology
- Uses TESS data from NASA/MAST
- Lomb-Scargle periodogram via VanderPlas (2018)
- Please cite the forthcoming CLARA paper if using this code or results

---

## License

MIT License. See `LICENSE` for details.
