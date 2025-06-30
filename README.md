# 🪐 CLARA: Controllable Learning for Anomaly Recognition in Astrophysics  
### Part 1 – Mapping Synthetic Set Design to Anomaly Detection in TESS Light Curves

CLARA is a research pipeline designed to explore how **synthetic set design influences the behavior of Unsupervised Random Forests (URFs)** when applied to light curves from the TESS mission.  
In **Part 1**, we present a comprehensive framework for generating, scoring, and evaluating light curve anomalies using synthetic light curves with tunable features.

This repository contains the **production-ready pipeline, models, and results** used in the CLARA Part 1 study.

---

## 📂 Project Structure
<pre><code>./
├── clara/ # Modular code used in the pipeline
├── notebooks/ # Additional Q&A and research notebooks {clara_part1_notebook.ipynb for part 1}
├── models/ # Saved URF models (URF1–4, urf-4 variants)
├── data/ # Intermediate data and synthetic light curves
├── figures/ # Key plots used in the paper
├── logs/ # Log files for reproducibility
├── results/ # CSVs and dataframes from evaluations
├── catalogues/ # TOI catalogues and download scripts
├── test/ # Evaluation/test subset results</code></pre>

##### Note: Tess data is present in the parent directory of this root folder. For this project, its path from this root project folder is ../downloaded_lc/tess_lc/{sector_number} where sector_number is 1 through 91 as needed to be downloaded. there is a section for downloading 2 min cadence curves in the part 1 notebook.

## 🛰️ About the Project

**CLARA (Clustering Lightcurves with Anomalies Revealed by AI)** is a research framework to study how synthetic light curve design influences unsupervised anomaly detection in TESS data using Unsupervised Random Forests (URFs).

This repository presents the full pipeline and results from **Part 1** of the CLARA project, which demonstrates:

- That URF behavior can be controlled by manipulating the input feature distributions of synthetic (non-anomalous) training sets.
- That anomaly detection performance — evaluated using real TOIs (TESS Objects of Interest) — varies systematically with synthetic light curve parameters like:
  - Transit count
  - Duration
  - Cadence
  - Noise level
- That interpretability is enhanced by studying how URF anomaly scores and importance scores change with these parameters.
- That the best-performing models achieve high TOI recall and meaningful feature importance rankings, even in the absence of supervised labels.

The project builds on ideas from Crake & Martínez-Galarza (2023) and extends them by exploring synthetic light curve design as a controllable input to steer URF outcomes for different scientific goals.

## ⚙️ Installation & Requirements

This project is designed for Python 3.9+ and tested on Linux systems. It can be run on modest hardware (e.g., 32 GB RAM, 4-core CPU) but benefits from parallelization.

### 📦 Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

**Key libraries used:**

- `numpy`, `pandas`, `scipy`, `scikit-learn` – Core numerical processing and machine learning
- `astropy`, `lightkurve`, `batman-package` – Light curve I/O, manipulation, and synthetic generation
- `matplotlib`, `seaborn`, `plotly`, `tqdm` – Plotting and progress tracking
- `joblib`, `glob`, `argparse`, `os` – File handling, CLI interfaces, model persistence


The **part 1** notebook performs the following:

- Loads real and synthetic light curves
- Extracts flux and Lomb-Scargle features
- Trains multiple URF variants (URF-1 through URF-4)
- Scores all light curves
- Evaluates performance using TOIs
- Analyzes correlations between synthetic parameters and model metrics

Intermediate feature files and model outputs are cached to disk for reproducibility.

### Visualize results
- All performance plots, score histograms, and correlation heatmaps are saved in the `figures/` folder.
- Evaluation outputs and metrics are stored in `results/`.

The `clara_urf4_subvariant_analysis.py` and `clara_urf4_test_suite.py` scripts inside `clara/` allow **deeper testing** and exploration of URF-4 model subvariants.

## Citation & Acknowledgements

This project builds upon:

- Crake & Martínez-Galarza (2023) — for baseline URF architecture
- VanderPlas (2018) — for Lomb-Scargle periodogram methodology
- TESS SPOC pipeline — for curated 2-minute cadence light curves

If using this code or results, please cite the forthcoming CLARA Part 1 paper.

TESS data courtesy of NASA and MAST archive.

## License

MIT License. See `LICENSE` file for details.
