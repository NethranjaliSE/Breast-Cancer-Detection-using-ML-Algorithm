# 🩺 Breast Cancer Prediction — Production ML Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-29%20Passing-brightgreen?style=for-the-badge)

<br/>

**A production-grade machine learning pipeline for early breast cancer detection.**  
Built on the Wisconsin Diagnosis Breast Cancer (WDBC) dataset using an ensemble of  
Random Forest · kNN · Naive Bayes · SVM · Logistic Regression.

<br/>

[🚀 Live Demo](#-live-demo) · [📊 Results](#-results) · [⚡ Quick Start](#-quick-start) · [🔌 API Docs](#-api-endpoints) · [📁 Structure](#-project-structure)

</div>

---

## 🌐 Live Demo

> **Hosted on Hugging Face Spaces**

| Resource | Link |
|---|---|
| 🤗 **Hugging Face Space** | [https://huggingface.co/spaces/YOUR_USERNAME/breast-cancer-ml](https://huggingface.co/spaces/YOUR_USERNAME/breast-cancer-ml) |
| 📓 **Kaggle Notebook** | [https://www.kaggle.com/YOUR_USERNAME/breast-cancer-ensemble-ml](https://www.kaggle.com/YOUR_USERNAME/breast-cancer-ensemble-ml) |
| 📖 **API Docs (Swagger)** | [https://YOUR_USERNAME-breast-cancer-ml.hf.space/docs](https://YOUR_USERNAME-breast-cancer-ml.hf.space/docs) |

> ⚠️ Replace `YOUR_USERNAME` with your actual Hugging Face / Kaggle username after deployment.

---

## 📌 Research Background

This project is based on the IEEE CTEMS 2018 paper:

> *"Comparison of Machine Learning Algorithms for Breast Cancer Prediction"*  
> University of Petroleum & Energy Studies (UPES) · Amity University  
> Algorithms compared: **Random Forest · kNN · Naïve Bayes**  
> Dataset: **Wisconsin Diagnosis Breast Cancer (WDBC)** — UCI ML Repository

This implementation **extends the paper** by adding SVM, Logistic Regression, and a Soft-Voting Ensemble, plus a full production API and test suite.

---

## 📊 Results

### Performance Metrics

![Performance Metrics](docs/images/01_performance_metrics.png)

> Bar chart comparing Accuracy, Precision, Recall, F1-Score, and AUC-ROC across all 6 models.

---

### Confusion Matrices

![Confusion Matrices](docs/images/02_confusion_matrices.png)

> Confusion matrices for every model. **Recall (sensitivity) is the priority metric** — a missed malignant tumour (false negative) has far greater clinical cost than a false alarm.

---

### ROC Curves

![ROC Curves](docs/images/03_roc_curves.png)

> ROC curves with AUC scores. The Ensemble and SVM both achieve AUC ≥ 0.997.

---

### Model Summary Heatmap

![Summary Heatmap](docs/images/04_summary_heatmap.png)

> Side-by-side heatmap of all metrics across all models — darker = better.

---

### 📈 Numeric Results Table

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC | CV (10-fold) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Random Forest | 94.74% | 94.55% | 97.20% | 95.85% | 0.993 | 96.1% |
| kNN | 95.91% | 93.86% | **100.0%** | 96.83% | 0.983 | 95.8% |
| Naive Bayes | 93.57% | 93.64% | 96.26% | 94.93% | 0.989 | 93.2% |
| SVM | **97.66%** | **98.13%** | 98.13% | **98.13%** | **0.998** | 97.5% |
| Logistic Regression | 97.08% | 99.04% | 96.26% | 97.63% | 0.998 | 96.9% |
| **Ensemble (Soft Voting)** | 95.91% | 94.64% | **99.07%** | 96.80% | 0.997 | 96.4% |

> 🥇 **Best Accuracy** → SVM (97.66%)  
> 🎯 **Best Recall** → Ensemble (99.07%) — recommended for clinical use  
> ⚕️ **Clinical recommendation** → Ensemble, because it misses the fewest malignant cases

---

## ✨ Features

- ✅ **6 models trained in one method** — `train_all_models(X_train, y_train)`
- ✅ **Soft-Voting Ensemble** averages probabilities from all 5 base classifiers
- ✅ **Production FastAPI** with `/predict`, `/predict/batch`, `/health`, `/metrics`
- ✅ **Pydantic v2** request / response validation with full OpenAPI schema
- ✅ **StandardScaler Pipeline** inside every model — zero data leakage
- ✅ **10-fold Stratified CV** mirroring the paper's methodology
- ✅ **29 tests** — unit + integration — pytest
- ✅ **Rotating file logger** + structured console output
- ✅ **Settings from env vars / `.env`** (12-factor app)
- ✅ **Model registry** with `manifest.json` versioning
- ✅ **Kaggle-ready** — auto-detects `/kaggle/working` save path

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/breast-cancer-ml.git
cd breast-cancer-ml
pip install -r requirements.txt
```

### 2. Train All Models

```bash
python scripts/train.py
```

```
╔══════════════════════════════════════════════════════╗
║   Breast Cancer ML — Production Training Pipeline    ║
╚══════════════════════════════════════════════════════╝
[DATA]  Shape        : (569, 30)
[DATA]  Benign (1)   : 357  | Malignant (0): 212
[SPLIT] Train=398    Test=171
✔ Random Forest trained
✔ kNN trained
✔ Naive Bayes trained
✔ SVM trained
✔ Logistic Regression trained
✔ Ensemble (Soft Voting) trained
```

### 3. Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

### 4. Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 17.99,
    "mean_texture": 10.38,
    "mean_perimeter": 122.8,
    "mean_area": 1001.0,
    "mean_smoothness": 0.1184
  }'
```

```json
{
  "prediction": 0,
  "label": "Malignant",
  "confidence": 0.9823,
  "malignant_probability": 0.9823,
  "benign_probability": 0.0177,
  "model_used": "ensemble_soft_voting"
}
```

### 5. Run Tests

```bash
pytest tests/ -v
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — app name, version, models ready |
| `GET` | `/models` | List all saved models with metadata & sizes |
| `GET` | `/metrics` | Performance metrics for all models |
| `POST` | `/predict` | Single-sample prediction with confidence score |
| `POST` | `/predict/batch` | Upload a CSV → predictions for every row |

Interactive docs available at **`http://localhost:8000/docs`** (Swagger UI).

---

## 📁 Project Structure

```
breast_cancer_ml/
│
├── src/                          # All source code
│   ├── data/
│   │   ├── loader.py             # load_data() — sklearn / CSV / URL
│   │   └── splitter.py           # Stratified train/test split
│   │
│   ├── features/
│   │   └── preprocessor.py       # BreastCancerPreprocessor (fit/transform/save)
│   │
│   ├── models/
│   │   ├── trainer.py            # ✨ train_all_models() — ONE method, 6 models
│   │   ├── evaluator.py          # Accuracy, Precision, Recall, F1, AUC, CV
│   │   └── registry.py           # save / load / list with manifest.json
│   │
│   ├── api/
│   │   ├── main.py               # FastAPI app factory + lifespan hooks
│   │   ├── routes.py             # All endpoint handlers
│   │   └── schemas.py            # Pydantic v2 request/response models
│   │
│   └── utils/
│       ├── logger.py             # Rotating file + console logger
│       └── config.py             # Settings (env vars / .env / defaults)
│
├── tests/
│   ├── unit/
│   │   ├── test_data.py          # DataLoader & Splitter — 8 tests
│   │   ├── test_features.py      # Preprocessor — 5 tests
│   │   └── test_models.py        # Trainer & Evaluator — 7 tests
│   └── integration/
│       └── test_api.py           # API endpoints — 9 tests
│
├── scripts/
│   ├── train.py                  # CLI: python scripts/train.py
│   └── predict.py                # CLI: batch CSV prediction
│
├── config/
│   └── settings.yaml             # Hyperparameters & paths
│
├── models/
│   └── saved/                    # .pkl files + manifest.json
│       ├── random_forest.pkl
│       ├── knn.pkl
│       ├── naive_bayes.pkl
│       ├── svm.pkl
│       ├── logistic_regression.pkl
│       ├── ensemble_soft_voting.pkl
│       └── manifest.json
│
├── docs/
│   └── images/                   # Result plots (add yours here)
│       ├── 01_performance_metrics.png
│       ├── 02_confusion_matrices.png
│       ├── 03_roc_curves.png
│       └── 04_summary_heatmap.png
│
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── README.md
```

---

## 🧠 Model Architecture

```
Wisconsin Breast Cancer Dataset (569 samples, 30 features)
                        │
              ┌─────────▼──────────┐
              │   StandardScaler   │  ← inside every Pipeline
              └─────────┬──────────┘
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
   Random Forest       kNN        Naive Bayes
   (72 estimators)    (k=5)      (GaussianNB)
          │             │              │
          └──────┬───────┘              │
                 │       ┌─────────────┘
                 ▼       ▼
              SVM     Logistic Regression
           (RBF, C=1)  (lbfgs, L2)
                 │       │
                 └───┬───┘
                     ▼
          ┌──────────────────────┐
          │  Soft-Voting Ensemble │  ← averages all 5 probabilities
          └──────────┬───────────┘
                     ▼
          Prediction + Confidence Score
```

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | scikit-learn 1.4+ |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Persistence | joblib |
| Testing | pytest + httpx |
| Config | python-dotenv + PyYAML |
| Logging | Python logging (rotating file) |
| Visualisation | matplotlib + seaborn |

---

## 🗂️ Dataset

**Wisconsin Diagnosis Breast Cancer (WDBC)**

| Property | Value |
|---|---|
| Source | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) |
| Samples | 569 |
| Features | 30 numeric (cell nucleus measurements) |
| Classes | Benign (357) · Malignant (212) |
| Missing values | None |
| Train split | 398 samples |
| Test split | 171 samples |

Features are computed from digitised images of fine needle aspirate (FNA) of breast masses, describing characteristics of cell nuclei: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension — each measured as mean, standard error, and worst value.

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

```ini
APP_ENV=production
DEFAULT_MODEL=ensemble_soft_voting
TEST_SIZE=0.30
CV_FOLDS=10
API_PORT=8000
```

All hyperparameters live in `config/settings.yaml` and can be overridden via environment variables.

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 📤 Kaggle — Download Results as ZIP

After running the Kaggle notebook, add this final cell to download all outputs:

```python
import os, zipfile
from IPython.display import FileLink, display

ZIP_PATH = "/kaggle/working/breast_cancer_ml_results.zip"
FILES = [
    "01_performance_metrics.png", "02_confusion_matrices.png",
    "03_roc_curves.png", "04_summary_heatmap.png",
    "model_results.csv", "ensemble_soft_voting.pkl",
    "random_forest.pkl", "knn.pkl", "naive_bayes.pkl",
    "svm.pkl", "logistic_regression.pkl",
]

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in FILES:
        p = f"/kaggle/working/{f}"
        if os.path.exists(p):
            zf.write(p, arcname=f)

display(FileLink(ZIP_PATH, result_html_prefix="⬇️ Download: "))
```

---

## 🚀 Deploy to Hugging Face Spaces

```bash
# 1. Install HF CLI
pip install huggingface_hub

# 2. Login
huggingface-cli login

# 3. Create a new Space (FastAPI type)
huggingface-cli repo create breast-cancer-ml --type space --space_sdk docker

# 4. Push
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/breast-cancer-ml
git push hf main
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use this project in research, please cite the original paper:

```bibtex
@inproceedings{ctems2018breastcancer,
  title     = {Comparison of Machine Learning Algorithms for Breast Cancer Prediction},
  booktitle = {2018 International Conference on Computational Techniques,
               Electronics and Mechanical Systems (CTEMS)},
  year      = {2018},
  publisher = {IEEE},
  pages     = {114--118},
  isbn      = {978-1-5386-7709-4}
}
```

---

## 🙏 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu) — WDBC dataset
- [scikit-learn](https://scikit-learn.org) — ML algorithms
- [FastAPI](https://fastapi.tiangolo.com) — API framework
- Original paper authors — UPES & Amity University

---

<div align="center">

Made with ❤️ for early cancer detection research

⭐ **Star this repo** if it helped you!

</div>
