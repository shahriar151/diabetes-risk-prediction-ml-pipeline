# 🏥 Diabetes Risk Prediction - End-to-End ML Pipeline

An end-to-end machine learning project for predicting diabetes risk, featuring data preprocessing, multiple ML models (Logistic Regression, Random Forest, SVM), hyperparameter tuning, and an interactive Streamlit dashboard.

## 📋 Project Overview

This project demonstrates a complete ML pipeline from data preprocessing to deployment:

- **Data Processing**: Missing value imputation, outlier detection, feature scaling
- **Dimensionality Reduction**: Principal Component Analysis (PCA)
- **Feature Selection**: SelectKBest and Recursive Feature Elimination (RFE)
- **Unsupervised Learning**: K-Means clustering for patient segmentation
- **Supervised Learning**: Three models with hyperparameter tuning
  - Logistic Regression (interpretability)
  - Random Forest (ensemble learning)
  - Support Vector Machine (non-linear boundaries)
- **Interactive Dashboard**: Streamlit web app for real-time predictions

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.9+
- Cursor IDE
- UV package manager

### Installation

1. **Clone/Create Project Directory**

```bash
mkdir diabetes-ml-pipeline
cd diabetes-ml-pipeline
```

2. **Create Project Structure**

```bash
# Create directories
mkdir data models notebooks src

# Create files
touch requirements.txt app.py README.md .gitignore
touch src/utils.py
touch notebooks/01_preprocessing_and_training.ipynb
```

3. **Install Dependencies with UV**

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install packages
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install from requirements.txt
uv pip install -r requirements.txt
```

### Setup Steps

**Step 1: Generate Dataset**

- Use the dataset generator artifact I provided earlier
- Download the CSV file
- Save it as `data/diabetes_risk_data_300.csv`

**Step 2: Train Models**

Open the Jupyter notebook in Cursor:

```bash
# Start Jupyter
jupyter notebook notebooks/01_preprocessing_and_training.ipynb

# Or use Jupyter Lab
jupyter lab
```

Run all cells in the notebook. This will:
- Load and preprocess the data
- Perform EDA and feature engineering
- Train and evaluate all three models
- Save model artifacts to `models/` directory

Expected output files in `models/`:
- `best_model.pkl`
- `logistic_regression_model.pkl`
- `random_forest_model.pkl`
- `svm_model.pkl`
- `scaler.pkl`
- `imputer.pkl`
- `feature_names.pkl`
- Various visualization PNG files

**Step 3: Run Streamlit Dashboard**

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
diabetes-ml-pipeline/
├── data/
│   └── diabetes_risk_data_300.csv          # Dataset
├── models/
│   ├── best_model.pkl                      # Best performing model
│   ├── logistic_regression_model.pkl       # LR model
│   ├── random_forest_model.pkl             # RF model
│   ├── svm_model.pkl                       # SVM model
│   ├── scaler.pkl                          # Feature scaler
│   ├── imputer.pkl                         # Missing value imputer
│   ├── feature_names.pkl                   # Feature list
│   └── *.png                               # Visualizations
├── notebooks/
│   └── 01_preprocessing_and_training.ipynb # Training notebook
├── src/
│   └── utils.py                            # Helper functions
├── app.py                                  # Streamlit dashboard
├── requirements.txt                        # Dependencies
├── README.md                               # This file
└── .gitignore                              # Git ignore rules
```

## 🎯 Features

### Dataset Features

| Feature | Description | Range |
|---------|-------------|-------|
| Age | Patient age | 20-80 years |
| BMI | Body Mass Index | 18-38 |
| Glucose | Blood glucose level | 60-200 mg/dL |
| BloodPressure | Diastolic BP | 60-120 mmHg |
| Insulin | Serum insulin | 15-615 µU/ml |
| SkinThickness | Triceps skin fold | 10-60 mm |
| Pregnancies | Number of pregnancies | 0-9 |
| DiabetesPedigreeFunction | Genetic risk score | 0-2 |

**Target**: `Outcome` (0 = No diabetes, 1 = Has diabetes)

### ML Pipeline Components

**1. Data Preprocessing**
- Missing value imputation (mean strategy)
- Outlier detection with boxplots
- Feature scaling with StandardScaler
- Train-test split (80-20, stratified)

**2. Exploratory Data Analysis**
- Distribution plots for all features
- Correlation heatmap
- Missing value analysis
- Class balance visualization

**3. Dimensionality Reduction**
- PCA for variance analysis
- Scree plot visualization
- Cumulative variance explained

**4. Feature Selection**
- SelectKBest with ANOVA F-statistic
- Recursive Feature Elimination (RFE)
- Feature importance ranking

**5. Unsupervised Learning**
- K-Means clustering (elbow method)
- Patient segmentation analysis
- Cluster visualization with PCA

**6. Model Training & Evaluation**
- Logistic Regression with GridSearchCV
- Random Forest with hyperparameter tuning
- SVM (RBF and linear kernels)
- 5-fold cross-validation
- ROC-AUC scoring

**7. Model Comparison**
- Accuracy, Precision, Recall, F1-Score
- ROC curves comparison
- Confusion matrices
- Best model selection

### Streamlit Dashboard Features

- **Manual Input Mode**
  - Individual patient data entry
  - Real-time risk prediction
  - Interactive gauge chart
  - Feature value visualization
  - Risk recommendations

- **Batch Upload Mode**
  - CSV file upload
  - Batch predictions
  - Results summary statistics
  - Risk distribution histogram
  - Downloadable results

- **Model Selection**
  - Switch between LR, RF, and SVM
  - Compare predictions across models

## 📊 Model Performance

Expected performance on test set (synthetic data):

| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|----------|
| Logistic Regression | ~75-80% | ~0.80-0.85 | ~0.75-0.80 |
| Random Forest | ~80-85% | ~0.85-0.90 | ~0.80-0.85 |
| SVM | ~75-82% | ~0.82-0.87 | ~0.75-0.82 |

*Note: Performance will vary based on the random data generation.*

## 🔧 Troubleshooting

**Issue: ModuleNotFoundError**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall packages
uv pip install -r requirements.txt
```

**Issue: Model files not found**
```bash
# Run the training notebook first
jupyter notebook notebooks/01_preprocessing_and_training.ipynb
# Execute all cells
```

**Issue: Streamlit port already in use**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

**Issue: Out of memory (on your AMD Ryzen 5)**
```bash
# Reduce dataset size in the generator (use 200-250 samples)
# Or reduce GridSearchCV cv folds from 5 to 3
```

## 🎨 Customization

### Change Dataset Size
Modify the data generator to create 200-500 samples based on your needs.

### Adjust Model Hyperparameters
Edit the parameter grids in the notebook:

```python
# Example: Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],  # Add more values
    'max_depth': [5, 10, 15, None],
    # Add more parameters
}
```

### Modify Dashboard Styling
Edit the CSS in `app.py`:

```python
st.markdown("""
<style>
    /* Your custom styles */
</style>
""", unsafe_allow_html=True)
```

## 📝 Using for Freelancing

### Portfolio Presentation

**Project Title**: "End-to-End ML Pipeline for Healthcare Risk Prediction"

**Key Highlights**:
- ✅ Complete data preprocessing pipeline
- ✅ Multiple ML models with hyperparameter tuning
- ✅ Interactive web dashboard with Streamlit
- ✅ Professional visualizations
- ✅ Clean, documented code
- ✅ Model versioning and persistence

**Fiverr Gig Ideas**:
1. "I will build a complete ML pipeline with Streamlit dashboard"
2. "I will create healthcare prediction models with web interface"
3. "I will develop end-to-end data science projects"

### Demonstration Tips

1. **Show the process**: Walk through the notebook step-by-step
2. **Live demo**: Use the Streamlit app for real-time predictions
3. **Highlight technical skills**: Mention GridSearchCV, PCA, model comparison
4. **Emphasize deployment**: Show how easy it is to use the app

### Add to Portfolio

```markdown
## Diabetes Risk Prediction System

**Tech Stack**: Python, scikit-learn, Streamlit, Pandas, Plotly

**Features**:
- Trained 3 ML models (LR, RF, SVM) with 80%+ accuracy
- Built interactive dashboard for real-time predictions
- Implemented complete preprocessing pipeline
- Achieved 0.85+ ROC-AUC score on test data

**GitHub**: [Your repo link]
**Live Demo**: [Streamlit Cloud link - optional]
```

## ⚠️ Important Notes

**Disclaimer**: This project uses synthetic data for educational and portfolio purposes only. It should NOT be used for actual medical diagnosis or treatment decisions.

**Real-World Application**: For production medical ML systems, you would need:
- Real, validated patient data
- Medical expert consultation
- Regulatory approval (FDA, etc.)
- HIPAA compliance
- Extensive testing and validation

## 🤝 Contributing

This is a personal portfolio project, but suggestions are welcome!

## 📧 Contact

**Developer**: Shahriar
**Purpose**: Educational & Portfolio Demonstration

---

**Built with**: Python 🐍 | scikit-learn 🤖 | Streamlit 🎈 | Plotly 📊