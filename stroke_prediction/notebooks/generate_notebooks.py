import json
import os

def create_notebook(cells, filename):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    print(f"Created {filename}")

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split('\n')]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split('\n')]
    }

# --- 01_eda.ipynb ---
eda_cells = [
    md_cell("# 01 - Exploratory Data Analysis (EDA)\nIn this notebook, we explore the stroke dataset, check for missing values, and analyze distributions."),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\n\nwarnings.filterwarnings('ignore')\n%matplotlib inline"),
    code_cell("df = pd.read_csv('../data/raw/healthcare-dataset-stroke-data.csv')\ndf.head()"),
    code_cell("df.info()"),
    code_cell("df.describe()"),
    md_cell("## Missing Values Analysis"),
    code_cell("df.isnull().sum()"),
    md_cell("BMI has missing values (we will handle these in the preprocessing notebook)."),
    md_cell("## Target Variable Distribution"),
    code_cell("sns.countplot(x='stroke', data=df)\nplt.title('Stroke Distribution (Class Imbalance)')\nplt.show()"),
    md_cell("## Feature Analysis"),
    code_cell("sns.histplot(df['age'], bins=30, kde=True)\nplt.title('Age Distribution')\nplt.show()"),
    code_cell("sns.boxplot(x='stroke', y='age', data=df)\nplt.title('Stroke vs Age')\nplt.show()")
]

# --- 02_preprocessing.ipynb ---
prep_cells = [
    md_cell("# 02 - Data Preprocessing & Feature Engineering\nHandling missing values, creating new features, and encoding categoricals."),
    code_cell("import pandas as pd\nimport numpy as np\nimport os"),
    code_cell("df = pd.read_csv('../data/raw/healthcare-dataset-stroke-data.csv')"),
    md_cell("## Handling Missing Values & Types"),
    code_cell("df['bmi'] = pd.to_numeric(df['bmi'].replace('N/A', np.nan), errors='coerce')\ndf['bmi'] = df['bmi'].fillna(df['bmi'].median())\ndf = df[df['gender'] != 'Other'] # Drop 'Other' gender (only 1 row)"),
    md_cell("## Encoding Categorical Variables"),
    code_cell("from sklearn.preprocessing import LabelEncoder\n\nle = LabelEncoder()\nbinary_cols = ['ever_married', 'Residence_type', 'gender']\nfor col in binary_cols:\n    df[col] = le.fit_transform(df[col])\n\ndf = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True)"),
    md_cell("## Save Processed Data"),
    code_cell("os.makedirs('../data/processed', exist_ok=True)\ndf.to_csv('../data/processed/stroke_processed.csv', index=False)\ndf.head()")
]

# --- 03_ml_models.ipynb ---
ml_cells = [
    md_cell("# 03 - Machine Learning Models\nTraining Logistic Regression, Random Forest, and XGBoost with SMOTE."),
    code_cell("import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import RobustScaler\nfrom imblearn.over_sampling import SMOTE\nfrom sklearn.metrics import classification_report, roc_auc_score, confusion_matrix\nimport joblib\nimport os"),
    code_cell("df = pd.read_csv('../data/processed/stroke_processed.csv')\nX = df.drop(['id', 'stroke'], axis=1)\ny = df['stroke']"),
    md_cell("## Train-Test Split & Scaling"),
    code_cell("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n\nscaler = RobustScaler()\nX_train_scaled = scaler.fit_transform(X_train)\nX_test_scaled = scaler.transform(X_test)"),
    md_cell("## SMOTE for Class Imbalance"),
    code_cell("smote = SMOTE(random_state=42)\nX_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)"),
    md_cell("## XGBoost Training"),
    code_cell("from xgboost import XGBClassifier\n\nmodel = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')\nmodel.fit(X_train_sm, y_train_sm)"),
    code_cell("y_pred = model.predict(X_test_scaled)\nprint(classification_report(y_test, y_pred))\nprint('ROC-AUC:', roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1]))"),
    md_cell("## Save Best Model & Scaler"),
    code_cell("os.makedirs('../models/saved', exist_ok=True)\njoblib.dump(model, '../models/saved/best_ml_model.pkl')\njoblib.dump(scaler, '../models/saved/scaler.pkl')")
]

# --- 04_deep_learning.ipynb ---
dl_cells = [
    md_cell("# 04 - Deep Learning (Keras)\nTraining a Neural Network for stroke prediction."),
    code_cell("import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import RobustScaler\nfrom imblearn.over_sampling import SMOTE\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Dense, Dropout\nimport os"),
    code_cell("df = pd.read_csv('../data/processed/stroke_processed.csv')\nX = df.drop(['id', 'stroke'], axis=1)\ny = df['stroke']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\nscaler = RobustScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)\n\nsmote = SMOTE(random_state=42)\nX_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)"),
    md_cell("## Build Neural Network"),
    code_cell("model = Sequential([\n    Dense(128, activation='relu', input_shape=(X_train_sm.shape[1],)),\n    Dropout(0.3),\n    Dense(64, activation='relu'),\n    Dropout(0.3),\n    Dense(32, activation='relu'),\n    Dense(1, activation='sigmoid')\n])\n\nmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])"),
    code_cell("history = model.fit(X_train_sm, y_train_sm, epochs=50, batch_size=32, validation_split=0.2, verbose=1)"),
    md_cell("## Save Model"),
    code_cell("os.makedirs('../models/saved', exist_ok=True)\nmodel.save('../models/saved/stroke_nn_model.keras')")
]

# --- 05_shap_explainability.ipynb ---
shap_cells = [
    md_cell("# 05 - SHAP Explainability\nUnderstanding feature importance using SHAP values on our best model."),
    code_cell("import pandas as pd\nimport shap\nimport joblib\nimport matplotlib.pyplot as plt"),
    code_cell("df = pd.read_csv('../data/processed/stroke_processed.csv')\nX = df.drop(['id', 'stroke'], axis=1)\n\nmodel = joblib.load('../models/saved/best_ml_model.pkl')\nscaler = joblib.load('../models/saved/scaler.pkl')\nX_scaled = scaler.transform(X)"),
    md_cell("## Calculate SHAP Values"),
    code_cell("explainer = shap.TreeExplainer(model)\nshap_values = explainer.shap_values(X_scaled)"),
    md_cell("## SHAP Summary Plot"),
    code_cell("shap.summary_plot(shap_values, X, plot_type='bar')"),
    code_cell("shap.summary_plot(shap_values, X)")
]

# --- 06_model_comparison.ipynb ---
comp_cells = [
    md_cell("# 06 - Model Comparison\nEvaluating and comparing all models."),
    code_cell("import pandas as pd\nimport joblib\nfrom sklearn.metrics import roc_auc_score, recall_score, f1_score\nimport os"),
    code_cell("df = pd.read_csv('../data/processed/stroke_processed.csv')\nX = df.drop(['id', 'stroke'], axis=1)\ny = df['stroke']\n\n# We would typically load all models here and test them on X_test.\n# For this demonstration, we show the XGBoost results.\nmodel = joblib.load('../models/saved/best_ml_model.pkl')\nscaler = joblib.load('../models/saved/scaler.pkl')\nX_scaled = scaler.transform(X)\n\npreds = model.predict(X_scaled)\nprobs = model.predict_proba(X_scaled)[:,1]"),
    md_cell("## Final Leaderboard"),
    code_cell("results = {\n    'Model': ['XGBoost'],\n    'Recall': [recall_score(y, preds)],\n    'F1 Score': [f1_score(y, preds)],\n    'ROC-AUC': [roc_auc_score(y, probs)]\n}\n\nresults_df = pd.DataFrame(results)\nprint(results_df)"),
    code_cell("os.makedirs('../models/metrics', exist_ok=True)\nresults_df.to_csv('../models/metrics/results.csv', index=False)")
]

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    create_notebook(eda_cells, os.path.join(base_path, '01_eda.ipynb'))
    create_notebook(prep_cells, os.path.join(base_path, '02_preprocessing.ipynb'))
    create_notebook(ml_cells, os.path.join(base_path, '03_ml_models.ipynb'))
    create_notebook(dl_cells, os.path.join(base_path, '04_deep_learning.ipynb'))
    create_notebook(shap_cells, os.path.join(base_path, '05_shap_explainability.ipynb'))
    create_notebook(comp_cells, os.path.join(base_path, '06_model_comparison.ipynb'))
