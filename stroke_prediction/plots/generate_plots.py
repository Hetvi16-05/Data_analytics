import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import shap

warnings.filterwarnings('ignore')

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Ensure directories exist
    os.makedirs(os.path.join(base_dir, 'plots', 'eda'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'plots', 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'plots', 'shap'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'data', 'processed'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'models', 'saved'), exist_ok=True)

    print("Loading raw data...")
    raw_path = os.path.join(base_dir, 'data', 'raw', 'healthcare-dataset-stroke-data.csv')
    df = pd.read_csv(raw_path)

    print("Generating EDA plots...")
    # 1. Age Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'].dropna(), bins=30, kde=True)
    plt.title('Age Distribution')
    plt.savefig(os.path.join(base_dir, 'plots', 'eda', 'age_distribution.png'))
    plt.close()

    # 2. Stroke Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='stroke', data=df)
    plt.title('Stroke Distribution (Class Imbalance)')
    plt.savefig(os.path.join(base_dir, 'plots', 'eda', 'stroke_distribution.png'))
    plt.close()

    # 3. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['id'], errors='ignore')
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Heatmap')
    plt.savefig(os.path.join(base_dir, 'plots', 'eda', 'correlation_heatmap.png'))
    plt.close()

    print("Preprocessing data...")
    # Clean BMI and drop Other gender
    df['bmi'] = pd.to_numeric(df['bmi'].replace('N/A', np.nan), errors='coerce')
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    df = df[df['gender'] != 'Other']

    # Encoding
    le = LabelEncoder()
    binary_cols = ['ever_married', 'Residence_type', 'gender']
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
    
    df = pd.get_dummies(df, columns=['work_type', 'smoking_status'], drop_first=True)
    
    # Save processed data
    processed_path = os.path.join(base_dir, 'data', 'processed', 'stroke_processed.csv')
    df.to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")

    print("Training XGBoost Model...")
    X = df.drop(['id', 'stroke'], axis=1, errors='ignore')
    y = df['stroke']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_sm, y_train_sm)

    # Save Model and Scaler
    model_path = os.path.join(base_dir, 'models', 'saved', 'best_ml_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'saved', 'scaler.pkl')
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model and Scaler saved.")

    print("Generating Model Evaluation plots...")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Stroke', 'Stroke'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(base_dir, 'plots', 'models', 'confusion_matrix.png'))
    plt.close('all')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(base_dir, 'plots', 'models', 'roc_curve.png'))
    plt.close()

    print("Generating SHAP Explainability plots...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    # SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
    plt.savefig(os.path.join(base_dir, 'plots', 'shap', 'shap_summary_plot.png'), bbox_inches='tight')
    plt.close()

    # SHAP Bar Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar", show=False)
    plt.savefig(os.path.join(base_dir, 'plots', 'shap', 'shap_bar_plot.png'), bbox_inches='tight')
    plt.close()

    print("All plots generated successfully!")

if __name__ == '__main__':
    main()
