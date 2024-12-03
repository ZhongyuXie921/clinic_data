import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from lifelines import KaplanMeierFitter
import warnings
import os
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn')
sns.set_palette("husl")

# Create directory for saving plots if it doesn't exist
SAVE_DIR = 'clinic_data'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 1. Data Loading and Preprocessing
def load_and_preprocess_data():
    """
    Load and preprocess cancer patient data
    """
    # Simulate dataset creation
    np.random.seed(42)
    n_patients = 1000
    
    data = {
        'patient_id': range(1, n_patients + 1),
        'age': np.random.normal(60, 15, n_patients).clip(20, 90),
        'gender': np.random.choice(['Male', 'Female'], n_patients),
        'cancer_type': np.random.choice(['Lung', 'Breast', 'Colon', 'Prostate'], n_patients),
        'stage': np.random.choice(['I', 'II', 'III', 'IV'], n_patients),
        'tumor_size': np.random.normal(3, 1.5, n_patients).clip(0.1, 10),
        'metastasis': np.random.choice(['Yes', 'No'], n_patients),
        'treatment': np.random.choice(['Surgery', 'Chemotherapy', 'Radiation', 'Combined'], n_patients),
        'survival_months': np.random.exponential(36, n_patients).clip(1, 60),
        'survival_status': np.random.choice(['Alive', 'Deceased'], n_patients),
        'smoking': np.random.choice(['Yes', 'No', 'Former'], n_patients),
        'comorbidity': np.random.choice(['None', 'Diabetes', 'Hypertension', 'Multiple'], n_patients)
    }
    
    df = pd.DataFrame(data)
    
    # Add correlations
    # Late-stage patients have shorter survival
    mask_late_stage = df['stage'].isin(['III', 'IV'])
    df.loc[mask_late_stage, 'survival_months'] *= 0.7
    
    # Patients with metastasis have shorter survival
    mask_metastasis = df['metastasis'] == 'Yes'
    df.loc[mask_metastasis, 'survival_months'] *= 0.6
    
    return df

# 2. Exploratory Data Analysis
def perform_eda(df):
    """
    Perform exploratory data analysis
    """
    # 2.1 Age distribution analysis
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='age', hue='gender', multiple="stack")
    plt.title('Age Distribution by Gender')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig(os.path.join(SAVE_DIR, 'age_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2.2 Cancer type distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='cancer_type', hue='stage')
    plt.title('Cancer Type Distribution by Stage')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(SAVE_DIR, 'cancer_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2.3 Relationship between survival and tumor size
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='tumor_size', y='survival_months', 
                    hue='metastasis', size='age', sizes=(20, 200))
    plt.title('Survival Months vs Tumor Size')
    plt.savefig(os.path.join(SAVE_DIR, 'survival_vs_tumor.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2.4 Survival boxplot by treatment type
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='treatment', y='survival_months')
    plt.title('Survival Months by Treatment Type')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(SAVE_DIR, 'survival_by_treatment.png'), bbox_inches='tight', dpi=300)
    plt.close()

# 3. Survival Analysis
def survival_analysis(df):
    """
    Perform survival analysis
    """
    kmf = KaplanMeierFitter()
    
    # 3.1 Survival analysis by cancer stage
    plt.figure(figsize=(12, 8))
    stages = df['stage'].unique()
    
    for stage in stages:
        mask = df['stage'] == stage
        kmf.fit(df[mask]['survival_months'],
                event_observed=(df[mask]['survival_status'] == 'Deceased'),
                label=f'Stage {stage}')
        kmf.plot()
    
    plt.title('Kaplan-Meier Survival Curves by Cancer Stage')
    plt.xlabel('Months')
    plt.ylabel('Survival Probability')
    plt.savefig(os.path.join(SAVE_DIR, 'survival_curves.png'), bbox_inches='tight', dpi=300)
    plt.close()

# 4. Prediction Model Building
def build_prediction_model(df):
    """
    Build survival prediction model
    """
    # 4.1 Prepare features
    features = ['age', 'tumor_size']
    categorical_features = ['gender', 'cancer_type', 'stage', 'metastasis', 
                          'treatment', 'smoking', 'comorbidity']
    
    # One-hot encode categorical features
    X = pd.get_dummies(df[features + categorical_features])
    y = (df['survival_status'] == 'Deceased').astype(int)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize numerical features
    scaler = StandardScaler()
    numerical_cols = ['age', 'tumor_size']
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = rf_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.savefig(os.path.join(SAVE_DIR, 'feature_importance.png'), bbox_inches='tight', dpi=300)
    plt.close()

# Main function
def main():
    # Load data
    df = load_and_preprocess_data()
    
    # Print basic information
    print("Dataset Basic Information:")
    print(df.info())
    print("\nBasic Statistical Description:")
    print(df.describe())
    
    # Perform analysis
    perform_eda(df)
    survival_analysis(df)
    build_prediction_model(df)

if __name__ == "__main__":
    main()