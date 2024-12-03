# Cancer Patient Data Analysis and Prediction

## Overview
This project provides a comprehensive analysis toolkit for cancer patient data, including exploratory data analysis (EDA), survival analysis, and predictive modeling. It generates synthetic patient data and performs various analyses to understand patterns in cancer progression, treatment outcomes, and survival rates.

## Features
- Synthetic cancer patient data generation with realistic correlations
- Exploratory Data Analysis (EDA) with visualizations:
  - Age distribution by gender
  - Cancer type distribution by stage
  - Survival analysis by tumor size
  - Treatment outcome analysis
- Kaplan-Meier survival analysis
- Machine learning-based survival prediction using Random Forest
- Automated visualization generation and storage

## Project Structure
clinic_data/
│
├── age_distribution.png
├── cancer_distribution.png
├── survival_vs_tumor.png
├── survival_by_treatment.png
├── survival_curves.png
└── feature_importance.png

## Requirements
- Python 3.7+
- Required packages:

pandas
numpy
matplotlib
seaborn
scikit-learn
lifelines

## Installation
1. Clone the repository:
 ```bash
 git clone https://github.com/yourusername/cancer-patient-analysis.git
 cd cancer-patient-analysis
```
2. Run the main script:
```   
python clinic_data.py
```
The script will:

Generate synthetic patient data
Perform exploratory data analysis
Generate survival analysis plots
Build and evaluate a prediction model
Save all visualizations in the clinic_data directory

## Visualizations
The project generates several visualizations:

### Age Distribution
![Age Distribution](clinic_data/age_distribution.png)
*Age distribution across different genders*

### Cancer Type Distribution
![Cancer Distribution](clinic_data/cancer_distribution.png)
*Cancer type distribution across different stages*

### Survival Analysis
![Survival vs Tumor](clinic_data/survival_vs_tumor.png)
*Relationship between tumor size and survival*

![Survival by Treatment](clinic_data/survival_by_treatment.png)
*Box plot of survival months by treatment type*

### Survival Curves
![Survival Curves](clinic_data/survival_curves.png)
*Kaplan-Meier survival curves by cancer stage*

### Feature Importance
![Feature Importance](clinic_data/feature_importance.png)
*Top 10 most important features in the prediction model*

Model Details
The prediction model uses Random Forest Classification to predict patient survival status based on various features including:

Demographic information (age, gender)
Clinical features (cancer type, stage, tumor size)
Treatment information
Lifestyle factors (smoking status)
Comorbidity information

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact
Your Name - your.email@example.com
Project Link: https://github.com/yourusername/cancer-patient-analysis


