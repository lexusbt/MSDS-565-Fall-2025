# Credit Card Default Prediction Assignment

## Assignment Overview

This assignment focuses on predicting credit card defaults using a **cost-aware, fair, and explainable** machine learning approach. This README covers the data preparation and feature engineering phase.

## Dataset

**Source**: [Default of Credit Card Clients Dataset](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset) from Kaggle

**Description**: The dataset contains information on default payments, demographic factors, credit data, payment history, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

**Size**: 30,000 records with 24 features

## Files

- `Credit_Card_Default_Data_Prep.ipynb` - Main notebook for data preparation and feature engineering
- `README.md` - This file
- `UCI_Credit_Card.csv` - Raw dataset (you need to download this)
- `credit_card_default_prepared.csv` - Prepared dataset (generated after running notebook)
- `feature_dictionary.csv` - Feature documentation (generated after running notebook)

## Setup Instructions

### 1. Download the Dataset

You have two options:

**Option A: Manual Download**
1. Visit https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
2. Download `UCI_Credit_Card.csv`
3. Place it in this directory (`Assignment - Credit Card Default Prediction/`)

**Option B: Kaggle API** (requires Kaggle account and API setup)
```bash
pip install kaggle
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset
unzip default-of-credit-card-clients-dataset.zip
```

### 2. Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

### 3. Run the Notebook

```bash
jupyter notebook Credit_Card_Default_Data_Prep.ipynb
```

## What's in the Notebook

### 1. **Setup and Imports**
   - Import necessary libraries
   - Configure visualization settings

### 2. **Data Loading**
   - Load the dataset
   - Initial data exploration

### 3. **Initial Data Exploration**
   - View dataset structure
   - Statistical summaries
   - Feature descriptions

### 4. **Data Quality Assessment**
   - Check for missing values
   - Identify duplicate records
   - Analyze target variable distribution
   - Detect class imbalance

### 5. **Data Cleaning and Preprocessing**
   - Clean categorical variables
   - Handle inconsistent values
   - Detect outliers

### 6. **Exploratory Data Analysis (EDA)**
   - **Demographic Analysis**: Age, gender, education, marital status
   - **Credit Limit Analysis**: Distribution and relationship with default
   - **Payment History Analysis**: Payment status patterns
   - **Bill and Payment Amount Analysis**: Spending and payment behavior
   - **Correlation Analysis**: Feature relationships

### 7. **Feature Engineering**
   Created 30+ new features including:

   **Payment Behavior Features**:
   - Average payment status
   - Maximum payment delay
   - Number of delayed/on-time payments
   - Payment status trend

   **Credit Utilization Features**:
   - Average/max bill amount
   - Bill amount volatility
   - Credit utilization ratio
   - Recent credit utilization

   **Payment Capacity Features**:
   - Average payment amount
   - Payment to bill ratio
   - Number of zero payments
   - Payment consistency

   **Demographic Features**:
   - One-hot encoded gender, education, marriage
   - Age groups
   - Credit per age ratio

   **Interaction Features**:
   - High utilization + delayed payments
   - Low payment ratio + high bills

### 8. **Fairness Analysis Preparation**
   - Analyze default rates across protected groups
   - Statistical significance testing
   - Disparate impact analysis (80% rule)
   - Identify potential fairness concerns

### 9. **Prepare Final Dataset**
   - Select features for modeling
   - Create clean, modeling-ready dataset
   - Quality checks

### 10. **Save Outputs**
   - Save prepared dataset
   - Create feature dictionary

## Key Findings

### Dataset Characteristics
- **Total Records**: 30,000
- **Default Rate**: ~22% (imbalanced dataset)
- **Features**: 24 original features → 50+ features after engineering
- **No Missing Values**: Clean dataset

### Important Patterns Discovered
1. **Payment history** is the strongest predictor of default
2. **Credit utilization** shows significant correlation with default risk
3. **Younger age groups** tend to have slightly higher default rates
4. **Education level** shows inverse relationship with default (higher education = lower default)
5. **Class imbalance** requires special handling (SMOTE, class weights, etc.)

### Fairness Considerations
- Protected attributes identified: Gender, Age, Education, Marital Status
- Some disparate impact detected across demographic groups
- Will require fairness-aware modeling approaches

## Next Steps

After completing data preparation, the next phases will include:

1. **Baseline Models**
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Neural Networks

2. **Cost-Sensitive Learning**
   - Define cost matrix (false negatives more costly than false positives)
   - Implement cost-sensitive algorithms
   - Adjust decision thresholds

3. **Fairness-Aware Modeling**
   - Apply fairness constraints
   - Use fairness metrics (demographic parity, equalized odds)
   - Implement bias mitigation techniques

4. **Explainability**
   - SHAP values
   - LIME
   - Feature importance analysis
   - Decision tree visualization

5. **Evaluation**
   - Standard metrics: Accuracy, Precision, Recall, F1, AUC
   - Cost-based metrics: Expected cost, cost-sensitive accuracy
   - Fairness metrics: Disparate impact, statistical parity difference
   - Business impact analysis

## References

- **Dataset Citation**: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

- **Fairness in ML**:
  - Mehrabi, N., et al. (2021). A Survey on Bias and Fairness in Machine Learning
  - Barocas, S., & Selbst, A. D. (2016). Big data's disparate impact

- **Cost-Sensitive Learning**:
  - Elkan, C. (2001). The foundations of cost-sensitive learning
  - Ling, C. X., & Sheng, V. S. (2008). Cost-sensitive learning

## Learning Objectives Achieved

✅ Understand credit card default data characteristics
✅ Perform thorough data quality assessment
✅ Create meaningful features for ML modeling
✅ Prepare data for fair and explainable ML approaches
✅ Analyze potential fairness concerns in financial data

## Tips for Beginners

1. **Run cells sequentially**: The notebook is designed to be run from top to bottom
2. **Read the markdown cells**: They provide context and explanations
3. **Examine the visualizations**: They reveal important patterns
4. **Understand each feature**: Review the feature dictionary
5. **Consider the business context**: Default prediction has real-world implications
6. **Think about fairness**: Financial ML must be fair and non-discriminatory

## Questions or Issues?

If you encounter any issues:
1. Make sure the dataset is in the correct location
2. Verify all libraries are installed
3. Check Python version (3.7+ recommended)
4. Review error messages carefully

---

**Happy Learning!** 🎓
