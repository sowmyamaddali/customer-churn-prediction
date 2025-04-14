# Customer Churn Prediction
An end-to-end machine learning project to predict customer churn using the Telco dataset from Kaggle.


## Dataset
- **Source:** [Telco Customer Churn (11.1.3+)](https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3)
- Contains information about a fictional telco company that provided home phone and internet services to 7043 customers in California in Q3. It indicates which customers have left, stayed, or signed up for their service.


## Exploratory Data Analysis (EDA)
Performed exploratory data analysis `churn_EDA.ipynb` to understand customer behavior patterns leading to churn. Key insights include:
1. Class Imbalance:
   - 73.5% customers did not churn (No)
   - 26.5% customers churned (Yes)
2. Demographics & Churn:
   - Senior citizens have a higher likelihood of churn
   - Customers with dependents are less likely to churn
   - Gender has no major impact on churn
3. Behavioral Patterns:
   - Month-to-month contract users churn more often than those with 1 or 2-year contracts
   - Customers using Bank Withdrawal as payment method churn more than others
   - Users with low tenure and low total charges churn more frequently
   - Fiber optic internet users have a higher churn rate compared to DSL or Cable


## Data Cleaning & Preprocessing
- Dropped irrelevant or redundant features like `customer_id`, `under_30`, `quarter`, `churn_category`, and `churn_reason` in `churn_DataPreProcessing.ipynb`
- Converted all binary features (Yes/No) and gender (Male/Female) to 1/0
- One-hot encoded multi-class categorical features:
    - internet_type
    - contract
    - payment_method
- Created two separate datasets:
    - `linear_data.csv`: for linear models (one-hot encoded with `drop_first=True)
    - `tree_data.csv`: for tree-based models (full one-hot encoding)


## Handling Class Imbalance
- Observed class imbalance: ~73% No Churn vs. ~27% Yes Churn in `churn_resampling.ipynb`
- Compared and implemented 3 resampling strategies (on training data only):
    - **Random UnderSampling:** reduced majority class
    - **Random OverSampling:** duplicated minority class
    - **SMOTE:** used to generate synthetic samples for the minority class
- Selected **SMOTE** as the preferred method for modeling


## Model 1: Logistic Regression (Baseline)
This is the first of several models used to predict customer churn.
- Trained on a SMOTE-balanced dataset to address class imbalance.
- Dropped multicollinear features based on a correlation matrix
- Used Logistic Regression as a baseline linear model due to its interpretability and strong performance
### Evaluation on Real-World Test Set
- ROC AUC Score: 0.97 (excellent class separability)
- Accuracy: 93%
- Precision / Recall (No Churn): 95% / 95%
- Precision / Recall (Churn): 87% / 87%
- F1-score: Balanced across both classes

## Model 2: Ridge Classifier (L2 Regression)
- Built on logistic regression by introducing L2 regularization
- Helped reduce potential multicollinearity and slightly improved performance
- Precision for churn class improved from 0.87 -> 0.92
- Overall accuracy increased from 93% -> 95%
- Retained all features but shrunk coefficient magnitudes for better generalization

## Model 3: Lasso Regression (L1 Regression)
- Applied L1 regularization to perform automatic feature selection
- Shrunk less important feature coefficients to zero
- Highlighted the most impactful features for churn:
    - `satisfaction_score`
    - `online_security`
    - `phone_service`
    - `internet_service`
- Improved model interpretability and reduced complexity
- Ideal when the goal is to identify key drivers of churn


## Conclusion: Linear Models
1. All three models (Logistic, Ridge, Lasso) performed well on the SMOTE-balanced dataset
2. **Logistic Regression:** Strong baseline with excellent ROC AUC (0.97)
3. **Ridge Classifier:** Slightly better precision and acuuracy, useful when all features matter
4. **Lasso Regression:** Best for understanding which features matter most, by reducing noise
5. These models laid the foundation for building interpretable churn prediction systems


## Model 4: Decision Tree Classifier
- Implemented a Decision Tree Classifier to predict customer churn
- Utilized the SMOTE-balanced dataset to address class imbalance
- Achieved the following performance metrics on the test set:
    - Accuracy: 95%
    - Precision: 0.96 (No Churn), 0.90 (Churn)
    - Recall: 0.96 (No Churn), 0.90 (Churn)
    - F1-Score: 0.96 (No Churn), 0.90 (Churn)
    - ROC AUC Score: 0.93
- The model demonstrated balanced performance across both classes.