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