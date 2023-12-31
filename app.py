# ===   Step 1: install the streamlit via terminal ===
# pip install streamlit 

# === Step 2: download the data file 'LoanStats_2019Q1.csv' from github ===
# https://github.com/Chunyan94/CreditRiskApp.git

# === Step 3: run the code via terminal ===
# streamlit run /full/path/to/app.py

# === step 4: upload the data set on app ===

# ********************** The project codes are as follows: *********************************          
# -------- Packages ---------
import streamlit as st 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

st.title("Credit Risk Data \n - Alex HOC and Chunyan JI - M2 IRFA \n - Date: Jan.05, 2024")
st.markdown('## Introduction')
st.text('This project is to access(predict) credit risk by exploring the individual loan(credit) data.')

uploaded_file = st.file_uploader('Upload the file here')

# ----------- Clean up the data  ----------- 
columns = [
    "loan_amnt", "int_rate", "installment", "home_ownership",
    "annual_inc", "verification_status", "issue_d", "loan_status",
    "pymnt_plan", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "total_acc",
    "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt",
    "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt", "next_pymnt_d",
    "collections_12_mths_ex_med", "policy_code", "application_type", "acc_now_delinq",
    "tot_coll_amt", "tot_cur_bal", "open_acc_6m", "open_act_il",
    "open_il_12m", "open_il_24m", "mths_since_rcnt_il", "total_bal_il",
    "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc",
    "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl",
    "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "hardship_flag", "debt_settlement_flag"
]


target = ["loan_status"]

if uploaded_file:

    df = pd.read_csv(uploaded_file, skiprows=1) [:-2]

    df = df.loc[:, columns].copy()
    df = df.dropna(axis='columns', how='all')
    df = df.dropna()
    # convert interest rate to numerical
    df['int_rate'] = df['int_rate'].str.replace('%', '')
    df['int_rate'] = df['int_rate'].astype('float') / 100
    # Remove the `Issued` loan status
    df = df.loc[df['loan_status'] != 'Issued']

    st.header("Data")
    st.write(df)

    st.header("Data statistics")
    st.write(df.describe())

st.text("Drop outliers: such as annula income = 0.")
# Identify numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns

# Select numeric columns with mean value not equal to 0
selected_numeric_columns = [col for col in numeric_columns if df[col].mean() != 0]

# Create a new DataFrame with selected numeric columns
df = df.loc[:,[selected_numeric_columns,'loan_status']].copy()
st.write(df.describe())

# ------- Data description --------
st.markdown('## All variables explained: https://wiki.datrics.ai/more-features')
st.markdown('## Dependent variable "y": loan statuses')
st.text('''
        The severity of post-issurance loan statuses, in ascending order of seriousness, is as follows:
            1. **Current**: Your payments are up to date.
            2. **In Grace Period**: Payments are overdue but within a 15-day grace period.
            3. **Late (16-30)**: Payments are 16-30 days overdue, the first stage of delinquency.
            4. **Late (31–120)**: Payments are 31–120 days overdue, the second stage of delinquency.
            5. **Default**: Payments are overdue for more than 120 days. Lender starts the process to charge off the loan.
            6. **Charged Off**: Lender charges off the loan when further payments aren't expected. This usually happens within 30 days after the loan enters default.
        Fore more details, please check the link:https://www.lendingclub.com/help/investing-faq/what-do-the-different-note-statuses-mean''')

def data_plot():
    st.markdown(" ### Visualization of Loan status")
    sd = st.selectbox(
        "Select a Plot", #Drop Down Menu Name
        [
 
            "Box Plot"  #First option in menu
            ,"Violin Plot" #Second option in menu
        #    , "Count Plot"
        ]
    )

    fig = plt.figure(figsize=(12, 6))

    #if sd == "Count Plot":
    #    sns.histplot(data = df, x = 'loan_status')

    if sd == "Box Plot":
        sns.boxplot(data = df, x = 'loan_status', y= 'loan_amnt', hue='loan_status')
        # Move the legend to the right top corner
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

    if sd == "Violin Plot":
        sns.violinplot(data = df, x = 'loan_status', y= 'loan_amnt', hue='loan_status')
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    

    st.pyplot(fig)

data_plot()

 # ----- Regroup loan_status -----
st.markdown('## Loan status distribution')
# Calculate counts and percentages
status_counts = df['loan_status'].value_counts()
status_percentages = df['loan_status'].value_counts(normalize=True) * 100

# Concatenate into a single DataFrame
status_summary = pd.concat([status_counts, status_percentages], axis=1)
status_summary.columns = ['Count', 'Percentage']

# Display the DataFrame
st.write(status_summary)

st.text('''The current loan status accounts for  $99% percent of data. 
    It is reasonable to group the rest togther into /Risky/.
    Current into /Low Risk/ group''' )
# Convert the target column values to low_risk and high_risk based on their values
new_LoanStatus = {'Current': 'low_risk'}   
df = df.replace(new_LoanStatus).copy()

new_LoanStatus2 = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'risky')    
df = df.replace(new_LoanStatus2).copy()
st.write(df['loan_status'].value_counts())


## --------- Split Train data & Test Data -------

st.title('Split dataset in features and target variable')

st.text("Regroup Loan status to 2 groups: Current - low_risk; Others - risky")


X_S = df.drop(columns=['loan_status']) # Features
X = pd.get_dummies(X_S, columns=["home_ownership","verification_status","issue_d",
                                  "pymnt_plan","initial_list_status","next_pymnt_d","debt_settlement_flag",
                                  "application_type","hardship_flag"], drop_first = True)

y = df['loan_status'] # Target variable
categorical_variables = ["home_ownership","verification_status","issue_d",
                         "pymnt_plan","initial_list_status","next_pymnt_d","debt_settlement_flag",
                         "application_type","hardship_flag"]
st.write(df.drop(columns=categorical_variables).groupby('loan_status').mean())

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

## --------- Decision Tree Test -------
st.title('Decision Tree Test')
# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
st.write(clf)
st.write("Decision Tree Plot:")
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(clf, filled=True, rounded=True, ax=ax)
st.pyplot(fig)

# Predict the response for test dataset
y_pred = clf.predict(X_test)
st.text('How accurate is the decsision classifier ?')
st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred))
st.write('However, the decision tree method could cause overfitting.')

## --------- Logistic regression Test -------
st.title('Logistic regression test')
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
logreg = LogisticRegression()
logreg_result = logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
st.write('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
logreg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test)

# Display accuracy
st.write('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# Get the coefficients and intercept
coefficients = logreg.coef_
intercept = logreg.intercept_

# Display the coefficients and intercept
st.write("Coefficients:", coefficients)
st.write("Intercept:", intercept)


#st.write(logreg_result)
#rfe = RFE(logreg, 20)
#rfe = rfe.fit(X, y.values.ravel())
#st.write(rfe.support_)
#st.write(rfe.ranking_)
#import statsmodels.api as sm
#logit_model=sm.Logit(y_train, X_train)
#result=logit_model.fit()
#st.write(result.summary2())