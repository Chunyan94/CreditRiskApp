# ******************************************************************************
# Alex HOC, Chunyan JI - IRFA Jan.05,2024
# ******************************************************************************

# === Step 1: install the streamlit via terminal ===
# pip install streamlit 

# === Step 2: download the data file 'LoanStats_2019Q1.csv' from github ===
# https://github.com/Chunyan94/CreditRiskApp.git

# === Step 3: run the code via terminal ===
# streamlit run /full/path/to/app.py

# === Step 4: upload the data set on app ===

# ********************* The project codes are as follows: ************************         
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

st.title("Credit Risk Data \n - Alex HOC and Chunyan JI - M2 IRFA \n - Date: Jan.05, 2024")
st.markdown('## Introduction')
st.text('This project is to access(predict) credit risk by exploring the individual loan(credit) data.')
st.markdown('## The statistical methods used in this projects are:')
st.text('''
        1. Decision Tree
        2. Random Forest
        3. Logistical Regression 
        4. Support Vector Machines (SVM)
        ''')


# ----------- Upload and clean the source data  ----------- 

uploaded_file = 'https://media.githubusercontent.com/media/Chunyan94/CreditRiskApp/main/LoanStats_2019Q1.csv'
st.write(uploaded_file)
# uploaded_file = st.file_uploader('Upload the file here')


columns = ["loan_amnt", "int_rate", "installment", "home_ownership",
           "annual_inc", "verification_status", "issue_d", "loan_status",
           "pymnt_plan","avg_cur_bal","bc_open_to_buy","dti","emp_title",
           "num_accts_ever_120_pd","num_actv_bc_tl","open_acc",'initial_list_status', 
           "next_pymnt_d", "debt_settlement_flag", "application_type", "hardship_flag",
           "term","grade","sub_grade","emp_length"]

target = ["loan_status"]

if uploaded_file:

    df = pd.read_csv(uploaded_file, skiprows=1,low_memory=False) [:-2]

    df = df.loc[:, columns].copy()
    df = df.dropna(axis='columns', how='all')
    df = df.dropna()
    # convert interest rate to numerical
    df['int_rate'] = df['int_rate'].str.replace('%', '')
    df['int_rate'] = df['int_rate'].astype('float') / 100
        # Remove the `Issued` loan status
    df = df.loc[df['loan_status'] != 'Issued']
    # subset 30% of the data to reduce running fast
    df = df.sample(frac=0.05).copy()
    df = df.reset_index(drop=True).copy()

    st.header("Data")
    st.write(df)

    st.header("Data statistics")
    st.write(df.describe())


# ------- Data description --------
st.markdown('## All variables')
st.text('''
        avg_cur_bal: Average current balance of all accounts
        bc_open_to_buy: Total open to buy on revolving bankcards
        dti: The borrower's debt to income ratio =  monthly payments on dedts(exclude mortgage)/income 
        emp_title: The job title 
        home_ownership: Type of ownership: RENT, OWN, MORTGAGE, NONE, OTHER
        installment: The monthly payment owed by the borrower if the loan originates.
        int_rate: Interest Rate on the loan
        issue_d: The date which the loan was funded
        loan_amnt: Total amount of loan
        loan_status: Current status of the loan
        num_accts_ever_120_pd: Number of accounts ever 120 or more days past due
        num_actv_bc_tl: Number of currently active bankcard accounts
        num_bc_tl: Number of bankcard accounts
        open_acc:The number of open credit lines in the borrower's credit file 
        term: duration of loan
        grade: assigned loan grade by lend club
        sub_grade: assigned loan subgrade

        
        For more details, please check: https://wiki.datrics.ai/more-features''')

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

st.markdown('## Loan status distribution')
# Calculate counts and percentages
status_counts = df['loan_status'].value_counts()
status_percentages = df['loan_status'].value_counts(normalize=True) * 100

# Concatenate into a single DataFrame
status_summary = pd.concat([status_counts, status_percentages], axis=1)
status_summary.columns = ['Count', 'Percentage']

st.write(status_summary)

def data_plot():
    st.markdown(" ### Visualization of Loan status")
    sd = st.selectbox(
        "Select a Plot", #Drop Down Menu Name
        [
 
            "Box Plot"  #First option in menu
            ,"Violin Plot" #Second option in menu
        ]
    )

    fig = plt.figure(figsize=(12, 6))

    if sd == "Box Plot":
        sns.boxplot(data = df, x = 'loan_status', y= 'loan_amnt', hue='loan_status')
 
    if sd == "Violin Plot":
        sns.violinplot(data = df, x = 'loan_status', y= 'loan_amnt', hue='loan_status')
    

    st.pyplot(fig)

data_plot()

# %%
## Relationship between 'loan_status' and 'annual_income'

fig, ax = plt.subplots(figsize=(15, 5))
# Modify the hue and set palette
bar_plot = sns.countplot(x='home_ownership', hue='loan_status', data=df, ax=ax)
ax.set_title('Loan Status by Home Ownership', fontweight='bold')
ax.set_xlabel('Home Ownership', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')

# Move the legend to the top-right
bar_plot.legend(title='Loan Status', loc='upper right', bbox_to_anchor=(1.25, 1))
# plt.show()
st.pyplot(fig)

# Annotate counts on top of bars for counts >= 5000
for p in bar_plot.patches:
    count = p.get_height()
    if count >= 5000:
        formatted_count = '{:,.0f}'.format(count).replace(',', ' ')
        ax.text(p.get_x() + p.get_width() / 2., p.get_height(), formatted_count,
                ha='center', va='bottom', fontweight='bold', color='black')

# Display the plot using Streamlit
# st.pyplot(fig)

# %%
## Relationship between Loan Status and Loan amount
# Streamlit app
# st.title("Loan Amount by Loan Status")

# # Create a boxplot using Seaborn
# fig, ax = plt.subplots(figsize=(15, 5))
# box_plot = sns.boxplot(x='loan_status', y='loan_amnt', data=df,hue = "loan_status", ax=ax)
# ax.set_title('Loan Amount by Loan Status ($)', fontweight='bold')
# ax.set_xlabel('Loan Status', fontweight='bold')
# ax.set_ylabel('Loan Amount ($)', fontweight='bold')

# # Display the plot using Streamlit
# st.pyplot(fig)

# %%
## Relationship between Loan Status and interest rate
st.title("Intest rate VS Loan Status")

# Create a boxplot using Seaborn
fig, ax = plt.subplots(figsize=(15, 5))
box_plot = sns.violinplot(x='loan_status', y='int_rate', data=df, hue='loan_status', ax=ax)
ax.set_title('Intrest rate by Loan Status', fontweight='bold')
ax.set_xlabel('Loan Status', fontweight='bold')
ax.set_ylabel('Intrest Rate ', fontweight='bold')

# Display the plot using Streamlit
st.pyplot(fig)
# %%
#  Loan Amount vs Annual Income
from scipy.stats import pearsonr
st.title("Scatter Plot of Loan Amount vs Annual Income")
df_clean = df.dropna(subset=['loan_amnt', 'annual_inc'])

# Calculate Pearson correlation
pearson_coef, p_value = pearsonr(df_clean['loan_amnt'], df_clean['annual_inc'])
st.write(f"Pearson Correlation Coefficient: {pearson_coef}")
st.write(f"P-Value: {p_value}")

# Create a scatter plot using Seaborn
fig, ax = plt.subplots(figsize=(10, 6))
scatter_plot = sns.scatterplot(data=df_clean, x='annual_inc', y='loan_amnt', palette="magma", hue='loan_amnt', ax=ax)
ax.set_title(f"Scatter Plot of Loan Amount vs Annual Income\nPearson Correlation: {pearson_coef:.2f}", fontweight='bold')
ax.set_xlabel('Annual Income ($)')
ax.set_ylabel('Loan Amount ($)')

# Display the plot using Streamlit
st.pyplot(fig)

## --------- Train data & Test Data -------



st.title('Split dataset in features and target variable')

st.text("Regroup Loan status to 2 groups: Current - low_risk; Others - risky")
# Convert the target column values to low_risk and high_risk based on their values
new_LoanStatus = {'Current': 'low_risk'}   
df = df.replace(new_LoanStatus).copy()

new_LoanStatus2 = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'risky')    
df = df.replace(new_LoanStatus2).copy()
st.write(df['loan_status'].value_counts())

X_S = df.drop(columns=['loan_status']) # Features
# X = pd.get_dummies(X_S, columns=["home_ownership","verification_status","issue_d",
#                                   "pymnt_plan","initial_list_status","next_pymnt_d","debt_settlement_flag",
#                                   "application_type","hardship_flag"], drop_first = True)
X = pd.get_dummies(X_S, columns=  ["home_ownership", "verification_status", "issue_d",
                      "pymnt_plan", "initial_list_status", "next_pymnt_d",
                      "debt_settlement_flag", "application_type", "hardship_flag","emp_title",
                      "term","grade","sub_grade","emp_length"], drop_first = True)

y = df['loan_status'] # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# ------------ Random Forest -----------
st.markdown("""
# Ensemble Learners

In this section, you will compare two ensemble algorithms to determine which algorithm results in the best performance. You will train a Balanced Random Forest Classifier and an Easy Ensemble AdaBoost classifier. For each algorithm, be sure to complete the following steps:

1. Train the model using the training data.
2. Calculate the balanced accuracy score from sklearn.metrics.
3. Print the confusion matrix from sklearn.metrics.
4. Generate a classification report using the `imbalanced_classification_report` from imbalanced-learn.
5. For the Balanced Random Forest Classifier only, print the feature importance sorted in descending order (most important feature to least important) along with the feature score

Note: Use a random state of 1 for each algorithm to ensure consistency between tests

### Balanced Random Forest Classifier
""")
# Scale data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaler = scaler.fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Resample the training data with the RandomOversampler
from imblearn.ensemble import BalancedRandomForestClassifier
# Specify the sampling strategy explicitly
sampling_strategy = 'all'

brfc = BalancedRandomForestClassifier(n_estimators =1000, random_state=1)
model = brfc.fit(X_train_scaled, y_train)
BalancedRandomForestClassifier()

# %%
# Calculate the balanced accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

predictions = model.predict(X_test_scaled)
accuracy_score(y_test, predictions)

# %%
# Display the confusion matrix
confusion_matrix(y_test, predictions)

# %%
# Print the imbalanced classification report
st.write(classification_report(y_test, predictions))

# %%
# List the features sorted in descending order by feature importance
sorted(zip(model.feature_importances_, X.columns), reverse=True)[:20]

# %% [markdown]
#  ------------------------ Easy Ensemble AdaBoost Classifier ------------------------ 

# %%
# Train the EasyEnsembleClassifier
# Train the Classifier
from imblearn.ensemble import EasyEnsembleClassifier

model = EasyEnsembleClassifier(base_estimator=None, n_estimators=100, n_jobs=1, random_state=1, 
                                   replacement=False, sampling_strategy='auto', verbose=0, 
                                   warm_start=False)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

EasyEnsembleClassifier()

# %%
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
predictions = model.predict(X_test_scaled)
balanced_accuracy_score(y_test, predictions)
st.write(balanced_accuracy_score(y_test, predictions))
# %%
# Display the confusion matrix
confusion_matrix(y_test, predictions)
st.write(confusion_matrix(y_test, predictions))
# %%
# # Print the imbalanced classification report
# from imblearn.metrics import classification_report_imbalanced
# # st.write(classification_report(y_test, predictions))

# report_dict = classification_report_imbalanced(y_test, predictions, output_dict=True)
# report_df = pd.DataFrame(report_dict)

# # Display the DataFrame using st.table
# st.table(report_df)

# %% 
# --------------------- Support Vector Machines --------------------- 
#import svm model
st.markdown("Support Vector Machines - SVM")
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
st.write("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# # Model Precision: what percentage of positive tuples are labeled as such?
# st.write("Precision:",metrics.precision_score(y_test, y_pred, pos_label='low_risk'))

# # Model Recall: what percentage of positive tuples are labelled as such?
# st.write("Recall:",metrics.recall_score(y_test, y_pred, pos_label='low_risk'))


# %% 
# -------------------------------- Logistic regression -------------------------------- 

st.title('Logistic regression test')
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg_result = logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

# import the metrics class
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
st.write(cnf_matrix)


# visualize the confusion matrix using Heatmap.
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted Loan Status')
st.pyplot(fig)

from sklearn.metrics import classification_report
target_names = ['Low Risk default', 'High Risk default']
st.write(classification_report(y_test, y_pred, target_names=target_names, zero_division=1))

st.write('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
# Display the coefficients and intercept
st.write("Coefficients:", logreg.coef_)
st.write("Intercept:", logreg.intercept_)

# Display the coefficients and intercept along with variable names
coefficients = logreg.coef_[0]
intercept = logreg.intercept_[0]

# Pair feature names with their coefficients
coef_dict = pd.DataFrame({"Variable": X_train.columns, "Coefficient": coefficients})

# Filter out rows with coefficients close to zero
# Sort coefficients by absolute value in descending order
coef_dict = coef_dict.reindex(coef_dict['Coefficient'].abs().sort_values(ascending=False).index)

# Select the top 10 coefficients
top_coef = coef_dict.head(20)

# Display coefficients and intercept
st.write("Intercept:", intercept)
st.write("Coefficients:")
st.table(top_coef)

#%% 
#----------------------------------- Decision Tree Test ----------------------------------- 

st.title('Decision Tree Test')
# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier(max_depth=3)
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


# %%%% 3: discuss the results from the data science point view %%%%
st.title("Discuss the results from the data science point view")




# %%%% 4: discuss the results from the business point view (answer the problematic) 
st.title("Discuss the results from the business point view")
