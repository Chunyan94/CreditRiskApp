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
# from pathlib import Path
from collections import Counter
from patsy import dmatrices
import statsmodels.api as sm

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

# %% Project introduction
st.title("Credit Risk Data \n - Alex HOC and Chunyan JI - M2 IRFA \n - Date: Jan.05, 2024")

st.write('Program is running. It might take some time to load the program. Please give me some time.')

# 1: Introduction of the Topic
st.markdown("## 1: Introduction of the Topic")
st.write("Credit scoring is a crucial aspect of evaluating a bank's customer creditworthiness, assessing their ability and willingness to repay debts. Given that less than half of the banked population is deemed eligible for lending, the demand for more sophisticated credit scoring solutions is evident.")

# 2: Define the Problematic (1 point)
st.markdown("## 2: Define the Problematic ")
st.write("The challenge at hand involves creating a Machine Learning model to evaluate Credit Default Risk and accessing the creditworthiness assessment of new customers. This requires a comprehensive understanding of various factors, such as total income, credit history, transaction analysis, and work experience, to formulate an effective predictive model.")

# 3: Select a Representative Data Set (1 point)
st.markdown("## 3: Select a Representative Data Set")
st.write("To illustrate the credit scoring assessment, we will utilize a subset of the Loan Club public dataset within the Datrics platform. This dataset contains essential information about the bank's customers financial data, loan requests, loan details, credit history, and current payment statuses.The dataset is very big file. To run it faster, we randomly selected small percentage of data. We also remove highly correlated variables, such as total loan amount and loan term")

# ----------- Upload and clean the source data  ----------- 
# uploaded_file = st.file_uploader('Upload the file here')
uploaded_file = 'https://media.githubusercontent.com/media/Chunyan94/CreditRiskApp/main/LoanStats_2019Q1.csv'
st.text("The data is accessible via the link:")
st.write(uploaded_file)

st.markdown('## The statistical methods used in this projects are:')
st.text('''
        1. Decision Tree
        2. Random Forest
        3. Logistical Regression 
        4. Zero-Inflated Poisson model
        ''')
# Select coloumns to display
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

    df = df.reset_index(drop=True).copy()

    st.header("Source Data")
    st.write(df)
    
st.write(f"Number of Rows: {len(df)}, Number of Columns: {len(df.columns)}")

# ---------- Data cleaning and feature varibales selection ---------- 

df['term'] = df['term'].str.extract('(\d+)').astype('float')
df['emp_length'] = df['emp_length'].str.extract('(\d+)').astype('float')
df['month_inc'] = df['annual_inc']/12

# df['loan_status_coded'] = pd.factorize(np.array(df['loan_status']))

feature_columns = ["loan_status","loan_amnt", "int_rate", "installment", "home_ownership",
           "month_inc",  "avg_cur_bal","bc_open_to_buy","dti","emp_title",
           "num_accts_ever_120_pd","num_actv_bc_tl","open_acc",'initial_list_status', 
           "debt_settlement_flag", "application_type","emp_length"]


fraction = st.slider("Select fraction(%) of source data to run faster", min_value=0.1, max_value=0.9, step=0.1)
# subset the data to run fast
df = df.sample(frac=fraction).copy()
df = df[feature_columns].copy()
df = df.reset_index(drop=True).copy()

# ------- Data description --------
st.markdown('### Feature variables')
st.text('''
        avg_cur_bal: Average current balance of all accounts
        month_inc: month income 
        bc_open_to_buy: available credit limit for spending on revolving credit accounts
        dti: The borrower's debt to income ratio =  monthly payments on dedts(exclude mortgage)/income 
        emp_title: The job title 
        emp_length: employee length
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
        initial_list_status: The initial listing status of the loan. Possible values are – W(whole loan), F

        
        For more details, please check: https://wiki.datrics.ai/more-features''')

st.write(df)
st.header("Data statistics")
st.write(df.describe())

st.markdown('## Dependent variable "y": loan statuses')
st.text('''
        The severity of post-issurance loan statuses, in ascending order of seriousness, is as follows:
            1. **Current**: someone who has settled all debts and/or fees due
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
        sns.boxplot(data = df, x = 'loan_status', y= 'installment', hue='loan_status')
        # ax.set_ylabel('Montly loan payment', fontweight='bold')
 
    if sd == "Violin Plot":
        sns.violinplot(data = df, x = 'loan_status', y= 'installment', hue='loan_status')
        # ax.set_ylabel('Montly loan payment', fontweight='bold')
    
    st.pyplot(fig)

data_plot()
st.write("We noticed there are many outliers in the Current (someone who has settled all debts and/or fees due) categories, the monthly loan payment has big variation. We need to check if it is a crucial information")

# %%
#   Monthly debt to pay (installment) and Monthly Income
from scipy.stats import pearsonr
st.markdown("Installment(Monthly debt to pay) vs Monthly Income")
df_clean = df.dropna(subset=['installment', 'month_inc'])

# Calculate Pearson correlation
pearson_coef, p_value = pearsonr(df_clean['installment'], df_clean['month_inc'])
st.write(f"Pearson Correlation Coefficient: {pearson_coef}")
st.write(f"P-Value: {p_value}")
st.write('''p value is < 0.01, it is assumed that monthly loan payment has positive correlation (coefficient > 0) with montly income. 
        However, there are some outliers. Some borrowers' monthly loan payment is higher than monthly income''')
         
fig = plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='month_inc', y='installment', palette="magma", hue='loan_amnt')
plt.title(f"Pearson Correlation: {pearson_coef:.2f}", fontweight='bold')
plt.xlabel('Monthly Income ($)')
plt.ylabel('Month loan payment ($)')

st.pyplot(fig)
# %%
## Relationship between 'loan_status' and 'home_ownership'
st.markdown('loan status and home ownership')

fig, ax = plt.subplots(figsize=(15, 5))
bar_plot = sns.countplot(x='home_ownership', hue='loan_status', data=df, ax=ax)
ax.set_title('Loan Status by Home Ownership', fontweight='bold')
ax.set_xlabel('Home Ownership', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
bar_plot.legend(title='Loan Status', loc='upper right')
# bar_plot.legend(title='Loan Status', loc='upper right', bbox_to_anchor=(1.25, 1))
st.pyplot(fig)


# %%
# Relationship between Loan Status and Loan amount

st.markdown("Debt ratio to income VS Loan Status")
fig, ax = plt.subplots(figsize=(15, 5))
box_plot = sns.boxplot(x='loan_status', y='dti', data=df,hue = "loan_status", ax=ax)
ax.set_title('Debt ratio(%) VS Loan', fontweight='bold')
ax.set_xlabel('Loan Status', fontweight='bold')
ax.set_ylabel('Debt ratio(%)', fontweight='bold')
st.pyplot(fig)

# %%
## Relationship between Loan Status and interest rate
st.markdown("Employee Length VS Loan Status")
fig, ax = plt.subplots(figsize=(15, 5))
box_plot = sns.violinplot(x='loan_status', y='emp_length', data=df, hue='loan_status', ax=ax)
ax.set_title('Employee Length(Year) VS Loan Status', fontweight='bold')
ax.set_xlabel('Loan status', fontweight='bold')
ax.set_ylabel('Employee Length(Year) ', fontweight='bold')
st.pyplot(fig)

## --------- Train data & Test Data -------
st.title('Split dataset to Train data & Test Data')

st.text("Regroup Loan status to 2 groups: Current - low_risk; Others - risky")
# Convert the target column values to low_risk and high_risk based on their values
new_LoanStatus = {'Current': 'low_risk'}   
df = df.replace(new_LoanStatus).copy()

new_LoanStatus2 = dict.fromkeys(['Late (31-120 days)', 'Late (16-30 days)', 'Default', 'In Grace Period'], 'risky')    
df = df.replace(new_LoanStatus2).copy()
st.write(df['loan_status'].value_counts())
# factorize loan_status
df['loan_status_coded'], _ = pd.factorize(np.array(df['loan_status']))
st.write(df['loan_status_coded'].value_counts())

display_dtypes = st.checkbox("Display DataFrame Data Types")
# Display DataFrame or data types based on the checkbox
if display_dtypes:
    st.write(df.dtypes)

df.dropna(inplace=True)  # This modifies the original DataFrame in place
# Drop missing values
df = df.dropna().copy()

# Extract features
X = df.drop(columns=['loan_status', 'loan_status_coded'])

# Identify categorical columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
st.write(f"Categorical variables are: {categorical_columns}")

# Factorize categorical columns
X[categorical_columns] = X[categorical_columns].apply(lambda x: pd.factorize(x)[0])

# Check for remaining missing values
remaining_missing_values = X.isnull().sum()
st.write("Columns with remaining missing values:")
st.write(remaining_missing_values[remaining_missing_values > 0])

# X = X_S  

# Check the data types again
display_dtypes = st.checkbox("Display Training Data Types(X)")
# Display DataFrame or data types based on the checkbox
if display_dtypes:
    st.write(X.dtypes)

# Extract target variable
y = df['loan_status_coded']

# Split dataset into training set and test set
# Add a slider for selecting the test size
size = st.slider("Select Test data Size", min_value=0.3, max_value=0.5, step=0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=1)

# Check data types of y_train
y_train = y_train.astype(int)
# st.write(y_train.dtypes)

# %% 
# -------------------- Random Forest ----------------
st.write('To note: the exact model accuracy could change based on the choice.')

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
sampling_strategy = 'all'

brfc = BalancedRandomForestClassifier(n_estimators =1000, random_state=1)
model = brfc.fit(X_train_scaled, y_train)
BalancedRandomForestClassifier()

# %%
# Calculate the balanced accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

predictions = model.predict(X_test_scaled)
st.write(accuracy_score(y_test, predictions))

# %%
# Display the confusion matrix
confusion_matrix(y_test, predictions)

# %%
# Print the imbalanced classification report
st.write(classification_report(y_test, predictions))

##%%
#List the features sorted in descending order by feature importance
feature_importances = sorted(zip(model.feature_importances_, X.columns), reverse=True)[:20]
feature_importances_df = pd.DataFrame(feature_importances, columns=['Importance', 'Feature'])
st.table(feature_importances_df)
# sorted(zip(model.feature_importances_, X.columns), reverse=True)[:20]


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

# %%
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
predictions = model.predict(X_test_scaled)
balanced_acc_score = balanced_accuracy_score(y_test, predictions)
st.write(f"Balanced Accuracy Score: {balanced_acc_score}")

# Display the confusion matrix
confusion_matrix(y_test, predictions)
st.write("Display the confusion matrix" )
st.write(confusion_matrix(y_test, predictions))
st.write('The Balanced Random Forest Classifier exhibits modest(about 50 % - 60%) accuracy in predicting the majority class{Current}. However, its performance is less robust in identifying instances of minority classes, such as {In Grace Period} and {Late (31-120 days)}, where precision, recall, and F1-scores are notably lower. The model\'s challenge in capturing instances of the minority classes may necessitate further adjustments or tuning to enhance its predictive capabilities. Considering the class imbalance, particularly the substantial number of instances in the {Current} class, understanding the implications of false positives and false negatives is vital in the context of risk management and lending decisions.')
st.write('This means that the model prediction correctly predicted the results for about half of the cases. In other words, for about half of the data that we analyzed and tested, the Random Forest model correctly identified whether it was a risky or not risky loan (low and high risk). By looking at the classification report table, we see that the model is very good at identifying the low_risk class. Indeed, an F1 Score of more than 70% reflects a solid and balanced performance between precision and recall. On the other hand, Random Forest has difficulty correctly predicting the high_risk class. Indeed, with an F1 Score of less than 10%, this indicates a significant imbalance between precision and recall despite an average recall score.')
st.write('Furthermore, we can assume that this Random Forest model tends to favor the class majority low_risk due to the predominance of this class in all data, leading to poor model performance on the high_risk class. This must be rebalanced in order to be able to fairly predict these two classes. This imbalance of the two classes may be due to too large a proportion of false positives for the high_risk class implying that the model tends to incorrectly classify the high_risk and low_risk classes. In other words, this means that individuals who have granted loans can be wrongly classified as high_risk/low_risk, thus potentially leading to inappropriate decisions. To fill these gaps, it is necessary to rebalance the data of the two target classes (high_risk and low_risk) in order to have a precise and fair model in the prediction of all classes. For example, it may be useful to increase the number of minority subcategories in the high_risk class in order to improve the precision of this class. Conversely, we can also reduce the number of majority low_risk subcategories in order to have balanced datasets. On the other hand, it may be interesting to add adjustment weights between these two classes. In this case, it would perhaps be necessary to grant a greater weight of error to the minority high_risk class than the majority low_risk class so that the model is penalized more for incorrect classifications of the high_risk class. Additionally, it may also be useful to move from categorical binary categories to continuous data variables to better capture complex predictions and information that is difficult to categorize. This also helps avoid “forcing” predictions into a specific binary and therefore reduces classification errors (particularly in cases where limits are found neither in high_risk nor in low_risk). In short, the Random Forest model, although effective in predicting the low_risk class, demonstrates significant shortcomings in the prediction of the high_risk class, mainly due to very low precision. Improvements should therefore focus on balancing data and adjusting the model to better capture the nuances of the high-risk cases.')

# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
# st.write(classification_report(y_test, predictions))

report_dict = classification_report_imbalanced(y_test, predictions, output_dict=True)
report_df = pd.DataFrame(report_dict)

# Display the DataFrame using st.table
st.table(report_df)

# %% 
# --------------------------------Zero-Inflated Poisson model-------------------------------- 

st.markdown('## Zero-Inflated Poisson model')
zip_training_results = sm.ZeroInflatedPoisson(endog=y_train, exog=X_train, exog_infl=X_train, inflation='logit').fit()
st.write(zip_training_results.summary())
st.write('Model Evaluation: There are several issues in terms of model fit and estimation. The standard errors are nan, indicating that there might be an issue with convergence. loan_amnt, int_rate, ... emp_length:  coefficients are very close to zero, suggesting little to no impact on the count of the dependent variable. ')



# %% 
# -------------------------------- Logistic regression -------------------------------- 

st.title('Logistic regression test')
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(max_iter=1000)
logreg_result = logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

# import the metrics class
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
st.write(cnf_matrix)

# visualize the confusion matrix using Heatmap.
class_names=y_train.unique() # name  of classes
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

st.write(' Notes: A small portion(about 5% when I run the model) of current(low risk) borrowers were classified as high risk by logistic regression. Could we consider it as misclaffication? Maybe not. Because in our data, we noticed that a portion of borrowers who paid their debt due by the data collection time. However, considering some have high debt ratio, high installment, interest rate...conditions, there is a increasing possibilty of default. It is important to consider their debt ratio, monthly income, and installment,etc.. More attention should be paid to those borrowers.')
st.write('Accuracy of logistic regression classifier on test set: {:.2f}.format(logreg.score(X_test, y_test))')

coefficients = logreg.coef_[0]
intercept = logreg.intercept_[0]
st.write("Intercept:", intercept)
# Pair feature names with their coefficients
coef_dict = pd.DataFrame({"Variable": X_train.columns, "Coefficient": coefficients})
st.write("Coefficients:")
coef_dict = coef_dict.reindex(coef_dict['Coefficient'].abs().sort_values(ascending=False).index)
st.write(coef_dict)
st.write('Negative coefficients suggestes decrease in the likelihood of default (high risk). Positive coefficients suggestes increase in the likelihood of default (high risk).However, it might also depend on variable scales. ')

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
st.write('Decision has hight accuracy. However, the decision tree method could cause overfitting. Besides, the interpretation of the model output could also a challeng to business.')

st.title("Compare the models")
st.write('Both Decision Tree Test and Logistic regression have high accuracy in this project. However,the decision tree could cause overfitting problem, while the logistic regression could be impacted by (low frequency class) data. Balanced Random Forest Classifier could have high accuracy and might be biased towards the majority class, leading to high accuracy while poorly predicting the minority class.  Zero-Inflated Poisson model is not applicable in our case. We might also consider other models for count data, such as the Negative Binomial regression. Negative Binomial models are more flexible than Poisson models and can handle overdispersion. ')
st.write('Based on the results above, logistic regression model has adavantages in interpretation, computation efficiency, and high accuracy. But we could improve the model by reducing varaibles (collilarity) and imputing missing values. We could also combine the Random forest modela and logistic regression model. First, we use the random forest model to select important varaibles, and then run the logistic regression model. It is not sure if it is a reasonable way to do it. More rigurous methods should apply in future research.')

# %%%%  discuss the results from the business point view (answer the problematic) 
st.title("Discuss the results from the business point view")
st.write("In our credit risk analysis project, we used variables influencing credit risk for informed decision-making. We start by visualizing the distribution of each variable and their relationships with 'loan_status.' It provides initial insights into potential risk factors. Next, we calculate correlations between independent variables and the target variable, 'loan_status,' revealing variables with significant linear relationships impacting credit risk. To assess the importance of each variable in predicting credit risk, we used machine learning models like decision trees, random forests, and logistic regression. Feature importance scores generated by these models highlight the variables contributing the most to our predictive capabilities. In the context of decision trees, we measure information gain to identify variables effectively splitting the data based on 'loan_status.' Additionally, mutual information provides insights into the dependence between variables and credit risk, enhancing our understanding of complex relationships. To go further, we can employ techniques to remove less important variables to observe their impact on model performance. This refinement enhances the efficiency and accuracy of our credit risk assessment model. In summary, our credit risk analysis project provides us with a certain degree of understanding of variable importance in assessing and mitigating credit risk.")
st.write('The important variables for credit risk analysis are : monthly income, interest rate, loan amount, installment(monthly debt payment),  current bank account, debt to income ration, employee title. Those variables abouve are important indicators of credit risk. Company could collect more accurate data about those parameters.')
st.write('There is a great proportion of Current borrows has very high debt ratio > 1. We should keep an eye on it.')
st.write('Looking at the confusion matrix table, the model missed a high number of borrowers by misclassifying them as low risk when they were actually high risk. This can have significant consequences on the lending activity of a financial institution. Indeed, the institution could be faced with significant losses which unfortunately could have been avoided. Additionally, the high proportion of false positives means that the model classifies many borrowers as high risk when they are actually low risk. This can lead to the rejection of a large number of creditworthy borrowers, resulting in lost revenue.')
