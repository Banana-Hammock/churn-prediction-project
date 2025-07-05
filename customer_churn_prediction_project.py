import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

churn_pred = pd.read_csv("C:\\Users\\Ritwik Bhandari\\Downloads\\customer_churn_data.csv")

#Shape of the data
churn_pred.shape

missing_values = churn_pred.isna().sum()
print(missing_values)

# churn_pred.TotalCharges = pd.to_numeric(churn_pred.TotalCharges, errors='coerce')
# churn_pred.isnull().sum()

# Counting the number of duplicated rows in the DataFrame
churn_pred.duplicated().value_counts()

#Summary Statistics
summary_stat = churn_pred.describe()
print(summary_stat)

#boxplot to visualize Churners and Non Churners
sns.countplot(data=churn_pred,x='Churn', palette=['#2ca02c', '#9467bd'])

#Count of TARGET Variable per category
churn_pred['Churn'].value_counts().plot(kind='barh', figsize=(8, 6))
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)
plt.title("Count of TARGET Variable per category", y=1.02);

#Percentge of of TARGET Variable per category
100*churn_pred['Churn'].value_counts()/len(churn_pred['Churn'])

#TARGET Variable value count for each category
churn_pred['Churn'].value_counts()

#Create a copy of base data for manupulation & processing
churn_pred_copy = churn_pred.copy()
churn_pred_copy.head()

#Summary of Categorical Features
churn_pred_copy.describe(include=['object']).T

# Get the max tenure
print(churn_pred_copy['tenure'].max()) #72

# Group the tenure in bins of 12 months
labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

churn_pred_copy['tenure_group'] = pd.cut(churn_pred_copy.tenure, range(1, 80, 12), right=False, labels=labels)

churn_pred_copy['tenure_group'].value_counts()

# churn_pred_copy['tenure_group'] = churn_pred_copy['tenure_group'].cat.codes

#Remove columns not required for processing
churn_pred_copy.drop(columns= ['customerID','tenure'], axis=1, inplace=True)

# churn_pred_copy.head()

churn_pred_copy.info(verbose = True)

for i, predictor in enumerate(churn_pred_copy.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=churn_pred_copy, x=predictor, hue='Churn', palette=['#2ca02c', '#9467bd'])

# Make a copy of churn_pred_copy for Bivariat analysis
# churn_pred_bivariat = churn_pred_copy.copy()

# #Step 1: Convert Binary Variables
# binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
# for col in binary_columns:
#     churn_pred_bivariat[col] = churn_pred_bivariat[col].map({'Yes': 1, 'No': 0})

#Step 2: One-Hot Encode Non-Binary Categorical Variables
# non_binary_columns = [
#     'gender', 'MultipleLines', 'InternetService', 'OnlineSecurity',
#     'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
#     'StreamingMovies', 'Contract', 'PaymentMethod'
# ]

# churn_pred_bivariat = pd.get_dummies(churn_pred_bivariat, columns=non_binary_columns, drop_first=True)

#Step 3: Convert Target Variable (Churn)
# churn_pred_bivariat['Churn'] = churn_pred_bivariat['Churn'].map({'Yes': 1, 'No': 0})

#Step 4: Verify Data Types : Ensure all columns are numeric:
# churn_pred_bivariat = churn_pred_bivariat.applymap(
#     lambda x: 1 if x is True else (0 if x is False else x)
# )

# print(churn_pred_bivariat.dtypes)

# tenure_mapping = {
#     '1 - 12': 1,
#     '13 - 24': 2,
#     '25 - 36': 3,
#     '37 - 48': 4,
#     '49 - 60': 5,
#     '61 - 72': 6
# }

# # Map the tenure groups to numeric values
# churn_pred_bivariat['tenure_group'] = churn_pred_bivariat['tenure_group'].map(tenure_mapping)

# #Correlation Matrix
# correlation_matrix = churn_pred_bivariat.corr()
# sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
# plt.title('Correlation Matrix')
# plt.show()

"""**Convert the target variable 'Churn' in a binary numeric variable i.e. Yes=1 ; No = 0**

I created a copy of Dataset therefore, Whatever changes I am going to make, won't effect my original Dataset.
"""

churn_pred_copy['Churn'] = np.where(churn_pred_copy.Churn == 'Yes',1,0)

"""**3. Convert all the categorical variables into dummy variables.**

The Dummy Variable Trap occurs when two or more dummy variables created by one-hot encoding are highly correlated (multi-collinear). This means that one variable can be predicted from the others, making it difficult to interpret predicted coefficient variables in regression models.
"""

churn_pred_dummies = pd.get_dummies(churn_pred_copy).astype(int)
churn_pred_dummies.head()

print(churn_pred_dummies.dtypes)

"""**4. Relationship between Monthly Charges and Total Charges.**

**Insights:** Total Charges increase as Monthly Charges increase - as expected.
"""

# sns.lmplot(data=churn_pred_bivariat, x='MonthlyCharges', y='TotalCharges', fit_reg=False)

# sns.jointplot(data=churn_pred_dummies, x='MonthlyCharges', y='TotalCharges', kind='scatter')

sns.lmplot(data=churn_pred_dummies, x='MonthlyCharges', y='TotalCharges', fit_reg=False)

"""**5. Monthly Charges by Churn**

**Insights:**

- **Overlapping Curves:** There isn't a strong association between monthly charges and the likelihood of churn.
- **Similar Peak:** Both curves have a peak around the same monthly charge value, indicating that both churned and non-churned customers are most likely to have similar monthly charges.
- To understand the relationship between churn and monthly charges better, it would be beneficial to consider other factors that might influence churn.
- The graph does not support the conclusion that higher monthly charges are a strong predictor of churn. Further analysis is needed to understand the factors that drive churn in this dataset.
  
"""

# Mth = sns.kdeplot(churn_pred_bivariat.MonthlyCharges[(churn_pred_bivariat["Churn"] == 0) ],
#                 color="Red", shade = True)
# Mth = sns.kdeplot(churn_pred_bivariat.MonthlyCharges[(churn_pred_bivariat["Churn"] == 1) ],
#                 ax =Mth, color="Blue", shade= True)
# Mth.legend(["No Churn","Churn"],loc='upper right')
# Mth.set_ylabel('Density')
# Mth.set_xlabel('Monthly Charges')
# Mth.set_title('Monthly charges by churn')

Mth = sns.kdeplot(churn_pred_dummies.MonthlyCharges[(churn_pred_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(churn_pred_dummies.MonthlyCharges[(churn_pred_dummies["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')

"""**6. Total Charges by Churn.**

**Insights:**
- **Right-Skewed Distribution:** Both curves exhibit a right-skewed distribution, meaning there are more customers with lower total charges and fewer with higher total charges.
- **Peak Near Zero:** The peak of both curves is near zero total charges, suggesting a large number of customers with low total charges.
- **No Strong Association with Churn:** The overlap between the curves suggests that total charges alone may not be a strong predictor of churn. Customers with both high and low total charges are likely to churn or stay.
- The graph doesn't show a clear relationship between total charges and churn.
"""

# Tot = sns.kdeplot(churn_pred_bivariat.TotalCharges[(churn_pred_bivariat["Churn"] == 0) ],
#                 color="Red", shade = True)
# Tot = sns.kdeplot(churn_pred_bivariat.TotalCharges[(churn_pred_bivariat["Churn"] == 1) ],
#                 ax =Tot, color="Blue", shade= True)
# Tot.legend(["No Churn","Churn"],loc='upper right')
# Tot.set_ylabel('Density')
# Tot.set_xlabel('Total Charges')
# Tot.set_title('Total charges by churn')

Tot = sns.kdeplot(churn_pred_dummies.TotalCharges[(churn_pred_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Tot = sns.kdeplot(churn_pred_dummies.TotalCharges[(churn_pred_dummies["Churn"] == 1) ],
                ax =Tot, color="Blue", shade= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')

"""**7. Build a corelation of all predictors with 'Churn'**

**Strongest Predictors of Churn:**
- **Contract_Month-to-month:** Customers with month-to-month contracts have a higher likelihood of churning.
- **MultipleLines_No phone service:** Customers without phone service are more prone to churn.

**Moderate Predictors of Churn:**
- **TotalCharges:** Higher total charges are associated with increased churn.
- **TechSupport_No:** Customers without tech support are more likely to churn.
  
- Several other predictors show moderate to weak correlations with Churn, including tenure, internet service type, online security, and payment method.

**Insights for the Company:**
- Focus on retaining customers with month-to-month contracts and those without phone service.
- Address technical support issues and consider offering more affordable plans.
- Target specific customer segments with incentives or promotions to reduce churn.
"""

plt.figure(figsize=(20,8))
churn_pred_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')

plt.figure(figsize=(12,12))
sns.heatmap(churn_pred_dummies.corr(), cmap="Paired")

"""<div style="border-radius: 30px 0 30px 0px; border: 2px solid #87CEEB; padding: 20px; background: linear-gradient(to right, #F0F8FF, #E6E6FA, #D3D3D3); text-align: left; box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);">
    <h1 style="color: #333333; text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.8); font-weight: bold; margin-bottom: 13px; font-size: 24px;"> 8. Feature extraction </h1>
</div>

# Bivariate Analysis
"""

new_df1_target0=churn_pred_copy.loc[churn_pred_copy["Churn"]==0]
new_df1_target1=churn_pred_copy.loc[churn_pred_copy["Churn"]==1]

def uniplot(df, col, title, hue=None, palette=['#2ca02c', '#9467bd']):
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30

    temp = pd.Series(data=hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4 * len(temp.unique())
    fig.set_size_inches(width, 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)

    # Use the default palette if none is provided
    ax = sns.countplot(data=df, x=col, order=df[col].value_counts().index, hue=hue, palette=palette)

plt.show()

# uniplot(
#     new_df1_target1,
#     col='Partner',
#     title='Distribution of Partner Status for Churned Customers',
#     hue='gender_Male')

uniplot(new_df1_target1, col='Partner', title='Distribution of Gender for Churned Customers', hue='gender')

"""**Insights:**

- The churned customer who **have partners** are churning more. Among them male are churning more than females.
- The churned customer who **don't have partners** have a very low ratio of churning and among them males are more churners than females.
"""

uniplot(new_df1_target0,col='Partner',title='Distribution of Gender for Non Churned Customers',hue='gender')

"""**Insights:**

- For non churned customer who **have partners** are not churning more among both male and female. Where as males customers have higher propertion of non churners than female customers.
- For non churned customer who **don't have partners**, have a moderate propertion of non churners where female non churners customers are quiet higher in ratio than males customers.
"""

uniplot(new_df1_target1,col='PaymentMethod',title='Distribution of PaymentMethod for Churned Customers',hue='gender')

"""**Insights:**

* Customers with **Credit Card** service have the highest proportion of churners among both male and females.
* Customers with **Electronic Cheque** have also have a moderate ratio of churners among both male and females. Where the Male customers are high churners than female customers.
* Customers with **Mailed Cheque** also shows churners. Where the Male customers are too high churners than female customers.
* Customers with **Bank Transfer** have the lowest ratio of churners for both male and female customers.

"""

uniplot(new_df1_target0,col='PaymentMethod',title='Distribution of PaymentMethod for Non Churned Customers',hue='gender')

"""**Insights:**


* Customers with **Credit Card** service have the highest proportion of non churners among male customers. Where female customers have a very low ratio.
* Customers with **Electronic Cheque** have also have a moderate ratio of non churners of both male and female customers. Where the Male customers are high churners than female customers.
* Customers with **Mailed Cheque** shows equally non churners among both male and female customers.
* Customers with **Bank Transfer** also shows non churners where the ratio of male non churners is quiet higher than female non churners.

"""

uniplot(new_df1_target1,col='Contract',title='Distribution of Contract for Churned Customers',hue='gender')

"""**Insights:**


* Customers with **One-year Contract** type have the highest proportion of churners among male customers. Where male customers are slightly higher churners than female customers.
* Customers with **Month-to-Month Contract** type have also have a high ratio of churners among Male customers. Where the ratio of female churners is too low.
* Customers with **Two-year Contract** type shows low churners among both male and female customers. But even though male churners are higher than female churners.

"""

uniplot(new_df1_target0,col='Contract',title='Distribution of Contract for Non Churned Customers',hue='gender')

"""**Insights:**

* Customers with **Month-to-Month** contract type have the highest ratio of Non churners among both Male and Female customers. Where the ratio of female churners is a little higher than male customers.
* Customers with **One-year Contract** type have the lowest proportion of churners among both male and female customers.
* Customers with **Two-year Contract** type also lowest proportion of churners among both male and female customers.
"""

uniplot(new_df1_target1,col='TechSupport',title='Distribution of TechSupport for Churned Customers',hue='gender')

"""**Insights:**

- Customers with **No Internet service** have the highest proportion of churners both male and female. Where Famale are churning more than male customers.
- Customers with **No TechSupport** have also a moderate ratio of churners. Where the Male customers are high churners than female customers.
- Customers **With TechSupport** also shows churners. Where the Male customers are high churners than female customers.
"""

uniplot(new_df1_target0,col='TechSupport',title='Distribution of TechSupport for Non Churned Customers',hue='gender')

"""**Insights:**

* Customers with **No internet service** have the highest proportion of non churners. Where Famale are higher non churners than male customers.
* Customers with **No TechSupport** have also a high ratio of non churners. Where the male customers are too high churners than female customers.
* Customers **With TechSupport** also shows churners. Where the Male customers are too high churners than female customers.

"""

uniplot(new_df1_target1,col='SeniorCitizen',title='Distribution of SeniorCitizen for Churned Customers',hue='gender')

"""**Insights:**
- **Senior citizen** male are high churners, Where the female senior citizen customers have a very low rate of churning.
- **Non Senior citizen** are also high churners in general then senior citizens. Where male are high churners than females.
"""

uniplot(new_df1_target0,col='SeniorCitizen',title='Distribution of SeniorCitizen for Non Churned Customers',hue='gender')

"""**Insights:**

- **Senior citizen** are lower non churners than **Non senior citizen** in general.
- **Female senior citizen** customers are lower non churnes than **male senior citizens**.
- **Non Senior citizen** female are higher non churners than males.
"""

uniplot(new_df1_target1,col='tenure_group',title='Distribution of tenure_group for Churned Customers',hue='gender')

"""**Insights:**

- The tenure group 13-24 have the highest female churners.
- The tenure group 61-72,37-48,49-60, 1-12 have high proportion of male churners.
- The tenure group 25-36 have high proportation of Male churners. Wheare's this tenure group have the lowest female churners.
  
"""

uniplot(new_df1_target0,col='tenure_group',title='Distribution of tenure_group for Non Churned Customers',hue='gender')

churn_pred_dummies.to_csv('churn_pred.csv')

"""<div style="border-radius: 30px 0 30px 0px; border: 2px solid #87CEEB; padding: 20px; background: linear-gradient(to right, #F0F8FF, #E6E6FA, #D3D3D3); text-align: left; box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);">
    <h1 style="color: #333333; text-shadow: 1px 1px 3px rgba(255, 255, 255, 0.8); font-weight: bold; margin-bottom: 10px; font-size: 24px;"> 9. Create Model  </h1>
</div>

# **Importing Libraries**
"""

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN

churn_pred_model=pd.read_csv("churn_pred.csv")

#Drop Unnamed column
# df= df.drop('Unnamed: 0',axis=1)
churn_pred_model.head()

churn_pred_model=churn_pred_model.drop('Unnamed: 0',axis=1)

x=churn_pred_model.drop('Churn',axis=1)
x.head()

y=churn_pred_model['Churn']
y

#Test-train split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

"""**Decision Tree Classifier**"""

model_dt=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)

y_pred=model_dt.predict(x_test)
y_pred

model_dt.fit(x_train,y_train)

model_dt.score(x_test,y_test)

print(classification_report(y_test, y_pred, labels=[0,1]))

"""# Random Forest"""

from sklearn.ensemble import RandomForestClassifier

model_rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)

model_rf.fit(x_train,y_train)

y_pred = model_rf.predict(x_test)
y_pred

model_rf.score(x_test,y_test)

print(classification_report(y_test, y_pred, labels=[0,1]))

"""# Performing PCA:"""

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(0.9)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_

model_pca=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)

model_pca.fit(x_train_pca,y_train)

y_pred_pca = model_pca.predict(x_test_pca)
y_pred_pca

model_pca.score(x_test_pca,y_test)

print(classification_report(y_test, y_pred_pca, labels=[0,1]))

"""# Conclusion:
- We use three models decision Tree Classifier, Random Forest Classifier and PCA. The results for all the modesl are same.
- The Dataset is a balance dataset, therefore there is not much difference in churn and on churn customers.
"""