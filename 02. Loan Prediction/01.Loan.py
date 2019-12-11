import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

traindf = pd.read_csv('Data/train_ctrUa4K.csv')
testdf = pd.read_csv('Data/test_lAUu6dG.csv')
ytrain = traindf['Loan_Status']
traindf.drop(['Loan_Status'], axis = 1, inplace = True)
totaldf = traindf.append(testdf, sort = False)

imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
totaldf.iloc[:, [1,2,5,3]] = imp.fit_transform(totaldf.iloc[:, [1,2,5,3]])
imp1 = SimpleImputer(missing_values = np.nan, strategy = 'mean')
totaldf.iloc[:, [8,9,10]] = imp1.fit_transform(totaldf.iloc[:, [8,9,10]])


totaldf.Dependents = totaldf.Dependents.replace('3+', '3').astype('uint8')
totaldf['Total_household_income'] = totaldf['ApplicantIncome'] + totaldf['CoapplicantIncome']
totaldf['LoanAmount'] = pd.qcut(totaldf['LoanAmount'], 6).astype('str')
totaldf['LoanAmount'] = totaldf['LoanAmount'].replace(to_replace=['(128.0, 148.0]', '(110.0, 128.0]', '(8.999, 90.0]', '(185.0, 700.0]',
 '(90.0, 110.0]' ,'(148.0, 185.0]'], value=[3,2,0,5,1,4])

# #Removing Outlier Values
# totaldf.boxplot()
# plt.show()

# totaldf.drop(totaldf[totaldf['Total_household_income'] >= 50000].index, inplace= True, axis = 0)
totaldf['Total_household_income'] = pd.qcut(totaldf['Total_household_income'], 6).astype('str')
totaldf['Total_household_income'] = totaldf['Total_household_income'].replace(to_replace=['(5314.0, 6436.333]' ,'(1441.999, 3666.333]', '(4544.667, 5314.0]',
 '(8660.333, 81000.0]' ,'(3666.333, 4544.667]', '(6436.333, 8660.333]'], value=[3,0,2,5,1,4])
totaldf = pd.get_dummies(totaldf, drop_first=True, columns = ['Gender', 'Married', 'Education',
       'Self_Employed', 'Property_Area'])
totaldf.drop(['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome'], axis = 1, inplace = True)
xtrain = totaldf.iloc[:len(traindf), :]
xtest = totaldf.iloc[len(traindf):, :]


# print(xtrain.columns)
# print(xtrain[xtrain.columns[0]].unique())
# print(xtrain[xtrain.columns[1]].unique())
# print(xtrain[xtrain.columns[2]].unique())
# print(xtrain[xtrain.columns[3]].unique())
# print(xtrain[xtrain.columns[4]].unique())

# lb = LabelEncoder()
# ytrain = lb.fit_transform(ytrain)
#
# xtrain['target'] = ytrain
#
# corr = xtrain.corr()
# ax = sns.heatmap(
#     corr,
#     cmap='RdBu_r',
#     center=0,
#     square=True,
#     annot=True
# )
# ax.set_xticklabels(
#     ax.get_xticklabels(),
#     rotation=45,
#     horizontalalignment='right'
# )
# plt.show()

# xtrain.drop(['LoanAmount', 'Loan_Amount_Term', 'Self_Employed_Yes', 'Gender_Male', 'Total_household_income', 'Property_Area_Urban'], axis = 1, inplace = True)
# xtest.drop(['LoanAmount', 'Loan_Amount_Term', 'Self_Employed_Yes', 'Gender_Male', 'Total_household_income', 'Property_Area_Urban'], axis = 1, inplace = True)

stand = StandardScaler()
xtrain = stand.fit_transform(xtrain)
xtest = stand.transform(xtest)


# def rep(x):
#     if x == 1.0:
#         x = 'Y'
#     else:
#         x = 'N'
#     return x

##########################################    Logistic Regression - 0.78
lr = LogisticRegression()
lr.fit(xtrain, ytrain)
y_pred = lr.predict(xtest)

# y_pred = list(map(lambda x: rep(x), list(y_pred)))


submissiondf = pd.DataFrame(y_pred, columns=['Loan_Status'])
submissiondf['Loan_ID'] = testdf['Loan_ID']
submissiondf = submissiondf[['Loan_ID', 'Loan_Status']]
submissiondf.to_csv('08.LR_Correlation_Submission.csv', index = False)


# print(accuracy_score(y_test, y_pred))
############################################   Random Forest - 0.70
# rfc = RandomForestClassifier(n_estimators=500)
# rfc.fit(xtrain, ytrain)
# rfc_pred = rfc.predict(xtest)

# submissiondf = pd.DataFrame(rfc_pred, columns=['Loan_Status'])
# submissiondf['Loan_ID'] = testdf['Loan_ID']
# submissiondf = submissiondf[['Loan_ID', 'Loan_Status']]
# submissiondf.to_csv('06.RFC__Submission.csv', index = False)

# print(accuracy_score(y_test, rfc_pred))

############################################    LightGBM - 0.65
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'auc',
#     'max_depth': 2,
#     'learning_rate': 0.3,
#     'feature_fraction': 0.2,
#     'is_unbalance': True
# }
#
# train_data = lgb.Dataset(xtrain,  ytrain)
# test_data = lgb.Dataset(xtest, reference = train_data)
# lgb_train = lgb.train(params,train_data, valid_sets = [train_data, test_data], num_boost_round=5000,)
# predicted = lgb_train.predict(xtest)
# print(predicted)
#
def rep(x):
    if x == 1.0:
        x = 'Y'
    else:
        x = 'N'
    return x
#
# predicted = list(map(lambda x: round(x), list(predicted)))
# print(predicted)
#
# predicted = list(map(lambda x: rep(x), list(predicted)))
#
# print(predicted)
#
# submissiondf = pd.DataFrame(predicted, columns=['Loan_Status'])
# submissiondf['Loan_ID'] = testdf['Loan_ID']
# submissiondf = submissiondf[['Loan_ID', 'Loan_Status']]
# submissiondf.to_csv('07.LightGBM_Submission.csv', index = False)
