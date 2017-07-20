

import os
""" First change the following directory link to where all input files do exist """
os.chdir("D:\\Book writing\\Codes\\Chapter 4")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split    
from sklearn.metrics import accuracy_score,classification_report

import matplotlib.pyplot as plt


hrattr_data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print (hrattr_data.head())

hrattr_data['Attrition_ind'] = 0
hrattr_data.loc[hrattr_data['Attrition']=='Yes','Attrition_ind'] = 1

dummy_busnstrvl = pd.get_dummies(hrattr_data['BusinessTravel'], prefix='busns_trvl')
dummy_dept = pd.get_dummies(hrattr_data['Department'], prefix='dept')
dummy_edufield = pd.get_dummies(hrattr_data['EducationField'], prefix='edufield')
dummy_gender = pd.get_dummies(hrattr_data['Gender'], prefix='gend')
dummy_jobrole = pd.get_dummies(hrattr_data['JobRole'], prefix='jobrole')
dummy_maritstat = pd.get_dummies(hrattr_data['MaritalStatus'], prefix='maritalstat') 
dummy_overtime = pd.get_dummies(hrattr_data['OverTime'], prefix='overtime') 

continuous_columns = ['Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction',
'HourlyRate', 'JobInvolvement', 'JobLevel','JobSatisfaction','MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction','StockOptionLevel', 'TotalWorkingYears', 
'TrainingTimesLastYear','WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
'YearsWithCurrManager']

hrattr_continuous = hrattr_data[continuous_columns]

hrattr_continuous['Age'].describe()
hrattr_data['BusinessTravel'].value_counts()

hrattr_data_new = pd.concat([dummy_busnstrvl,dummy_dept,dummy_edufield,dummy_gender,dummy_jobrole,
  dummy_maritstat,dummy_overtime,hrattr_continuous,hrattr_data['Attrition_ind']],axis=1)

# Train & Test split
x_train,x_test,y_train,y_test = train_test_split(hrattr_data_new.drop(['Attrition_ind'],axis=1),
                                                 hrattr_data_new['Attrition_ind'],train_size = 0.7,random_state=42)

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_fit = DecisionTreeClassifier(criterion="gini",max_depth=5,min_samples_split=2,min_samples_leaf=1,random_state=42)
dt_fit.fit(x_train,y_train)

print ("\nDecision Tree - Train Confusion Matrix\n\n",pd.crosstab(y_train,dt_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nDecision Tree - Train accuracy:",round(accuracy_score(y_train,dt_fit.predict(x_train)),3))
print ("\nDecision Tree - Train Classification Report\n",classification_report(y_train,dt_fit.predict(x_train)))

print ("\n\nDecision Tree - Test Confusion Matrix\n\n",pd.crosstab(y_test,dt_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nDecision Tree - Test accuracy:",round(accuracy_score(y_test,dt_fit.predict(x_test)),3))
print ("\nDecision Tree - Test Classification Report\n",classification_report(y_test,dt_fit.predict(x_test)))


# Tuning class weights to analyze accuracy, precision & recall
dummyarray = np.empty((6,10))
dt_wttune = pd.DataFrame(dummyarray)

dt_wttune.columns = ["zero_wght","one_wght","tr_accuracy","tst_accuracy","prec_zero","prec_one",
                     "prec_ovll","recl_zero","recl_one","recl_ovll"]

zero_clwghts = [0.01,0.1,0.2,0.3,0.4,0.5]

for i in range(len(zero_clwghts)):
    clwght = {0:zero_clwghts[i],1:1.0-zero_clwghts[i]}
    dt_fit = DecisionTreeClassifier(criterion="gini",max_depth=5,min_samples_split=2,
                                    min_samples_leaf=1,random_state=42,class_weight = clwght)
    dt_fit.fit(x_train,y_train)
    dt_wttune.loc[i, 'zero_wght'] = clwght[0]       
    dt_wttune.loc[i, 'one_wght'] = clwght[1]     
    dt_wttune.loc[i, 'tr_accuracy'] = round(accuracy_score(y_train,dt_fit.predict(x_train)),3)    
    dt_wttune.loc[i, 'tst_accuracy'] = round(accuracy_score(y_test,dt_fit.predict(x_test)),3)    
        
    clf_sp = classification_report(y_test,dt_fit.predict(x_test)).split()
    dt_wttune.loc[i, 'prec_zero'] = float(clf_sp[5])   
    dt_wttune.loc[i, 'prec_one'] = float(clf_sp[10])   
    dt_wttune.loc[i, 'prec_ovll'] = float(clf_sp[17])   
    
    dt_wttune.loc[i, 'recl_zero'] = float(clf_sp[6])   
    dt_wttune.loc[i, 'recl_one'] = float(clf_sp[11])   
    dt_wttune.loc[i, 'recl_ovll'] = float(clf_sp[18])
    print ("\nClass Weights",clwght,"Train accuracy:",round(accuracy_score(y_train,dt_fit.predict(x_train)),3),"Test accuracy:",round(accuracy_score(y_test,dt_fit.predict(x_test)),3))
    print ("Test Confusion Matrix\n\n",pd.crosstab(y_test,dt_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      


# Bagging Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

dt_fit = DecisionTreeClassifier(criterion="gini",max_depth=5,min_samples_split=2,min_samples_leaf=1,random_state=42,
                                class_weight = {0:0.3,1:0.7})

bag_fit = BaggingClassifier(base_estimator= dt_fit,n_estimators=5000,max_samples=0.67,max_features=1.0,
                            bootstrap=True,bootstrap_features=True,n_jobs=-1,random_state=42)

bag_fit.fit(x_train, y_train)

print ("\nBagging - Train Confusion Matrix\n\n",pd.crosstab(y_train,bag_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nBagging- Train accuracy",round(accuracy_score(y_train,bag_fit.predict(x_train)),3))
print ("\nBagging  - Train Classification Report\n",classification_report(y_train,bag_fit.predict(x_train)))

print ("\n\nBagging - Test Confusion Matrix\n\n",pd.crosstab(y_test,bag_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nBagging - Test accuracy",round(accuracy_score(y_test,bag_fit.predict(x_test)),3))
print ("\nBagging - Test Classification Report\n",classification_report(y_test,bag_fit.predict(x_test)))



# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf_fit = RandomForestClassifier(n_estimators=5000,criterion="gini",max_depth=5,min_samples_split=2,bootstrap=True,
                                max_features='auto',random_state=42,min_samples_leaf=1,class_weight = {0:0.3,1:0.7})
rf_fit.fit(x_train,y_train)       

print ("\nRandom Forest - Train Confusion Matrix\n\n",pd.crosstab(y_train,rf_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nRandom Forest - Train accuracy",round(accuracy_score(y_train,rf_fit.predict(x_train)),3))
print ("\nRandom Forest  - Train Classification Report\n",classification_report(y_train,rf_fit.predict(x_train)))

print ("\n\nRandom Forest - Test Confusion Matrix\n\n",pd.crosstab(y_test,rf_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nRandom Forest - Test accuracy",round(accuracy_score(y_test,rf_fit.predict(x_test)),3))
print ("\nRandom Forest - Test Classification Report\n",classification_report(y_test,rf_fit.predict(x_test)))


# Plot of Variable importance by mean decrease in gini
model_ranks = pd.Series(rf_fit.feature_importances_,index=x_train.columns, name='Importance').sort_values(ascending=False, inplace=False)
model_ranks.index.name = 'Variables'
top_features = model_ranks.iloc[:31].sort_values(ascending=True,inplace=False)
plt.figure(figsize=(20,10))
ax = top_features.plot(kind='barh')
_ = ax.set_title("Variable Importance Plot")
_ = ax.set_xlabel('Mean decrease in Variance')
_ = ax.set_yticklabels(top_features.index, fontsize=13)




# Random Forest Classifier - Grid Search
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV

pipeline = Pipeline([
        ('clf',RandomForestClassifier(criterion='gini',class_weight = {0:0.3,1:0.7}))])

parameters = {
        'clf__n_estimators':(2000,3000,5000),
        'clf__max_depth':(5,15,30),
        'clf__min_samples_split':(2,3),
        'clf__min_samples_leaf':(1,2)  }

grid_search = GridSearchCV(pipeline,parameters,n_jobs=-1,cv=5,verbose=1,scoring='accuracy')
grid_search.fit(x_train,y_train)

print ('Best Training score: %0.3f' % grid_search.best_score_)
print ('Best parameters set:')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ('\t%s: %r' % (param_name, best_parameters[param_name]))

predictions = grid_search.predict(x_test)

print ("Testing accuracy:",round(accuracy_score(y_test, predictions),4))
print ("\nComplete report of Testing data\n",classification_report(y_test, predictions))
print ("\n\nRandom Forest Grid Search- Test Confusion Matrix\n\n",pd.crosstab(y_test, predictions,rownames = ["Actuall"],colnames = ["Predicted"]))      


# Adaboost Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
dtree = DecisionTreeClassifier(criterion='gini',max_depth=1)

adabst_fit = AdaBoostClassifier(base_estimator= dtree,
        n_estimators=5000,learning_rate=0.05,random_state=42)

adabst_fit.fit(x_train, y_train)

print ("\nAdaBoost - Train Confusion Matrix\n\n",pd.crosstab(y_train,adabst_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nAdaBoost  - Train accuracy",round(accuracy_score(y_train,adabst_fit.predict(x_train)),3))
print ("\nAdaBoost  - Train Classification Report\n",classification_report(y_train,adabst_fit.predict(x_train)))

print ("\n\nAdaBoost  - Test Confusion Matrix\n\n",pd.crosstab(y_test,adabst_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nAdaBoost  - Test accuracy",round(accuracy_score(y_test,adabst_fit.predict(x_test)),3))
print ("\nAdaBoost - Test Classification Report\n",classification_report(y_test,adabst_fit.predict(x_test)))

# Gradientboost Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbc_fit = GradientBoostingClassifier(loss='deviance',learning_rate=0.05,n_estimators=5000,
                                     min_samples_split=2,min_samples_leaf=1,max_depth=1,random_state=42 )
gbc_fit.fit(x_train,y_train)

print ("\nGradient Boost - Train Confusion Matrix\n\n",pd.crosstab(y_train,gbc_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nGradient Boost - Train accuracy",round(accuracy_score(y_train,gbc_fit.predict(x_train)),3))
print ("\nGradient Boost  - Train Classification Report\n",classification_report(y_train,gbc_fit.predict(x_train)))

print ("\n\nGradient Boost - Test Confusion Matrix\n\n",pd.crosstab(y_test,gbc_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nGradient Boost - Test accuracy",round(accuracy_score(y_test,gbc_fit.predict(x_test)),3))
print ("\nGradient Boost - Test Classification Report\n",classification_report(y_test,gbc_fit.predict(x_test)))

  
# Xgboost Classifier
import xgboost as xgb

xgb_fit = xgb.XGBClassifier(max_depth=2, n_estimators=5000, learning_rate=0.05)
xgb_fit.fit(x_train, y_train)

print ("\nXGBoost - Train Confusion Matrix\n\n",pd.crosstab(y_train,xgb_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nXGBoost - Train accuracy",round(accuracy_score(y_train,xgb_fit.predict(x_train)),3))
print ("\nXGBoost  - Train Classification Report\n",classification_report(y_train,xgb_fit.predict(x_train)))

print ("\n\nXGBoost - Test Confusion Matrix\n\n",pd.crosstab(y_test,xgb_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nXGBoost - Test accuracy",round(accuracy_score(y_test,xgb_fit.predict(x_test)),3))
print ("\nXGBoost - Test Classification Report\n",classification_report(y_test,xgb_fit.predict(x_test)))


#Ensemble of Ensembles - by fitting various classifiers
clwght = {0:0.3,1:0.7}

# Classifier 1
from sklearn.linear_model import LogisticRegression
clf1_logreg_fit = LogisticRegression(fit_intercept=True,class_weight=clwght)
clf1_logreg_fit.fit(x_train,y_train)

print ("\nLogistic Regression for Ensemble - Train Confusion Matrix\n\n",pd.crosstab(y_train,clf1_logreg_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nLogistic Regression for Ensemble - Train accuracy",round(accuracy_score(y_train,clf1_logreg_fit.predict(x_train)),3))
print ("\nLogistic Regression for Ensemble - Train Classification Report\n",classification_report(y_train,clf1_logreg_fit.predict(x_train)))

print ("\n\nLogistic Regression for Ensemble - Test Confusion Matrix\n\n",pd.crosstab(y_test,clf1_logreg_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nLogistic Regression for Ensemble - Test accuracy",round(accuracy_score(y_test,clf1_logreg_fit.predict(x_test)),3))
print ("\nLogistic Regression for Ensemble - Test Classification Report\n",classification_report(y_test,clf1_logreg_fit.predict(x_test)))


# Classifier 2
from sklearn.tree import DecisionTreeClassifier
clf2_dt_fit = DecisionTreeClassifier(criterion="gini",max_depth=5,min_samples_split=2,
                                     min_samples_leaf=1,random_state=42,class_weight=clwght)
clf2_dt_fit.fit(x_train,y_train)

print ("\nDecision Tree for Ensemble - Train Confusion Matrix\n\n",pd.crosstab(y_train,clf2_dt_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nDecision Tree for Ensemble - Train accuracy",round(accuracy_score(y_train,clf2_dt_fit.predict(x_train)),3))
print ("\nDecision Tree for Ensemble - Train Classification Report\n",classification_report(y_train,clf2_dt_fit.predict(x_train)))

print ("\n\nDecision Tree for Ensemble - Test Confusion Matrix\n\n",pd.crosstab(y_test,clf2_dt_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nDecision Tree for Ensemble - Test accuracy",round(accuracy_score(y_test,clf2_dt_fit.predict(x_test)),3))
print ("\nDecision Tree for Ensemble - Test Classification Report\n",classification_report(y_test,clf2_dt_fit.predict(x_test)))


# Classifier 3
from sklearn.ensemble import RandomForestClassifier
clf3_rf_fit = RandomForestClassifier(n_estimators=10000,criterion="gini",max_depth=6,
                                min_samples_split=2,min_samples_leaf=1,class_weight = clwght)
clf3_rf_fit.fit(x_train,y_train)       

print ("\nRandom Forest for Ensemble - Train Confusion Matrix\n\n",pd.crosstab(y_train,clf3_rf_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nRandom Forest for Ensemble - Train accuracy",round(accuracy_score(y_train,clf3_rf_fit.predict(x_train)),3))
print ("\nRandom Forest for Ensemble - Train Classification Report\n",classification_report(y_train,clf3_rf_fit.predict(x_train)))

print ("\n\nRandom Forest for Ensemble - Test Confusion Matrix\n\n",pd.crosstab(y_test,clf3_rf_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nRandom Forest for Ensemble - Test accuracy",round(accuracy_score(y_test,clf3_rf_fit.predict(x_test)),3))
print ("\nRandom Forest for Ensemble - Test Classification Report\n",classification_report(y_test,clf3_rf_fit.predict(x_test)))


# Classifier 4
from sklearn.ensemble import AdaBoostClassifier
clf4_dtree = DecisionTreeClassifier(criterion='gini',max_depth=1,class_weight = clwght)
clf4_adabst_fit = AdaBoostClassifier(base_estimator= clf4_dtree,
        n_estimators=5000,learning_rate=0.05,random_state=42)

clf4_adabst_fit.fit(x_train, y_train)

print ("\nAdaBoost for Ensemble  - Train Confusion Matrix\n\n",pd.crosstab(y_train,clf4_adabst_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nAdaBoost for Ensemble   - Train accuracy",round(accuracy_score(y_train,clf4_adabst_fit.predict(x_train)),3))
print ("\nAdaBoost for Ensemble   - Train Classification Report\n",classification_report(y_train,clf4_adabst_fit.predict(x_train)))

print ("\n\nAdaBoost for Ensemble   - Test Confusion Matrix\n\n",pd.crosstab(y_test,clf4_adabst_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nAdaBoost for Ensemble   - Test accuracy",round(accuracy_score(y_test,clf4_adabst_fit.predict(x_test)),3))
print ("\nAdaBoost for Ensemble  - Test Classification Report\n",classification_report(y_test,clf4_adabst_fit.predict(x_test)))


ensemble = pd.DataFrame()

ensemble["log_output_one"] = pd.DataFrame(clf1_logreg_fit.predict_proba(x_train))[1]
ensemble["dtr_output_one"] = pd.DataFrame(clf2_dt_fit.predict_proba(x_train))[1]
ensemble["rf_output_one"] = pd.DataFrame(clf3_rf_fit.predict_proba(x_train))[1]
ensemble["adb_output_one"] = pd.DataFrame(clf4_adabst_fit.predict_proba(x_train))[1]

ensemble = pd.concat([ensemble,pd.DataFrame(y_train).reset_index(drop = True )],axis=1)

# Fitting meta-classifier
meta_logit_fit =  LogisticRegression(fit_intercept=False)
meta_logit_fit.fit(ensemble[['log_output_one','dtr_output_one','rf_output_one','adb_output_one']],ensemble['Attrition_ind'])

coefs =  meta_logit_fit.coef_
print ("Co-efficients for LR, DT, RF & AB are:",coefs)

ensemble_test = pd.DataFrame()
ensemble_test["log_output_one"] = pd.DataFrame(clf1_logreg_fit.predict_proba(x_test))[1]
ensemble_test["dtr_output_one"] = pd.DataFrame(clf2_dt_fit.predict_proba(x_test))[1]
ensemble_test["rf_output_one"] = pd.DataFrame(clf3_rf_fit.predict_proba(x_test))[1]
ensemble_test["adb_output_one"] = pd.DataFrame(clf4_adabst_fit.predict_proba(x_test))[1]

ensemble_test["all_one"] = meta_logit_fit.predict(ensemble_test[['log_output_one','dtr_output_one','rf_output_one','adb_output_one']])

ensemble_test = pd.concat([ensemble_test,pd.DataFrame(y_test).reset_index(drop = True )],axis=1)

print ("\n\nEnsemble of Models - Test Confusion Matrix\n\n",pd.crosstab(ensemble_test['Attrition_ind'],ensemble_test['all_one'],rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nEnsemble of Models - Test accuracy",round(accuracy_score(ensemble_test['Attrition_ind'],ensemble_test['all_one']),3))
print ("\nEnsemble of Models - Test Classification Report\n",classification_report(ensemble_test['Attrition_ind'],ensemble_test['all_one']))




# Ensemble of Ensembles - by applying bagging on simple classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

clwght = {0:0.3,1:0.7}

eoe_dtree = DecisionTreeClassifier(criterion='gini',max_depth=1,class_weight = clwght)
eoe_adabst_fit = AdaBoostClassifier(base_estimator= eoe_dtree,
        n_estimators=500,learning_rate=0.05,random_state=42)
eoe_adabst_fit.fit(x_train, y_train)

print ("\nAdaBoost - Train Confusion Matrix\n\n",pd.crosstab(y_train,eoe_adabst_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nAdaBoost - Train accuracy",round(accuracy_score(y_train,eoe_adabst_fit.predict(x_train)),3))
print ("\nAdaBoost  - Train Classification Report\n",classification_report(y_train,eoe_adabst_fit.predict(x_train)))

print ("\n\nAdaBoost - Test Confusion Matrix\n\n",pd.crosstab(y_test,eoe_adabst_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nAdaBoost - Test accuracy",round(accuracy_score(y_test,eoe_adabst_fit.predict(x_test)),3))
print ("\nAdaBoost - Test Classification Report\n",classification_report(y_test,eoe_adabst_fit.predict(x_test)))


bag_fit = BaggingClassifier(base_estimator= eoe_adabst_fit,n_estimators=50,
                            max_samples=1.0,max_features=1.0,
                            bootstrap=True,
                            bootstrap_features=False,
                            n_jobs=-1,
                            random_state=42)

bag_fit.fit(x_train, y_train)

print ("\nEnsemble of AdaBoost - Train Confusion Matrix\n\n",pd.crosstab(y_train,bag_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nEnsemble of AdaBoost - Train accuracy",round(accuracy_score(y_train,bag_fit.predict(x_train)),3))
print ("\nEnsemble of AdaBoost  - Train Classification Report\n",classification_report(y_train,bag_fit.predict(x_train)))

print ("\n\nEnsemble of AdaBoost - Test Confusion Matrix\n\n",pd.crosstab(y_test,bag_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nEnsemble of AdaBoost - Test accuracy",round(accuracy_score(y_test,bag_fit.predict(x_test)),3))
print ("\nEnsemble of AdaBoost - Test Classification Report\n",classification_report(y_test,bag_fit.predict(x_test)))








