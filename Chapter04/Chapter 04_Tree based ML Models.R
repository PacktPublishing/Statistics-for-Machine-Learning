


rm(list = ls())

# First change the following directory link to where all the input files do exist
setwd("D:\\Book writing\\Codes\\Chapter 4")


library(randomForest)
library(gbm)
library(xgboost)
library(C50)
library(e1071)
library(caret)

hrattr_data = read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

str(hrattr_data)
summary(hrattr_data)

hrattr_data$Attrition_ind = 0;hrattr_data$Attrition_ind[hrattr_data$Attrition=="Yes"]=1
hrattr_data$Attrition_ind = as.factor(hrattr_data$Attrition_ind)

remove_cols = c("EmployeeCount","EmployeeNumber","Over18","StandardHours","Attrition")
hrattr_data_new = hrattr_data[,!(names(hrattr_data) %in% remove_cols)]

set.seed(123)
numrow = nrow(hrattr_data_new)
trnind = sample(1:numrow,size = as.integer(0.7*numrow))
train_data = hrattr_data_new[trnind,]
test_data = hrattr_data_new[-trnind,]

frac_trzero = (table(train_data$Attrition_ind)[[1]])/nrow(train_data)
frac_trone = (table(train_data$Attrition_ind)[[2]])/nrow(train_data)

frac_tszero = (table(test_data$Attrition_ind)[[1]])/nrow(test_data)
frac_tsone = (table(test_data$Attrition_ind)[[2]])/nrow(test_data)

prec_zero <- function(act,pred){  tble = table(act,pred)
return( round( tble[1,1]/(tble[1,1]+tble[2,1]),4)  ) }

prec_one <- function(act,pred){ tble = table(act,pred)
return( round( tble[2,2]/(tble[2,2]+tble[1,2]),4)   ) }

recl_zero <- function(act,pred){tble = table(act,pred)
return( round( tble[1,1]/(tble[1,1]+tble[1,2]),4)   ) }

recl_one <- function(act,pred){ tble = table(act,pred)
return( round( tble[2,2]/(tble[2,2]+tble[2,1]),4)  ) }

accrcy <- function(act,pred){ tble = table(act,pred)
return( round((tble[1,1]+tble[2,2])/sum(tble),4)) }


# Decision Trees using C5.0 package
library(C50)
dtree_fit = C5.0(train_data[-31],train_data$Attrition_ind,costs = NULL,
                 control = C5.0Control(minCases = 1))

summary(dtree_fit)

tr_y_pred = predict(dtree_fit, train_data,type = "class")
ts_y_pred = predict(dtree_fit,test_data,type = "class")

tr_y_act = train_data$Attrition_ind;ts_y_act = test_data$Attrition_ind

tr_tble = table(tr_y_act,tr_y_pred)
print(paste("Train Confusion Matrix"))
print(tr_tble)

tr_acc = accrcy(tr_y_act,tr_y_pred)
trprec_zero = prec_zero(tr_y_act,tr_y_pred); trrecl_zero = recl_zero(tr_y_act,tr_y_pred)
trprec_one = prec_one(tr_y_act,tr_y_pred); trrecl_one = recl_one(tr_y_act,tr_y_pred)

trprec_ovll = trprec_zero *frac_trzero + trprec_one*frac_trone
trrecl_ovll = trrecl_zero *frac_trzero + trrecl_one*frac_trone

print(paste("Decision Tree Train accuracy:",tr_acc))
print(paste("Decision Tree - Train Classification Report"))
print(paste("Zero_Precision",trprec_zero,"Zero_Recall",trrecl_zero))
print(paste("One_Precision",trprec_one,"One_Recall",trrecl_one))
print(paste("Overall_Precision",round(trprec_ovll,4),"Overall_Recall",round(trrecl_ovll,4)))

ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_y_act,ts_y_pred)
tsprec_zero = prec_zero(ts_y_act,ts_y_pred); tsrecl_zero = recl_zero(ts_y_act,ts_y_pred)
tsprec_one = prec_one(ts_y_act,ts_y_pred); tsrecl_one = recl_one(ts_y_act,ts_y_pred)

tsprec_ovll = tsprec_zero *frac_tszero + tsprec_one*frac_tsone
tsrecl_ovll = tsrecl_zero *frac_tszero + tsrecl_one*frac_tsone

print(paste("Decision Tree Test accuracy:",ts_acc))
print(paste("Decision Tree - Test Classification Report"))
print(paste("Zero_Precision",tsprec_zero,"Zero_Recall",tsrecl_zero))
print(paste("One_Precision",tsprec_one,"One_Recall",tsrecl_one))
print(paste("Overall_Precision",round(tsprec_ovll,4),"Overall_Recall",round(tsrecl_ovll,4)))


#Decision Trees using C5.0 package - Error Costs
library(C50)

class_zero_wgt = c(0.01,0.1,0.2,0.3,0.4,0.5)

for (cwt in class_zero_wgt){
  cwtz = cwt
  cwto = 1-cwtz
  cstvr = cwto/cwtz
  
  error_cost <- matrix(c(0, 1, cstvr, 0), nrow = 2)
  
  dtree_fit = C5.0(train_data[-31],train_data$Attrition_ind,costs = error_cost,
                   control = C5.0Control(minCases = 1))
  
  summary(dtree_fit)
  
  tr_y_pred = predict(dtree_fit, train_data,type = "class")
  ts_y_pred = predict(dtree_fit,test_data,type = "class")
  
  tr_y_act = train_data$Attrition_ind;ts_y_act = test_data$Attrition_ind
  tr_acc = accrcy(tr_y_act,tr_y_pred)
  ts_acc = accrcy(ts_y_act,ts_y_pred)
  
  print(paste("Class weights","{0:",cwtz,"1:",cwto,"}",
              "Decision Tree Train accuracy:",tr_acc,
              "Decision Tree Test accuracy:",ts_acc))
  ts_tble = table(ts_y_act,ts_y_pred)
  print(paste("Test Confusion Matrix"))
  print(ts_tble)
 
}


# Bagging Classifier - using Random forest package but all variables selected
library(randomForest)

set.seed(43)
rf_fit = randomForest(Attrition_ind~.,data = train_data,mtry=30,maxnodes= 64,
                      classwt = c(0.3,0.7),ntree=5000,nodesize = 1)

tr_y_pred = predict(rf_fit,data = train_data,type = "response")
ts_y_pred = predict(rf_fit,newdata = test_data,type = "response")

tr_y_act = train_data$Attrition_ind;ts_y_act = test_data$Attrition_ind

tr_tble = table(tr_y_act,tr_y_pred)
print(paste("Train Confusion Matrix"))
print(tr_tble)

tr_acc = accrcy(tr_y_act,tr_y_pred)
trprec_zero = prec_zero(tr_y_act,tr_y_pred); trrecl_zero = recl_zero(tr_y_act,tr_y_pred)
trprec_one = prec_one(tr_y_act,tr_y_pred); trrecl_one = recl_one(tr_y_act,tr_y_pred)

trprec_ovll = trprec_zero *frac_trzero + trprec_one*frac_trone
trrecl_ovll = trrecl_zero *frac_trzero + trrecl_one*frac_trone

print(paste("Random Forest Train accuracy:",tr_acc))
print(paste("Random Forest - Train Classification Report"))
print(paste("Zero_Precision",trprec_zero,"Zero_Recall",trrecl_zero))
print(paste("One_Precision",trprec_one,"One_Recall",trrecl_one))
print(paste("Overall_Precision",round(trprec_ovll,4),"Overall_Recall",round(trrecl_ovll,4)))

ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_y_act,ts_y_pred)
tsprec_zero = prec_zero(ts_y_act,ts_y_pred); tsrecl_zero = recl_zero(ts_y_act,ts_y_pred)
tsprec_one = prec_one(ts_y_act,ts_y_pred); tsrecl_one = recl_one(ts_y_act,ts_y_pred)

tsprec_ovll = tsprec_zero *frac_tszero + tsprec_one*frac_tsone
tsrecl_ovll = tsrecl_zero *frac_tszero + tsrecl_one*frac_tsone

print(paste("Random Forest Test accuracy:",ts_acc))
print(paste("Random Forest - Test Classification Report"))
print(paste("Zero_Precision",tsprec_zero,"Zero_Recall",tsrecl_zero))
print(paste("One_Precision",tsprec_one,"One_Recall",tsrecl_one))
print(paste("Overall_Precision",round(tsprec_ovll,4),"Overall_Recall",round(tsrecl_ovll,4)))


# Random Forest
library(randomForest)
set.seed(43)
rf_fit = randomForest(Attrition_ind~.,data = train_data,mtry=6,maxnodes= 64,
                      classwt = c(0.3,0.7),ntree=5000,nodesize = 1)
tr_y_pred = predict(rf_fit,data = train_data,type = "response")
ts_y_pred = predict(rf_fit,newdata = test_data,type = "response")

tr_y_act = train_data$Attrition_ind;ts_y_act = test_data$Attrition_ind

tr_tble = table(tr_y_act,tr_y_pred)
print(paste("Train Confusion Matrix"))
print(tr_tble)

tr_acc = accrcy(tr_y_act,tr_y_pred)
trprec_zero = prec_zero(tr_y_act,tr_y_pred); trrecl_zero = recl_zero(tr_y_act,tr_y_pred)
trprec_one = prec_one(tr_y_act,tr_y_pred); trrecl_one = recl_one(tr_y_act,tr_y_pred)

trprec_ovll = trprec_zero *frac_trzero + trprec_one*frac_trone
trrecl_ovll = trrecl_zero *frac_trzero + trrecl_one*frac_trone

print(paste("Random Forest Train accuracy:",tr_acc))
print(paste("Random Forest - Train Classification Report"))
print(paste("Zero_Precision",trprec_zero,"Zero_Recall",trrecl_zero))
print(paste("One_Precision",trprec_one,"One_Recall",trrecl_one))
print(paste("Overall_Precision",round(trprec_ovll,4),"Overall_Recall",round(trrecl_ovll,4)))

ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_y_act,ts_y_pred)
tsprec_zero = prec_zero(ts_y_act,ts_y_pred); tsrecl_zero = recl_zero(ts_y_act,ts_y_pred)
tsprec_one = prec_one(ts_y_act,ts_y_pred); tsrecl_one = recl_one(ts_y_act,ts_y_pred)

tsprec_ovll = tsprec_zero *frac_tszero + tsprec_one*frac_tsone
tsrecl_ovll = tsrecl_zero *frac_tszero + tsrecl_one*frac_tsone

print(paste("Random Forest Test accuracy:",ts_acc))
print(paste("Random Forest - Test Classification Report"))
print(paste("Zero_Precision",tsprec_zero,"Zero_Recall",tsrecl_zero))
print(paste("One_Precision",tsprec_one,"One_Recall",tsrecl_one))
print(paste("Overall_Precision",round(tsprec_ovll,4),"Overall_Recall",round(tsrecl_ovll,4)))



# Grid Search - Random Forest
library(e1071)
library(randomForest)
rf_grid = tune(randomForest,Attrition_ind~.,data = train_data,classwt = c(0.3,0.7),ranges = list(
  mtry = c(5,6),
  maxnodes = c(32,64),
  ntree = c(3000,5000),
  nodesize = c(1,2)
),
tunecontrol = tune.control(cross = 5)
)

print(paste("Best parameter from Grid Search"))
print(summary(rf_grid))

best_model = rf_grid$best.model

tr_y_pred = predict(best_model,data = train_data,type = "response")
ts_y_pred = predict(best_model,newdata = test_data,type = "response")

tr_y_act = train_data$Attrition_ind;ts_y_act = test_data$Attrition_ind

tr_tble = table(tr_y_act,tr_y_pred)
print(paste("Random Forest Grid search Train Confusion Matrix"))
print(tr_tble)

tr_acc = accrcy(tr_y_act,tr_y_pred)
trprec_zero = prec_zero(tr_y_act,tr_y_pred); trrecl_zero = recl_zero(tr_y_act,tr_y_pred)
trprec_one = prec_one(tr_y_act,tr_y_pred); trrecl_one = recl_one(tr_y_act,tr_y_pred)

trprec_ovll = trprec_zero *frac_trzero + trprec_one*frac_trone
trrecl_ovll = trrecl_zero *frac_trzero + trrecl_one*frac_trone

print(paste("Random Forest Grid Search Train accuracy:",tr_acc))
print(paste("Random Forest Grid Search - Train Classification Report"))
print(paste("Zero_Precision",trprec_zero,"Zero_Recall",trrecl_zero))
print(paste("One_Precision",trprec_one,"One_Recall",trrecl_one))
print(paste("Overall_Precision",round(trprec_ovll,4),"Overall_Recall",round(trrecl_ovll,4)))

ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Random Forest Grid search Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_y_act,ts_y_pred)
tsprec_zero = prec_zero(ts_y_act,ts_y_pred); tsrecl_zero = recl_zero(ts_y_act,ts_y_pred)
tsprec_one = prec_one(ts_y_act,ts_y_pred); tsrecl_one = recl_one(ts_y_act,ts_y_pred)

tsprec_ovll = tsprec_zero *frac_tszero + tsprec_one*frac_tsone
tsrecl_ovll = tsrecl_zero *frac_tszero + tsrecl_one*frac_tsone

print(paste("Random Forest Grid Search Test accuracy:",ts_acc))
print(paste("Random Forest Grid Search - Test Classification Report"))
print(paste("Zero_Precision",tsprec_zero,"Zero_Recall",tsrecl_zero))
print(paste("One_Precision",tsprec_one,"One_Recall",tsrecl_one))
print(paste("Overall_Precision",round(tsprec_ovll,4),"Overall_Recall",round(tsrecl_ovll,4)))


# Adaboost classifier using C5.0 with trails included for boosting
library(C50)

class_zero_wgt = 0.3
class_one_wgt = 1-class_zero_wgt
cstvr = class_one_wgt/class_zero_wgt

error_cost <- matrix(c(0, 1, cstvr, 0), nrow = 2)

# Fitting Adaboost model  
ada_fit = C5.0(train_data[-31],train_data$Attrition_ind,costs = error_cost,
                   trails = 5000,control = C5.0Control(minCases = 1))

summary(ada_fit)

tr_y_pred = predict(ada_fit, train_data,type = "class")
ts_y_pred = predict(ada_fit,test_data,type = "class")

tr_y_act = train_data$Attrition_ind;ts_y_act = test_data$Attrition_ind

tr_tble = table(tr_y_act,tr_y_pred)
print(paste("AdaBoost - Train Confusion Matrix"))
print(tr_tble)

tr_acc = accrcy(tr_y_act,tr_y_pred)
trprec_zero = prec_zero(tr_y_act,tr_y_pred); trrecl_zero = recl_zero(tr_y_act,tr_y_pred)
trprec_one = prec_one(tr_y_act,tr_y_pred); trrecl_one = recl_one(tr_y_act,tr_y_pred)

trprec_ovll = trprec_zero *frac_trzero + trprec_one*frac_trone
trrecl_ovll = trrecl_zero *frac_trzero + trrecl_one*frac_trone

print(paste("AdaBoost Train accuracy:",tr_acc))
print(paste("AdaBoost - Train Classification Report"))
print(paste("Zero_Precision",trprec_zero,"Zero_Recall",trrecl_zero))
print(paste("One_Precision",trprec_one,"One_Recall",trrecl_one))
print(paste("Overall_Precision",round(trprec_ovll,4),"Overall_Recall",round(trrecl_ovll,4)))

ts_tble = table(ts_y_act,ts_y_pred)
print(paste("AdaBoost - Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_y_act,ts_y_pred)
tsprec_zero = prec_zero(ts_y_act,ts_y_pred); tsrecl_zero = recl_zero(ts_y_act,ts_y_pred)
tsprec_one = prec_one(ts_y_act,ts_y_pred); tsrecl_one = recl_one(ts_y_act,ts_y_pred)

tsprec_ovll = tsprec_zero *frac_tszero + tsprec_one*frac_tsone
tsrecl_ovll = tsrecl_zero *frac_tszero + tsrecl_one*frac_tsone

print(paste("AdaBoost Test accuracy:",ts_acc))
print(paste("AdaBoost - Test Classification Report"))
print(paste("Zero_Precision",tsprec_zero,"Zero_Recall",tsrecl_zero))
print(paste("One_Precision",tsprec_one,"One_Recall",tsrecl_one))
print(paste("Overall_Precision",round(tsprec_ovll,4),"Overall_Recall",round(tsrecl_ovll,4)))


# Gradient boosting
library(gbm)
library(caret)

set.seed(43)

# Giving weights to all the observations in a way that total weights will be euqal 1
model_weights <- ifelse(train_data$Attrition_ind == "0",
                        (1/table(train_data$Attrition_ind)[1]) * 0.3,
                        (1/table(train_data$Attrition_ind)[2]) * 0.7)

# Setting parameters for GBM
grid <- expand.grid(n.trees = 5000, interaction.depth = 1, shrinkage = .04, n.minobsinnode = 1)

# Fitting the GBM model
gbm_fit <- train(Attrition_ind ~ ., data = train_data, method = "gbm", weights = model_weights,
                 tuneGrid=grid,verbose = FALSE)
# To print variable importance plot
summary(gbm_fit)

tr_y_pred = predict(gbm_fit, train_data,type = "raw")
ts_y_pred = predict(gbm_fit,test_data,type = "raw")

tr_y_act = train_data$Attrition_ind;ts_y_act = test_data$Attrition_ind

tr_tble = table(tr_y_act,tr_y_pred)
print(paste("Gradient Boosting - Train Confusion Matrix"))
print(tr_tble)

tr_acc = accrcy(tr_y_act,tr_y_pred)
trprec_zero = prec_zero(tr_y_act,tr_y_pred); trrecl_zero = recl_zero(tr_y_act,tr_y_pred)
trprec_one = prec_one(tr_y_act,tr_y_pred); trrecl_one = recl_one(tr_y_act,tr_y_pred)

trprec_ovll = trprec_zero *frac_trzero + trprec_one*frac_trone
trrecl_ovll = trrecl_zero *frac_trzero + trrecl_one*frac_trone

print(paste("Gradient Boosting Train accuracy:",tr_acc))
print(paste("Gradient Boosting - Train Classification Report"))
print(paste("Zero_Precision",trprec_zero,"Zero_Recall",trrecl_zero))
print(paste("One_Precision",trprec_one,"One_Recall",trrecl_one))
print(paste("Overall_Precision",round(trprec_ovll,4),"Overall_Recall",round(trrecl_ovll,4)))

ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Gradient Boosting - Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_y_act,ts_y_pred)
tsprec_zero = prec_zero(ts_y_act,ts_y_pred); tsrecl_zero = recl_zero(ts_y_act,ts_y_pred)
tsprec_one = prec_one(ts_y_act,ts_y_pred); tsrecl_one = recl_one(ts_y_act,ts_y_pred)

tsprec_ovll = tsprec_zero *frac_tszero + tsprec_one*frac_tsone
tsrecl_ovll = tsrecl_zero *frac_tszero + tsrecl_one*frac_tsone

print(paste("Gradient Boosting Test accuracy:",ts_acc))
print(paste("Gradient Boosting - Test Classification Report"))
print(paste("Zero_Precision",tsprec_zero,"Zero_Recall",tsrecl_zero))
print(paste("One_Precision",tsprec_one,"One_Recall",tsrecl_one))
print(paste("Overall_Precision",round(tsprec_ovll,4),"Overall_Recall",round(tsrecl_ovll,4)))


# Use the following code for performing cross validation on data - At the moment commented though
#fitControl <- trainControl(method = "repeatedcv", number = 4, repeats = 4)

# gbmFit1 <- train(Attrition_ind ~ ., data = train_data, method = "gbm", 
#                  trControl = fitControl,tuneGrid=grid,verbose = FALSE)



# Xgboost Classifier
library(xgboost); library(caret)

hrattr_data = read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
str(hrattr_data); summary(hrattr_data)

# Target variable creation
hrattr_data$Attrition_ind = 0;
hrattr_data$Attrition_ind[hrattr_data$Attrition=="Yes"]=1

# Columns to be removed due to no change in its value across observations
remove_cols = c("EmployeeCount","EmployeeNumber","Over18","StandardHours","Attrition")
hrattr_data_new = hrattr_data[,!(names(hrattr_data) %in% remove_cols)]


# List of  variables with continuous values
continuous_columns = c('Age','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction',
                       'HourlyRate', 'JobInvolvement', 'JobLevel','JobSatisfaction','MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
                       'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction','StockOptionLevel', 'TotalWorkingYears', 
                       'TrainingTimesLastYear','WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                       'YearsWithCurrManager')


# list of categorical variables
ohe_feats = c('BusinessTravel', 'Department', 'EducationField','Gender',
              'JobRole','MaritalStatus','OverTime')

# one-hot-encoding categorical features
dummies <- dummyVars(~ BusinessTravel+Department+EducationField+Gender+JobRole+MaritalStatus+OverTime, data = hrattr_data_new)
df_all_ohe <- as.data.frame(predict(dummies, newdata = hrattr_data_new))

# Cleaning column names & replace . with _
colClean <- function(x){ colnames(x) <- gsub("\\.", "_", colnames(x)); x } 
df_all_ohe = colClean(df_all_ohe)

hrattr_data_new$Attrition_ind = as.integer(hrattr_data_new$Attrition_ind)

# Combining both continuous & dummy variables from categories
hrattr_data_v3 = cbind(df_all_ohe,hrattr_data_new[,(names(hrattr_data_new) %in% continuous_columns)],
                       hrattr_data_new$Attrition_ind)

names(hrattr_data_v3)[52] = "Attrition_ind"

# Train & Test split based on 70% & 30% 
set.seed(123)
numrow = nrow(hrattr_data_v3)
trnind = sample(1:numrow,size = as.integer(0.7*numrow))
train_data = hrattr_data_v3[trnind,]
test_data = hrattr_data_v3[-trnind,]

# Custom functions for calculation of Precision & Recall 
frac_trzero = (table(train_data$Attrition_ind)[[1]])/nrow(train_data)
frac_trone = (table(train_data$Attrition_ind)[[2]])/nrow(train_data)

frac_tszero = (table(test_data$Attrition_ind)[[1]])/nrow(test_data)
frac_tsone = (table(test_data$Attrition_ind)[[2]])/nrow(test_data)

prec_zero <- function(act,pred){  tble = table(act,pred)
return( round( tble[1,1]/(tble[1,1]+tble[2,1]),4)  ) }

prec_one <- function(act,pred){ tble = table(act,pred)
return( round( tble[2,2]/(tble[2,2]+tble[1,2]),4)   ) }

recl_zero <- function(act,pred){tble = table(act,pred)
return( round( tble[1,1]/(tble[1,1]+tble[1,2]),4)   ) }

recl_one <- function(act,pred){ tble = table(act,pred)
return( round( tble[2,2]/(tble[2,2]+tble[2,1]),4)  ) }

accrcy <- function(act,pred){ tble = table(act,pred)
return( round((tble[1,1]+tble[2,2])/sum(tble),4)) }

y = train_data$Attrition_ind

# XGBoost Classifier Training
xgb <- xgboost(data = data.matrix(train_data[,-52]),label = y,eta = 0.04,
               max_depth = 2, nround=5000, subsample = 0.5,
               colsample_bytree = 0.5, seed = 1, eval_metric = "logloss",
               objective = "binary:logistic",nthread = 3
)

# XGBoost value prediction on train & test data
tr_y_pred_prob <- predict(xgb, data.matrix(train_data[,-52]))
tr_y_pred <- as.numeric(tr_y_pred_prob > 0.5)

ts_y_pred_prob <- predict(xgb, data.matrix(test_data[,-52]))
ts_y_pred <- as.numeric(ts_y_pred_prob > 0.5)
tr_y_act = train_data$Attrition_ind;ts_y_act = test_data$Attrition_ind
tr_tble = table(tr_y_act,tr_y_pred)

# XGBoost Metric predictions on Train Data
print(paste("Xgboost - Train Confusion Matrix"))
print(tr_tble)

tr_acc = accrcy(tr_y_act,tr_y_pred)
trprec_zero = prec_zero(tr_y_act,tr_y_pred); trrecl_zero = recl_zero(tr_y_act,tr_y_pred)
trprec_one = prec_one(tr_y_act,tr_y_pred); trrecl_one = recl_one(tr_y_act,tr_y_pred)
trprec_ovll = trprec_zero *frac_trzero + trprec_one*frac_trone
trrecl_ovll = trrecl_zero *frac_trzero + trrecl_one*frac_trone

print(paste("Xgboost Train accuracy:",tr_acc))
print(paste("Xgboost - Train Classification Report"))
print(paste("Zero_Precision",trprec_zero,"Zero_Recall",trrecl_zero))
print(paste("One_Precision",trprec_one,"One_Recall",trrecl_one))
print(paste("Overall_Precision",round(trprec_ovll,4),"Overall_Recall",round(trrecl_ovll,4)))


# XGBoost Metric predictions on Test Data
ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Xgboost - Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_y_act,ts_y_pred)
tsprec_zero = prec_zero(ts_y_act,ts_y_pred); tsrecl_zero = recl_zero(ts_y_act,ts_y_pred)
tsprec_one = prec_one(ts_y_act,ts_y_pred); tsrecl_one = recl_one(ts_y_act,ts_y_pred)
tsprec_ovll = tsprec_zero *frac_tszero + tsprec_one*frac_tsone
tsrecl_ovll = tsrecl_zero *frac_tszero + tsrecl_one*frac_tsone

print(paste("Xgboost Test accuracy:",ts_acc))
print(paste("Xgboost - Test Classification Report"))
print(paste("Zero_Precision",tsprec_zero,"Zero_Recall",tsrecl_zero))
print(paste("One_Precision",tsprec_one,"One_Recall",tsrecl_one))
print(paste("Overall_Precision",round(tsprec_ovll,4),"Overall_Recall",round(tsrecl_ovll,4)))



# Ensemble of Ensembles
rm(list = ls())

setwd("D:\\Book writing\\Codes\\Chapter 4")

hrattr_data = read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
str(hrattr_data)
summary(hrattr_data)

hrattr_data$Attrition_ind = 0;hrattr_data$Attrition_ind[hrattr_data$Attrition=="Yes"]=1
hrattr_data$Attrition_ind = as.factor(hrattr_data$Attrition_ind)

remove_cols = c("EmployeeCount","EmployeeNumber","Over18","StandardHours","Attrition")
hrattr_data_new = hrattr_data[,!(names(hrattr_data) %in% remove_cols)]

set.seed(123)
numrow = nrow(hrattr_data_new)
trnind = sample(1:numrow,size = as.integer(0.7*numrow))
train_data = hrattr_data_new[trnind,]
test_data = hrattr_data_new[-trnind,]

frac_trzero = (table(train_data$Attrition_ind)[[1]])/nrow(train_data)
frac_trone = (table(train_data$Attrition_ind)[[2]])/nrow(train_data)

frac_tszero = (table(test_data$Attrition_ind)[[1]])/nrow(test_data)
frac_tsone = (table(test_data$Attrition_ind)[[2]])/nrow(test_data)

prec_zero <- function(act,pred){  tble = table(act,pred)
return( round( tble[1,1]/(tble[1,1]+tble[2,1]),4)  ) }

prec_one <- function(act,pred){ tble = table(act,pred)
return( round( tble[2,2]/(tble[2,2]+tble[1,2]),4)   ) }

recl_zero <- function(act,pred){tble = table(act,pred)
return( round( tble[1,1]/(tble[1,1]+tble[1,2]),4)   ) }

recl_one <- function(act,pred){ tble = table(act,pred)
return( round( tble[2,2]/(tble[2,2]+tble[2,1]),4)  ) }

accrcy <- function(act,pred){ tble = table(act,pred)
return( round((tble[1,1]+tble[2,2])/sum(tble),4)) }


# Ensemble of Ensembles with different type of Classifiers
train_data$Attrition_ind = as.factor(train_data$Attrition_ind)

# Classifier 1 - Logistic Regression
glm_fit = glm(Attrition_ind ~.,family = "binomial",data = train_data)
glm_probs = predict(glm_fit,newdata = train_data,type = "response")

# Classifier 2 - Decision Tree classifier
library(C50)
dtree_fit = C5.0(train_data[-31],train_data$Attrition_ind,
                 control = C5.0Control(minCases = 1))
dtree_probs = predict(dtree_fit,newdata = train_data,type = "prob")[,2]
                      
# Classifier 3 - Random Forest
library(randomForest)
rf_fit = randomForest(Attrition_ind~.,data = train_data,mtry=6,maxnodes= 64,ntree=5000,nodesize = 1)
rf_probs = predict(rf_fit,newdata = train_data,type = "prob")[,2]


# Classifier 4 - Adaboost
ada_fit = C5.0(train_data[-31],train_data$Attrition_ind,trails = 5000,control = C5.0Control(minCases = 1))
ada_probs = predict(ada_fit,newdata = train_data,type = "prob")[,2]

# Ensemble of Models
ensemble = data.frame(glm_probs,dtree_probs,rf_probs,ada_probs)
ensemble = cbind(ensemble,train_data$Attrition_ind)
names(ensemble)[5] = "Attrition_ind"
rownames(ensemble) <- 1:nrow(ensemble)

# Meta-classifier on top of individual classifiers
meta_clf = glm(Attrition_ind~.,data = ensemble,family = "binomial")
meta_probs = predict(meta_clf, ensemble,type = "response")

ensemble$pred_class = 0
ensemble$pred_class[meta_probs>0.5]=1

# Train confusion & accuracy metrics
tr_y_pred = ensemble$pred_class
tr_y_act = train_data$Attrition_ind;ts_y_act = test_data$Attrition_ind
tr_tble = table(tr_y_act,tr_y_pred)
print(paste("Ensemble - Train Confusion Matrix"))
print(tr_tble)

tr_acc = accrcy(tr_y_act,tr_y_pred)
print(paste("Ensemble Train accuracy:",tr_acc))

# Now verifing on test data
glm_probs = predict(glm_fit,newdata = test_data,type = "response")
dtree_probs = predict(dtree_fit,newdata = test_data,type = "prob")[,2]
rf_probs = predict(rf_fit,newdata = test_data,type = "prob")[,2]
ada_probs = predict(ada_fit,newdata = test_data,type = "prob")[,2]

ensemble_test = data.frame(glm_probs,dtree_probs,rf_probs,ada_probs)
ensemble_test = cbind(ensemble_test,test_data$Attrition_ind)
names(ensemble_test)[5] = "Attrition_ind"

rownames(ensemble_test) <- 1:nrow(ensemble_test)
meta_test_probs = predict(meta_clf,newdata = ensemble_test,type = "response")
ensemble_test$pred_class = 0
ensemble_test$pred_class[meta_test_probs>0.5]=1

# Test confusion & accuracy metrics
ts_y_pred = ensemble_test$pred_class
ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Ensemble - Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_y_act,ts_y_pred)
print(paste("Ensemble Test accuracy:",ts_acc))














