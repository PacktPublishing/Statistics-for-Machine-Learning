
rm(list = ls())

# First change the following directory link to where all the input files do exist
setwd("D:\\Book writing\\Codes\\Chapter 3")

library(mctest)
library(dummies)
library(Information)
library(pROC)

credit_data = read.csv("credit_data.csv")
credit_data$class = credit_data$class-1 

# I.V Calculation
IV <- create_infotables(data=credit_data, y="class", parallel=FALSE)
for (i in 1:length(colnames(credit_data))-1){
  seca = IV[[1]][i][1]
  sum(seca[[1]][5])
  print(paste(colnames(credit_data)[i],",IV_Value:",round(sum(seca[[1]][5]),4)))
}

# Dummy variables creation
dummy_stseca =data.frame(dummy(credit_data$Status_of_existing_checking_account))
dummy_ch = data.frame(dummy(credit_data$Credit_history))
dummy_purpose =  data.frame(dummy(credit_data$Purpose))
dummy_savacc = data.frame(dummy(credit_data$Savings_Account))
dummy_presc = data.frame(dummy(credit_data$Present_Employment_since))
dummy_perssx = data.frame(dummy(credit_data$Personal_status_and_sex))
dummy_othdts = data.frame(dummy(credit_data$Other_debtors))
dummy_property = data.frame(dummy(credit_data$Property))
dummy_othinstpln = data.frame(dummy(credit_data$Other_installment_plans))
dummy_forgnwrkr = data.frame(dummy(credit_data$Foreign_worker))

# Cleaning the variables name from . to _
colClean <- function(x){ colnames(x) <- gsub("\\.", "_", colnames(x)); x } 
dummy_stseca = colClean(dummy_stseca) ;dummy_ch = colClean(dummy_ch) 
dummy_purpose = colClean(dummy_purpose); dummy_savacc= colClean(dummy_savacc)
dummy_presc= colClean(dummy_presc);dummy_perssx= colClean(dummy_perssx);
dummy_othdts= colClean(dummy_othdts);dummy_property= colClean(dummy_property);
dummy_othinstpln= colClean(dummy_othinstpln);dummy_forgnwrkr= colClean(dummy_forgnwrkr);


continuous_columns = c('Duration_in_month', 'Credit_amount','Installment_rate_in_percentage_of_disposable_income',
                      'Age_in_years','Number_of_existing_credits_at_this_bank')

credit_continuous = credit_data[,continuous_columns]
credit_data_new = cbind(dummy_stseca,dummy_ch,dummy_purpose,dummy_savacc,dummy_presc,dummy_perssx,
                        dummy_othdts,dummy_property,dummy_othinstpln,dummy_forgnwrkr,credit_continuous,credit_data$class)

colnames(credit_data_new)[51] <- "class"

# Setting seed for repeatability of results of train & test split
set.seed(123)
numrow = nrow(credit_data_new)
trnind = sample(1:numrow,size = as.integer(0.7*numrow))
train_data = credit_data_new[trnind,]
test_data = credit_data_new[-trnind,]

remove_cols_extra_dummy = c("Status_of_existing_checking_account_A11","Credit_history_A30",
                            "Purpose_A40","Savings_Account_A61","Present_Employment_since_A71","Personal_status_and_sex_A91",
                            "Other_debtors_A101","Property_A121","Other_installment_plans_A141","Foreign_worker_A201")

# Removing insignificant variables one by one
remove_cols_insig = c("Purpose_A46","Purpose_A45","Purpose_A44","Savings_Account_A63", "Other_installment_plans_A143",
                      "Property_A123","Status_of_existing_checking_account_A12",
                      "Present_Employment_since_A72","Present_Employment_since_A75",
                      "Present_Employment_since_A73","Credit_history_A32","Credit_history_A33",
                      "Purpose_A40","Present_Employment_since_A74","Purpose_A49","Purpose_A48",
                      "Property_A122","Personal_status_and_sex_A92","Foreign_worker_A202",
                      "Personal_status_and_sex_A94","Purpose_A42","Other_debtors_A102",
                      "Age_in_years","Savings_Account_A64","Savings_Account_A62",
                      "Savings_Account_A65", "Other_debtors_A103")

remove_cols = c(remove_cols_extra_dummy,remove_cols_insig)

glm_fit = glm(class ~.,family = "binomial",data = train_data[,!(names(train_data) %in% remove_cols)])
# Significance check - p_value
summary(glm_fit)

# Multi collinearity check - VIF
remove_cols_vif = c(remove_cols,"class")
vif_table = imcdiag(train_data[,!(names(train_data) %in% remove_cols_vif)],train_data$class,detr=0.001, conf=0.99)
vif_table  

# Predicting probabilities
train_data$glm_probs = predict(glm_fit,newdata = train_data,type = "response")
test_data$glm_probs = predict(glm_fit,newdata = test_data,type = "response")

# Area under ROC

ROC1 <- roc(as.factor(train_data$class),train_data$glm_probs)
plot(ROC1, col = "blue")
print(paste("Area under the curve",round(auc(ROC1),4))) 

# Actual prediction based on threshold tuning 
threshold_vals = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)
for (thld in threshold_vals){
  train_data$glm_pred = 0
  train_data$glm_pred[train_data$glm_probs>thld]=1
  
  tble = table(train_data$glm_pred,train_data$class)
  acc = (tble[1,1]+tble[2,2])/sum(tble)
  print(paste("Threshold",thld,"Train accuracy",round(acc,4)))
  
}

# Best threshold from above search is 0.5 with accuracy as 0.7841
best_threshold = 0.5

# Train confusion matrix & accuracy
train_data$glm_pred = 0
train_data$glm_pred[train_data$glm_probs>best_threshold]=1
tble = table(train_data$glm_pred,train_data$class)
acc = (tble[1,1]+tble[2,2])/sum(tble)
print(paste("Confusion Matrix - Train Data"))
print(tble)
print(paste("Train accuracy",round(acc,4)))

# Test confusion matrix & accuracy
test_data$glm_pred = 0
test_data$glm_pred[test_data$glm_probs>best_threshold]=1
tble_test = table(test_data$glm_pred,test_data$class)
acc_test = (tble_test[1,1]+tble_test[2,2])/sum(tble_test)
print(paste("Confusion Matrix - Test Data"))
print(tble_test)
print(paste("Test accuracy",round(acc_test,4)))



# Random Forest
library(randomForest)
library(e1071)

credit_data = read.csv("credit_data.csv")

credit_data$class = credit_data$class-1 
credit_data$class = as.factor(credit_data$class)

set.seed(123)
numrow = nrow(credit_data)
trnind = sample(1:numrow,size = as.integer(0.7*numrow))
train_data = credit_data[trnind,]
test_data = credit_data[-trnind,]

rf_fit = randomForest(class~.,data = train_data,mtry=4,maxnodes= 2000,ntree=1000,nodesize = 2)
rf_pred = predict(rf_fit,data = train_data,type = "response")
rf_predt = predict(rf_fit,newdata = test_data,type = "response")

tble = table(train_data$class,rf_pred)
tblet = table(test_data$class,rf_predt)

acc = (tble[1,1]+tble[2,2])/sum(tble)
acct = (tblet[1,1]+tblet[2,2])/sum(tblet)
print(paste("Train acc",round(acc,4),"Test acc",round(acct,4)))

# Grid Search
rf_grid = tune(randomForest,class~.,data = train_data,ranges = list(
              mtry = c(4,5),
              maxnodes = c(700,1000),
              ntree = c(1000,2000,3000),
              nodesize = c(1,2)
          ),
          tunecontrol = tune.control(cross = 5)
)

summary(rf_grid)

best_model = rf_grid$best.model
summary(best_model)

y_pred_train = predict(best_model,data = train_data)
train_conf_mat = table(train_data$class,y_pred_train)

print(paste("Train Confusion Matrix - Grid Search:"))
print(train_conf_mat)

train_acc = (train_conf_mat[1,1]+train_conf_mat[2,2])/sum(train_conf_mat)
print(paste("Train_accuracy-Grid Search:",round(train_acc,4)))

y_pred_test = predict(best_model,newdata = test_data)
test_conf_mat = table(test_data$class,y_pred_test)

print(paste("Test Confusion Matrix - Grid Search:"))
print(test_conf_mat)

test_acc = (test_conf_mat[1,1]+test_conf_mat[2,2])/sum(test_conf_mat)
print(paste("Test_accuracy-Grid Search:",round(test_acc,4)))

# Variable Importance
vari = varImpPlot(best_model)
print(paste("Variable Importance - Table"))
print(vari)



