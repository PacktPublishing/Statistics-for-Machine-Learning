
rm(list = ls())

# First change the following directory link to where all the input files do exist
setwd("D:\\Book writing\\Codes\\Chapter 2")



# Simple Linear Regression
wine_quality = read.csv("winequality-red.csv",header=TRUE,sep = ";",check.names = FALSE)
names(wine_quality) <- gsub(" ", "_", names(wine_quality))

set.seed(123)
numrow = nrow(wine_quality)
trnind = sample(1:numrow,size = as.integer(0.7*numrow))
train_data = wine_quality[trnind,]
test_data = wine_quality[-trnind,]

x_train = train_data$alcohol;y_train = train_data$quality
x_test = test_data$alcohol; y_test = test_data$quality

x_mean = mean(x_train); y_mean = mean(y_train)
x_var = sum((x_train - x_mean)**2) ; y_var = sum((y_train-y_mean)**2)
covariance = sum((x_train-x_mean)*(y_train-y_mean))

b1 = covariance/x_var  
b0 = y_mean - b1*x_mean

pred_y = b0+b1*x_test

R2 <- 1 - (sum((y_test-pred_y )^2)/sum((y_test-mean(y_test))^2))
print(paste("Test Adjusted R-squared :",round(R2,4)))




library(usdm)

# Multi linear Regression
wine_quality = read.csv("winequality-red.csv",header=TRUE,sep = ";",check.names = FALSE)
names(wine_quality) <- gsub(" ", "_", names(wine_quality))

set.seed(123)
numrow = nrow(wine_quality)
trnind = sample(1:numrow,size = as.integer(0.7*numrow))
train_data = wine_quality[trnind,]
test_data = wine_quality[-trnind,]

xvars = c("volatile_acidity","chlorides","free_sulfur_dioxide", 
           "total_sulfur_dioxide","pH","sulphates","alcohol")
yvar = "quality"

frmla = paste(yvar,"~",paste(xvars,collapse = "+"))
lr_fit = lm(as.formula(frmla),data = train_data)
print(summary(lr_fit))

#VIF calculation
wine_v2 = train_data[,xvars]
print(vif(wine_v2))

#Test prediction
pred_y = predict(lr_fit,newdata = test_data)
R2 <- 1 - (sum((test_data[,yvar]-pred_y )^2)/sum((test_data[,yvar]-mean(test_data[,yvar]))^2))
print(paste("Test Adjusted R-squared :",R2))




# xvars = c("fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide", 
#           "total_sulfur_dioxide","density","pH","sulphates","alcohol")







# Ridge regression
library(glmnet)

wine_quality = read.csv("winequality-red.csv",header=TRUE,sep = ";",check.names = FALSE)
names(wine_quality) <- gsub(" ", "_", names(wine_quality))

set.seed(123)
numrow = nrow(wine_quality)
trnind = sample(1:numrow,size = as.integer(0.7*numrow))
train_data = wine_quality[trnind,]; test_data = wine_quality[-trnind,]

xvars = c("fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide", 
           "total_sulfur_dioxide","density","pH","sulphates","alcohol")
yvar = "quality"

x_train = as.matrix(train_data[,xvars]);y_train = as.double(as.matrix(train_data[,yvar]))
x_test = as.matrix(test_data[,xvars])

print(paste("Ridge Regression"))
lambdas = c(1e-4,1e-3,1e-2,0.1,0.5,1.0,5.0,10.0)
initrsq = 0
for (lmbd in lambdas){
  ridge_fit = glmnet(x_train,y_train,alpha = 0,lambda = lmbd)
  pred_y = predict(ridge_fit,x_test)
  R2 <- 1 - (sum((test_data[,yvar]-pred_y )^2)/sum((test_data[,yvar]-mean(test_data[,yvar]))^2))
  
  if (R2 > initrsq){
    print(paste("Lambda:",lmbd,"Test Adjusted R-squared :",round(R2,4)))
    initrsq = R2
  }
}



# Lasso Regression
print(paste("Lasso Regression"))
lambdas = c(1e-4,1e-3,1e-2,0.1,0.5,1.0,5.0,10.0)
initrsq = 0
for (lmbd in lambdas){
  lasso_fit = glmnet(x_train,y_train,alpha = 1,lambda = lmbd)
  pred_y = predict(lasso_fit,x_test)
  R2 <- 1 - (sum((test_data[,yvar]-pred_y )^2)/sum((test_data[,yvar]-mean(test_data[,yvar]))^2))
  
  if (R2 > initrsq){
    print(paste("Lambda:",lmbd,"Test Adjusted R-squared :",round(R2,4)))
    initrsq = R2
  }
}


