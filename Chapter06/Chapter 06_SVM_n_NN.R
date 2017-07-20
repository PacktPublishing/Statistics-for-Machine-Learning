




rm(list = ls())

# First change the following directory link to where all the input files do exist
setwd("D:\\Book writing\\Codes\\Chapter 6")

letter_data = read.csv("letterdata.csv")

set.seed(123)
numrow = nrow(letter_data)
trnind = sample(1:numrow,size = as.integer(0.7*numrow))
train_data = letter_data[trnind,]
test_data = letter_data[-trnind,]


library(e1071)

accrcy <- function(matrx){ 
  return( sum(diag(matrx)/sum(matrx)))}

precsn <- function(matrx){
  return(diag(matrx) / rowSums(matrx))
}

recll <- function(matrx){
  return(diag(matrx) / colSums(matrx))
}




# SVM - Linear Kernel
svm_fit = svm(letter~.,data = train_data,kernel="linear",cost=1.0,scale = TRUE)

tr_y_pred = predict(svm_fit, train_data)
ts_y_pred = predict(svm_fit,test_data)

tr_y_act = train_data$letter;ts_y_act = test_data$letter

tr_tble = table(tr_y_act,tr_y_pred)
print(paste("Train Confusion Matrix"))
print(tr_tble)
tr_acc = accrcy(tr_tble)
print(paste("SVM Linear Kernel Train accuracy:",round(tr_acc,4)))

tr_prec = precsn(tr_tble)
print(paste("SVM Linear Kernel Train Precision:"))
print(tr_prec)

tr_rcl = recll(tr_tble)
print(paste("SVM Linear Kernel Train Recall:"))
print(tr_rcl)

ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_tble)
print(paste("SVM Linear Kernel Test accuracy:",round(ts_acc,4)))

ts_prec = precsn(ts_tble)
print(paste("SVM Linear Kernel Test Precision:"))
print(ts_prec)

ts_rcl = recll(ts_tble)
print(paste("SVM Linear Kernel Test Recall:"))
print(ts_rcl)


# SVM - Polynomial Kernel
svm_poly_fit = svm(letter~.,data = train_data,kernel="poly",cost=1.0,degree = 2  ,scale = TRUE)

tr_y_pred = predict(svm_poly_fit, train_data)
ts_y_pred = predict(svm_poly_fit,test_data)

tr_y_act = train_data$letter;ts_y_act = test_data$letter


tr_tble = table(tr_y_act,tr_y_pred)
print(paste("Train Confusion Matrix"))
print(tr_tble)
tr_acc = accrcy(tr_tble)
print(paste("SVM Polynomial Kernel Train accuracy:",round(tr_acc,4)))

tr_prec = precsn(tr_tble)
print(paste("SVM Polynomial Kernel Train Precision:"))
print(tr_prec)

tr_rcl = recll(tr_tble)
print(paste("SVM Polynomial Kernel Train Recall:"))
print(tr_rcl)

ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_tble)
print(paste("SVM Polynomial Kernel Test accuracy:",round(ts_acc,4)))

ts_prec = precsn(ts_tble)
print(paste("SVM Polynomial Kernel Test Precision:"))
print(ts_prec)

ts_rcl = recll(ts_tble)
print(paste("SVM Polynomial Kernel Test Recall:"))
print(ts_rcl)



# SVM - RBF Kernel
svm_rbf_fit = svm(letter~.,data = train_data,kernel="radial",cost=1.0,gamma = 0.2  ,scale = TRUE)

tr_y_pred = predict(svm_rbf_fit, train_data)
ts_y_pred = predict(svm_rbf_fit,test_data)

tr_y_act = train_data$letter;ts_y_act = test_data$letter

tr_tble = table(tr_y_act,tr_y_pred)
print(paste("Train Confusion Matrix"))
print(tr_tble)
tr_acc = accrcy(tr_tble)
print(paste("SVM RBF Kernel Train accuracy:",round(tr_acc,4)))

tr_prec = precsn(tr_tble)
print(paste("SVM RBF Kernel Train Precision:"))
print(tr_prec)

tr_rcl = recll(tr_tble)
print(paste("SVM RBF Kernel Train Recall:"))
print(tr_rcl)

ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_tble)
print(paste("SVM RBF Kernel Test accuracy:",round(ts_acc,4)))

ts_prec = precsn(ts_tble)
print(paste("SVM RBF Kernel Test Precision:"))
print(ts_prec)

ts_rcl = recll(ts_tble)
print(paste("SVM RBF Kernel Test Recall:"))
print(ts_rcl)



# Grid search - RBF Kernel
library(e1071)
svm_rbf_grid = tune(svm,letter~.,data = train_data,kernel="radial",scale=TRUE,ranges = list(
  cost = c(0.1,0.3,1,3,10,30),
  gamma = c(0.001,0.01,0.1,0.3,1)
  
),
tunecontrol = tune.control(cross = 5)
)

print(paste("Best parameter from Grid Search"))
print(summary(svm_rbf_grid))

best_model = svm_rbf_grid$best.model

tr_y_pred = predict(best_model,data = train_data,type = "response")
ts_y_pred = predict(best_model,newdata = test_data,type = "response")

tr_y_act = train_data$letter;ts_y_act = test_data$letter


tr_tble = table(tr_y_act,tr_y_pred)
print(paste("Train Confusion Matrix"))
print(tr_tble)
tr_acc = accrcy(tr_tble)
print(paste("SVM RBF Kernel Train accuracy:",round(tr_acc,4)))

tr_prec = precsn(tr_tble)
print(paste("SVM RBF Kernel Train Precision:"))
print(tr_prec)

tr_rcl = recll(tr_tble)
print(paste("SVM RBF Kernel Train Recall:"))
print(tr_rcl)

ts_tble = table(ts_y_act,ts_y_pred)
print(paste("Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_tble)
print(paste("SVM RBF Kernel Test accuracy:",round(ts_acc,4)))

ts_prec = precsn(ts_tble)
print(paste("SVM RBF Kernel Test Precision:"))
print(ts_prec)

ts_rcl = recll(ts_tble)
print(paste("SVM RBF Kernel Test Recall:"))
print(ts_rcl)




# Artificial Neural Networks
setwd("D:\\Book writing\\Codes\\Chapter 6")
digits_data = read.csv("digitsdata.csv")

remove_cols = c("target")
x_data = digits_data[,!(names(digits_data) %in% remove_cols)]
y_data = digits_data[,c("target")]


normalize <- function(x) {return((x - min(x)) / (max(x) - min(x)))}


data_norm <- as.data.frame(lapply(x_data, normalize))
data_norm <- replace(data_norm, is.na(data_norm), 0.0)
data_norm_v2 = data.frame(as.factor(y_data),data_norm)
names(data_norm_v2)[1] = "target"


set.seed(123)
numrow = nrow(data_norm_v2)
trnind = sample(1:numrow,size = as.integer(0.7*numrow))
train_data = data_norm_v2[trnind,]
test_data = data_norm_v2[-trnind,]

f <- as.formula(paste("target ~", paste(names(train_data)[!names(train_data) %in% "target"], collapse = " + ")))

library(nnet)
accuracy <- function(mat){return(sum(diag(mat)) / sum(mat))}

nnet_fit = nnet(f,train_data,size=c(9),maxit=200)
y_pred = predict(nnet_fit,newdata = test_data,type = "class")
tble = table(test_data$target,y_pred)
print(accuracy(tble))


#Plotting nnet from the github packages
require(RCurl)
root.url<-'https://gist.githubusercontent.com/fawda123'
raw.fun<-paste(
  root.url,
  '5086859/raw/cc1544804d5027d82b70e74b83b3941cd2184354/nnet_plot_fun.r',
  sep='/')
script<-getURL(raw.fun, ssl.verifypeer = FALSE)
eval(parse(text = script))
rm('script','raw.fun')

# Ploting the neural net
plot(nnet_fit)


# Grid Search - ANN
neurons = c(1,2,3,4,5,6,7,8,9,10,11,12,13)
iters = c(200,300,400,500,600,700,800,900)

initacc = 0

for(itr in iters){
  for(nd in neurons){
    nnet_fit = nnet(f,train_data,size=c(nd),maxit=itr,trace=FALSE)
    y_pred = predict(nnet_fit,newdata = test_data,type = "class")
    tble = table(test_data$target,y_pred)
    acc = accuracy(tble)
    
    if (acc>initacc){
      print(paste("Neurons",nd,"Iterations",itr,"Test accuracy",acc))
      initacc = acc
    }
    
  }
}



