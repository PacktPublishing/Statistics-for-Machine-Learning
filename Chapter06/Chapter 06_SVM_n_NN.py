

import os
""" First change the following directory link to where all input files do exist """
os.chdir("D:\\Book writing\\Codes\\Chapter 6")



import pandas as pd
letterdata = pd.read_csv("letterdata.csv")
print (letterdata.head())

x_vars = letterdata.drop(['letter'],axis=1)
y_var = letterdata["letter"]

y_var = y_var.replace({'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,
'K':11,'L':12,'M':13,'N':14,'O':15,'P':16,'Q':17,'R':18,'S':19,'T':20,
'U':21,'V':22,'W':23,'X':24,'Y':25,'Z':26})

from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_vars,y_var,train_size = 0.7,random_state=42)


# Linear Classifier
from sklearn.svm import SVC
svm_fit = SVC(kernel='linear',C=1.0,random_state=43)
svm_fit.fit(x_train,y_train)

print ("\nSVM Linear Classifier - Train Confusion Matrix\n\n",pd.crosstab(y_train,svm_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]) )     
print ("\nSVM Linear Classifier - Train accuracy:",round(accuracy_score(y_train,svm_fit.predict(x_train)),3))
print ("\nSVM Linear Classifier - Train Classification Report\n",classification_report(y_train,svm_fit.predict(x_train)))

print ("\n\nSVM Linear Classifier - Test Confusion Matrix\n\n",pd.crosstab(y_test,svm_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nSVM Linear Classifier - Test accuracy:",round(accuracy_score(y_test,svm_fit.predict(x_test)),3))
print ("\nSVM Linear Classifier - Test Classification Report\n",classification_report(y_test,svm_fit.predict(x_test)))


#Polynomial Kernel
svm_poly_fit = SVC(kernel='poly',C=1.0,degree=2)
svm_poly_fit.fit(x_train,y_train)

print ("\nSVM Polynomial Kernel Classifier - Train Confusion Matrix\n\n",pd.crosstab(y_train,svm_poly_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]) )     
print ("\nSVM Polynomial Kernel Classifier - Train accuracy:",round(accuracy_score(y_train,svm_poly_fit.predict(x_train)),3))
print ("\nSVM Polynomial Kernel Classifier - Train Classification Report\n",classification_report(y_train,svm_poly_fit.predict(x_train)))

print ("\n\nSVM Polynomial Kernel Classifier - Test Confusion Matrix\n\n",pd.crosstab(y_test,svm_poly_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nSVM Polynomial Kernel Classifier - Test accuracy:",round(accuracy_score(y_test,svm_poly_fit.predict(x_test)),3))
print ("\nSVM Polynomial Kernel Classifier - Test Classification Report\n",classification_report(y_test,svm_poly_fit.predict(x_test)))


#RBF Kernel
svm_rbf_fit = SVC(kernel='rbf',C=1.0, gamma=0.1)
svm_rbf_fit.fit(x_train,y_train)

print ("\nSVM RBF Kernel Classifier - Train Confusion Matrix\n\n",pd.crosstab(y_train,svm_rbf_fit.predict(x_train),rownames = ["Actuall"],colnames = ["Predicted"]) )     
print ("\nSVM RBF Kernel Classifier - Train accuracy:",round(accuracy_score(y_train,svm_rbf_fit.predict(x_train)),3))
print ("\nSVM RBF Kernel Classifier - Train Classification Report\n",classification_report(y_train,svm_rbf_fit.predict(x_train)))

print ("\n\nSVM RBF Kernel Classifier - Test Confusion Matrix\n\n",pd.crosstab(y_test,svm_rbf_fit.predict(x_test),rownames = ["Actuall"],colnames = ["Predicted"]))      
print ("\nSVM RBF Kernel Classifier - Test accuracy:",round(accuracy_score(y_test,svm_rbf_fit.predict(x_test)),3))
print ("\nSVM RBF Kernel Classifier - Test Classification Report\n",classification_report(y_test,svm_rbf_fit.predict(x_test)))



# Grid Search - RBF Kernel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV

pipeline = Pipeline([('clf',SVC(kernel='rbf',C=1,gamma=0.1 ))])

parameters = {'clf__C':(0.1,0.3,1,3,10,30),
              'clf__gamma':(0.001,0.01,0.1,0.3,1)}

grid_search_rbf = GridSearchCV(pipeline,parameters,n_jobs=-1,cv=5,verbose=1,scoring='accuracy')
grid_search_rbf.fit(x_train,y_train)


print ('RBF Kernel Grid Search Best Training score: %0.3f' % grid_search_rbf.best_score_)
print ('RBF Kernel Grid Search Best parameters set:')
best_parameters = grid_search_rbf.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ('\t%s: %r' % (param_name, best_parameters[param_name]))

predictions = grid_search_rbf.predict(x_test)

print ("\nRBF Kernel Grid Search - Testing accuracy:",round(accuracy_score(y_test, predictions),4))
print ("\nRBF Kernel Grid Search - Test Classification Report\n",classification_report(y_test, predictions))
print ("\n\nRBF Kernel Grid Search- Test Confusion Matrix\n\n",pd.crosstab(y_test, predictions,rownames = ["Actuall"],colnames = ["Predicted"]))      






# Neural Networks -  Classifying hand-written digits using Scikit Learn

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier
digits = load_digits()
X = digits.data
y = digits.target


# Checking dimensions
print (X.shape)
print (y.shape)

# Plotting first digit
import matplotlib.pyplot as plt 
plt.matshow(digits.images[0]) 
plt.show() 

#X_df = pd.DataFrame(X)
#y_df = pd.DataFrame(y)
#y_df.columns = ['target']
#digitdata = pd.concat([y_df,X_df],axis=1)
#digitdata.to_csv("digitsdata.csv",index= False)

from sklearn.model_selection import train_test_split
x_vars_stdscle = StandardScaler().fit_transform(X)
x_train,x_test,y_train,y_test = train_test_split(x_vars_stdscle,y,train_size = 0.7,random_state=42)


# Grid Search - Neural Network 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score,classification_report

pipeline = Pipeline([('mlp',MLPClassifier(hidden_layer_sizes= (100,50,),activation='relu',
            solver='adam',alpha=0.0001,max_iter=300 ))  ])

parameters = {'mlp__alpha':(0.001,0.01,0.1,0.3,0.5,1.0),
              'mlp__max_iter':(100,200,300)}

grid_search_nn = GridSearchCV(pipeline,parameters,n_jobs=-1,cv=5,verbose=1,scoring='accuracy')
grid_search_nn.fit(x_train,y_train)

print ('\n\nNeural Network Best Training score: %0.3f' % grid_search_nn.best_score_)
print ('\nNeural Network Best parameters set:')
best_parameters = grid_search_nn.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ('\t%s: %r' % (param_name, best_parameters[param_name]))

predictions_train = grid_search_nn.predict(x_train)
predictions_test = grid_search_nn.predict(x_test)

print ("\nNeural Network Training accuracy:",round(accuracy_score(y_train, predictions_train),4))
print ("\nNeural Network Complete report of Training data\n",classification_report(y_train, predictions_train))
print ("\n\nNeural Network Grid Search- Train Confusion Matrix\n\n",pd.crosstab(y_train, predictions_train,rownames = ["Actuall"],colnames = ["Predicted"]))      

print ("\n\nNeural Network Testing accuracy:",round(accuracy_score(y_test, predictions_test),4))
print ("\nNeural Network Complete report of Testing data\n",classification_report(y_test, predictions_test))
print ("\n\nNeural Network Grid Search- Test Confusion Matrix\n\n",pd.crosstab(y_test, predictions_test,rownames = ["Actuall"],colnames = ["Predicted"]))      







# Neural Networks -  Classifying hand-written digits using Keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adadelta,Adam,RMSprop
from keras.utils import np_utils


digits = load_digits()
X = digits.data
y = digits.target

print (X.shape)
print (y.shape)

print ("\nPrinting first digit")
plt.matshow(digits.images[0]) 
plt.show() 


x_vars_stdscle = StandardScaler().fit_transform(X)
x_train,x_test,y_train,y_test = train_test_split(x_vars_stdscle,y,train_size = 0.7,random_state=42)

# Definiting hyper parameters
np.random.seed(1337) 
nb_classes = 10
batch_size = 128
nb_epochs = 200

Y_train = np_utils.to_categorical(y_train, nb_classes)

print (Y_train.shape)

print (y_train[0])
print (Y_train[0])


#Deep Layer Model building in Keras

model = Sequential()

model.add(Dense(100,input_shape= (64,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')


# Model Training
model.fit(x_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,verbose=1)


#Model Prediction
y_train_predclass = model.predict_classes(x_train,batch_size=batch_size)
y_test_predclass = model.predict_classes(x_test,batch_size=batch_size)

print ("\n\nDeep Neural Network  - Train accuracy:"),(round(accuracy_score(y_train,y_train_predclass),3))

print ("\nDeep Neural Network  - Train Classification Report")
print (classification_report(y_train,y_train_predclass))

print ("\nDeep Neural Network - Train Confusion Matrix\n")
print (pd.crosstab(y_train,y_train_predclass,rownames = ["Actuall"],colnames = ["Predicted"]) )  


print ("\nDeep Neural Network  - Test accuracy:"),(round(accuracy_score(y_test,y_test_predclass),3))

print ("\nDeep Neural Network  - Test Classification Report")
print (classification_report(y_test,y_test_predclass))

print ("\nDeep Neural Network - Test Confusion Matrix\n")
print (pd.crosstab(y_test,y_test_predclass,rownames = ["Actuall"],colnames = ["Predicted"]) )





