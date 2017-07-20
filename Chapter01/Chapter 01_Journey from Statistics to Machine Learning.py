

import os


""" First change the following directory link to where all input files do exist """

os.chdir("D:\Book writing\Codes\Chapter 1")



import numpy as np
from scipy import stats


data = np.array([4,5,1,2,7,2,6,9,3])

# Calculate Mean
dt_mean = np.mean(data) ; print ("Mean :",round(dt_mean,2))
              
# Calculate Median                 
dt_median = np.median(data) ; print ("Median :",dt_median)        

# Calculate Mode                     
dt_mode =  stats.mode(data); print ("Mode :",dt_mode[0][0])                   


# Deviance calculations

import numpy as np
from statistics import variance,stdev

game_points = np.array([35,56,43,59,63,79,35,41,64,43,93,60,77,24,82])

# Calculate Variance
dt_var = variance(game_points) ; print ("Sample variance:", round(dt_var,2))

# Calculate Standard Deviation
dt_std = stdev(game_points) ; print ("Sample std.dev:",round(dt_std,2))
               
# Calculate Range
dt_rng = np.max(game_points,axis=0) - np.min(game_points,axis=0) ; print ("Range:",dt_rng)


#Calculate percentiles
print ("Quantiles:")
for val in [20,80,100]:
    dt_qntls = np.percentile(game_points,val) 
    print (str(val)+"%" ,dt_qntls)
                                
# Calculate IQR                           
q75, q25 = np.percentile(game_points, [75 ,25]); print ("Inter quartile range:",q75-q25 )
                        
                        
# Hypothesis testing
#import scipy                       
          
from scipy import stats              

xbar = 990; mu0 = 1000; s = 12.5; n = 30
# Test Statistic
t_smple  = (xbar-mu0)/(s/np.sqrt(float(n))); print ("Test Statistic:",round(t_smple,2))
# Critical value from t-table
alpha = 0.05
t_alpha = stats.t.ppf(alpha,n-1); print ("Critical value from t-table:",round(t_alpha,3))          
#Lower tail p-value from t-table                        
p_val = stats.t.sf(np.abs(t_smple), n-1); print ("Lower tail p-value from t-table", p_val)                        
                      

# Normal Distribution
from scipy import stats
xbar = 67; mu0 = 52; s = 16.3

# Calculating z-score
z = (67-52)/16.3

# Calculating probability under the curve    
p_val = 1- stats.norm.cdf(z)
print ("Prob. to score more than 67 is ",round(p_val*100,2),"%")



# Chi-square independence test
import pandas as pd
from scipy import stats

survey = pd.read_csv("survey.csv")  
# Tabulating 2 variables with row & column variables respectively
survey_tab = pd.crosstab(survey.Smoke, survey.Exer, margins = True)
# Creating observed table for analysis
observed = survey_tab.ix[0:4,0:3] 

contg = stats.chi2_contingency(observed= observed)
p_value = round(contg[1],3)
print ("P-value is: ",p_value)



#ANOVA
import pandas as pd
from scipy import stats

fetilizers = pd.read_csv("fetilizers.csv")

one_way_anova = stats.f_oneway(fetilizers["fertilizer1"], fetilizers["fertilizer2"], fetilizers["fertilizer3"])

print ("Statistic :", round(one_way_anova[0],2),", p-value :",round(one_way_anova[1],3))





# Train & Test split
import pandas as pd      
from sklearn.model_selection import train_test_split              
                        
original_data = pd.read_csv("mtcars.csv")     

train_data,test_data = train_test_split(original_data,train_size = 0.7,random_state=42)


# Linear Regressio vs. Gradient Descent             
               
import numpy as np                        
import pandas as pd
                       
train_data = pd.read_csv("mtcars.csv")                       
                        
X = np.array(train_data["hp"])  ; y = np.array(train_data["mpg"]) 
X = X.reshape(32,1); y = y.reshape(32,1)

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept = True) 
 
model.fit(X,y)       
print ("Linear Regression Results")        
print ("Intercept",model.intercept_[0] ,"Coefficient",model.coef_[0])   
                   

def gradient_descent(x, y,learn_rate, conv_threshold,batch_size,max_iter):    
    converged = False
    iter = 0
    m = batch_size 
 
    t0 = np.random.random(x.shape[1])
    t1 = np.random.random(x.shape[1])

    MSE = (sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])/ m)    

    while not converged:        
        grad0 = 1.0/m * sum([(t0 + t1*x[i] - y[i]) for i in range(m)]) 
        grad1 = 1.0/m * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)])

        temp0 = t0 - learn_rate * grad0
        temp1 = t1 - learn_rate * grad1
    
        t0 = temp0
        t1 = temp1

        MSE_New = (sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] ) / m)

        if abs(MSE - MSE_New ) <= conv_threshold:
            print ('Converged, iterations: ', iter)
            converged = True
    
        MSE = MSE_New   
        iter += 1 
    
        if iter == max_iter:
            print ('Max interactions reached')
            converged = True

    return t0,t1

if __name__ == '__main__':
    Inter, Coeff = gradient_descent(x = X,y = y,learn_rate=0.00003 ,conv_threshold=1e-8, batch_size=32,max_iter=1500000)
    print ("Gradient Descent Results")
    print (('Intercept = %s Coefficient = %s') %(Inter, Coeff)) 




# Train Validation Test split      

import pandas as pd      
from sklearn.model_selection import train_test_split              
                        
original_data = pd.read_csv("mtcars.csv")                   
 

def data_split(dat,trf = 0.5,vlf=0.25,tsf = 0.25):
    nrows = dat.shape[0]    
    trnr = int(nrows*trf)
    vlnr = int(nrows*vlf)    
    
    tr_data,rmng = train_test_split(dat,train_size = trnr,random_state=42)
    vl_data, ts_data = train_test_split(rmng,train_size = vlnr,random_state=45)  
    
    return (tr_data,vl_data,ts_data)


train_data, validation_data, test_data = data_split(original_data,trf=0.5,vlf=0.25,tsf=0.25)





# Grid search on Decision Trees
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.pipeline import Pipeline



input_data = pd.read_csv("ad.csv",header=None)                       

X_columns = set(input_data.columns.values)
y = input_data[len(input_data.columns.values)-1]
X_columns.remove(len(input_data.columns.values)-1)
X = input_data[list(X_columns)]

X_train, X_test,y_train,y_test = train_test_split(X,y,train_size = 0.7,random_state=33)

pipeline = Pipeline([
    ('clf', DecisionTreeClassifier(criterion='entropy'))
])
parameters = {
    'clf__max_depth': (50,100,150),
    'clf__min_samples_split': (2, 3),
    'clf__min_samples_leaf': (1, 2, 3)
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)

print ('\n Best score: \n', grid_search.best_score_)
print ('\n Best parameters set: \n')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ('\t%s: %r' % (param_name, best_parameters[param_name]))
print ("\n Confusion Matrix on Test data \n",confusion_matrix(y_test,y_pred))
print ("\n Test Accuracy \n",accuracy_score(y_test,y_pred))
print ("\nPrecision Recall f1 table \n",classification_report(y_test, y_pred))






























