


import os


""" First change the following directory link to where all input files do exist """
os.chdir("D:\\Book writing\\Codes\\Chapter 2")


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.model_selection import train_test_split    
#from sklearn.metrics import r2_score


wine_quality = pd.read_csv("winequality-red.csv",sep=';')  
# Step for converting white space in columns to _ value for better handling 
wine_quality.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)

# Simple Linear Regression - chart
model = sm.OLS(wine_quality['quality'],sm.add_constant(wine_quality['alcohol'])).fit()
 
print (model.summary())

plt.scatter(wine_quality['alcohol'],wine_quality['quality'],label = 'Actual Data')
plt.plot(wine_quality['alcohol'],model.params[0]+model.params[1]*wine_quality['alcohol'],
         c ='r',label="Regression fit")
plt.title('Wine Quality regressed on Alchohol')
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.show()


# Simple Linear Regression - Model fit
import pandas as pd
from sklearn.model_selection import train_test_split    
from sklearn.metrics import r2_score


wine_quality = pd.read_csv("winequality-red.csv",sep=';')  
wine_quality.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)

x_train,x_test,y_train,y_test = train_test_split(wine_quality['alcohol'],wine_quality["quality"],train_size = 0.7,random_state=42)

x_train = pd.DataFrame(x_train);x_test = pd.DataFrame(x_test)
y_train = pd.DataFrame(y_train);y_test = pd.DataFrame(y_test)

def mean(values):
    return round(sum(values)/float(len(values)),2)

alcohol_mean = mean(x_train['alcohol'])
quality_mean = mean(y_train['quality'])

alcohol_variance = round(sum((x_train['alcohol'] - alcohol_mean)**2),2)
quality_variance = round(sum((y_train['quality'] - quality_mean)**2),2)

covariance = round(sum((x_train['alcohol'] - alcohol_mean) * (y_train['quality'] - quality_mean )),2)
b1 = covariance/alcohol_variance
b0 = quality_mean - b1*alcohol_mean
print ("\n\nIntercept (B0):",round(b0,4),"Co-efficient (B1):",round(b1,4))
y_test["y_pred"] = pd.DataFrame(b0+b1*x_test['alcohol'])
R_sqrd = 1- ( sum((y_test['quality']-y_test['y_pred'])**2) / sum((y_test['quality'] - mean(y_test['quality']))**2 ))
print ("Test R-squared value:",round(R_sqrd,4))


# Plots - pair plots
eda_colnms = [ 'volatile_acidity',  'chlorides', 'sulphates', 'alcohol','quality']
sns.set(style='whitegrid',context = 'notebook')
sns.pairplot(wine_quality[eda_colnms],size = 2.5,x_vars= eda_colnms,y_vars=eda_colnms)
plt.show()



# Correlation coefficients
corr_mat = np.corrcoef(wine_quality[eda_colnms].values.T)
sns.set(font_scale=1)
full_mat = sns.heatmap(corr_mat, cbar=True, annot=True, square=True, fmt='.2f', 
                       annot_kws={'size': 15}, yticklabels=eda_colnms, xticklabels=eda_colnms)

plt.show()



# Multi linear regression model
colnms = [ 'volatile_acidity',  'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide',
 'pH', 'sulphates', 'alcohol']


pdx = wine_quality[colnms]
pdy = wine_quality["quality"]

x_train,x_test,y_train,y_test = train_test_split(pdx,pdy,train_size = 0.7,random_state=42)
x_train_new = sm.add_constant(x_train)
x_test_new = sm.add_constant(x_test)

#random.seed(434)
full_mod = sm.OLS(y_train,x_train_new)
full_res = full_mod.fit()
print ("\n \n",full_res.summary())


print ("\nVariance Inflation Factor")
cnames = x_train.columns
for i in np.arange(0,len(cnames)):
    xvars = list(cnames)
    yvar = xvars.pop(i)
    mod = sm.OLS(x_train[yvar],sm.add_constant(x_train_new[xvars]))
    res = mod.fit()
    vif = 1/(1-res.rsquared)
    print (yvar,round(vif,3))

# Predition of data    
y_pred = full_res.predict(x_test_new)
y_pred_df = pd.DataFrame(y_pred)
y_pred_df.columns = ['y_pred']
pred_data = pd.DataFrame(y_pred_df['y_pred'])
y_test_new = pd.DataFrame(y_test)
#y_test_new.reset_index(inplace=True)

pred_data['y_test'] = pd.DataFrame(y_test_new['quality'])

# R-square calculation
rsqd = r2_score(y_test_new['quality'].tolist(), y_pred_df['y_pred'].tolist())
print ("\nTest R-squared value:",round(rsqd,4))






# Ridge Regression
from sklearn.linear_model import Ridge

wine_quality = pd.read_csv("winequality-red.csv",sep=';')  
wine_quality.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)

all_colnms = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
 'pH', 'sulphates', 'alcohol']


pdx = wine_quality[all_colnms]
pdy = wine_quality["quality"]

x_train,x_test,y_train,y_test = train_test_split(pdx,pdy,train_size = 0.7,random_state=42)

alphas = [1e-4,1e-3,1e-2,0.1,0.5,1.0,5.0,10.0]

initrsq = 0

print ("\nRidge Regression: Best Parameters\n")
for alph in alphas:
    ridge_reg = Ridge(alpha=alph) 
    ridge_reg.fit(x_train,y_train)    
    tr_rsqrd = ridge_reg.score(x_train,y_train)
    ts_rsqrd = ridge_reg.score(x_test,y_test)    

    if ts_rsqrd > initrsq:
        print ("Lambda: ",alph,"Train R-Squared value:",round(tr_rsqrd,5),"Test R-squared value:",round(ts_rsqrd,5))
        initrsq = ts_rsqrd

# Coeffients of Ridge regression of best alpha value
ridge_reg = Ridge(alpha=0.001) 
ridge_reg.fit(x_train,y_train) 
 

print ("\nRidge Regression coefficient values of Alpha = 0.001\n")
for i in range(11):
    print (all_colnms[i],": ",ridge_reg.coef_[i])

# Lasso Regression
from sklearn.linear_model import Lasso

alphas = [1e-4,1e-3,1e-2,0.1,0.5,1.0,5.0,10.0]
initrsq = 0
print ("\nLasso Regression: Best Parameters\n")

for alph in alphas:
    lasso_reg = Lasso(alpha=alph) 
    lasso_reg.fit(x_train,y_train)    
    tr_rsqrd = lasso_reg.score(x_train,y_train)
    ts_rsqrd = lasso_reg.score(x_test,y_test)    

    if ts_rsqrd > initrsq:
        print ("Lambda: ",alph,"Train R-Squared value:",round(tr_rsqrd,5),"Test R-squared value:",round(ts_rsqrd,5))
        initrsq = ts_rsqrd

# Coeffients of Lasso regression of best alpha value
lasso_reg = Lasso(alpha=0.001) 
lasso_reg.fit(x_train,y_train) 

print ("\nLasso Regression coefficient values of Alpha = 0.001\n")
for i in range(11):
    print (all_colnms[i],": ",lasso_reg.coef_[i])

