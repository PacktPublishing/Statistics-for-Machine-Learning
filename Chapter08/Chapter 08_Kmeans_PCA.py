


import os
""" First change the following directory link to where all input files do exist """
os.chdir("D:\\Book writing\\Codes\\Chapter 8")


# K-means clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


iris = pd.read_csv("iris.csv")
print (iris.head())

x_iris = iris.drop(['class'],axis=1)
y_iris = iris["class"]


k_means_fit = KMeans(n_clusters=3,max_iter=300)
k_means_fit.fit(x_iris)

print ("\nK-Means Clustering - Confusion Matrix\n\n",pd.crosstab(y_iris,k_means_fit.labels_,rownames = ["Actuall"],colnames = ["Predicted"]) )     
print ("\nSilhouette-score: %0.3f" % silhouette_score(x_iris, k_means_fit.labels_, metric='euclidean'))

for k in range(2,10):
    k_means_fitk = KMeans(n_clusters=k,max_iter=300)
    k_means_fitk.fit(x_iris)
    print ("For K value",k,",Silhouette-score: %0.3f" % silhouette_score(x_iris, k_means_fitk.labels_, metric='euclidean'))
    

# Avg. within-cluster sum of squares
K = range(1,10)

KM = [KMeans(n_clusters=k).fit(x_iris) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(x_iris, centrds, 'euclidean') for centrds in centroids]

cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
avgWithinSS = [sum(d)/x_iris.shape[0] for d in dist]

# Total with-in sum of square
wcss = [sum(d**2) for d in dist]
tss = sum(pdist(x_iris)**2)/x_iris.shape[0]
bss = tss-wcss



# elbow curve - Avg. within-cluster sum of squares
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, avgWithinSS, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
#plt.title('Elbow for KMeans clustering')


# elbow curve - percentage of variance explained
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, bss/tss*100, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Percentage of variance explained')
#plt.title('Elbow for KMeans clustering')





# Calculation of eigenvectors & eigenvalues
import numpy as np
w,v = np.linalg.eig(np.array([[ 0.91335 ,0.75969 ],[ 0.75969,0.69702]]))
print ("\nEigen Values\n",w) 
print ("\nEigen Vectors\n",v)




# PCA - Principal Component Analysis
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits



digits = load_digits()
X = digits.data
y = digits.target


print (digits.data[0].reshape(8,8))

plt.matshow(digits.images[0]) 
plt.show() 


from sklearn.preprocessing import scale
X_scale = scale(X,axis=0)

pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X_scale)


zero_x, zero_y = [],[] ; one_x, one_y = [],[]
two_x,two_y = [],[]; three_x, three_y = [],[]
four_x,four_y = [],[]; five_x,five_y = [],[]
six_x,six_y = [],[]; seven_x,seven_y = [],[]
eight_x,eight_y = [],[]; nine_x,nine_y = [],[]


for i in range(len(reduced_X)):
    if y[i] == 0:
        zero_x.append(reduced_X[i][0])
        zero_y.append(reduced_X[i][1])
        
    elif y[i] == 1:
        one_x.append(reduced_X[i][0])
        one_y.append(reduced_X[i][1])

    elif y[i] == 2:
        two_x.append(reduced_X[i][0])
        two_y.append(reduced_X[i][1])

    elif y[i] == 3:
        three_x.append(reduced_X[i][0])
        three_y.append(reduced_X[i][1])

    elif y[i] == 4:
        four_x.append(reduced_X[i][0])
        four_y.append(reduced_X[i][1])

    elif y[i] == 5:
        five_x.append(reduced_X[i][0])
        five_y.append(reduced_X[i][1])

    elif y[i] == 6:
        six_x.append(reduced_X[i][0])
        six_y.append(reduced_X[i][1])

    elif y[i] == 7:
        seven_x.append(reduced_X[i][0])
        seven_y.append(reduced_X[i][1])

    elif y[i] == 8:
        eight_x.append(reduced_X[i][0])
        eight_y.append(reduced_X[i][1])
    
    elif y[i] == 9:
        nine_x.append(reduced_X[i][0])
        nine_y.append(reduced_X[i][1])



zero = plt.scatter(zero_x, zero_y, c='r', marker='x',label='zero')
one = plt.scatter(one_x, one_y, c='g', marker='+')
two = plt.scatter(two_x, two_y, c='b', marker='s')

three = plt.scatter(three_x, three_y, c='m', marker='*')
four = plt.scatter(four_x, four_y, c='c', marker='h')
five = plt.scatter(five_x, five_y, c='r', marker='D')

six = plt.scatter(six_x, six_y, c='y', marker='8')
seven = plt.scatter(seven_x, seven_y, c='k', marker='*')
eight = plt.scatter(eight_x, eight_y, c='r', marker='x')

nine = plt.scatter(nine_x, nine_y, c='b', marker='D')


plt.legend((zero,one,two,three,four,five,six,seven,eight,nine),
           ('zero','one','two','three','four','five','six','seven','eight','nine'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=10)

plt.xlabel('PC 1')
plt.ylabel('PC 2')

plt.show()




# 3-Dimensional data
pca_3d = PCA(n_components=3)
reduced_X3D = pca_3d.fit_transform(X_scale)

print (pca_3d.explained_variance_ratio_)



zero_x, zero_y,zero_z = [],[],[] ; one_x, one_y,one_z = [],[],[]
two_x,two_y,two_z = [],[],[]; three_x, three_y,three_z = [],[],[]
four_x,four_y,four_z = [],[],[]; five_x,five_y,five_z = [],[],[]
six_x,six_y,six_z = [],[],[]; seven_x,seven_y,seven_z = [],[],[]
eight_x,eight_y,eight_z = [],[],[]; nine_x,nine_y,nine_z = [],[],[]


for i in range(len(reduced_X3D)):
    
    if y[i]==10:
        continue
    
    elif y[i] == 0:
        zero_x.append(reduced_X3D[i][0])
        zero_y.append(reduced_X3D[i][1])
        zero_z.append(reduced_X3D[i][2])
        
    elif y[i] == 1:
        one_x.append(reduced_X3D[i][0])
        one_y.append(reduced_X3D[i][1])
        one_z.append(reduced_X3D[i][2])

    elif y[i] == 2:
        two_x.append(reduced_X3D[i][0])
        two_y.append(reduced_X3D[i][1])
        two_z.append(reduced_X3D[i][2])

    elif y[i] == 3:
        three_x.append(reduced_X3D[i][0])
        three_y.append(reduced_X3D[i][1])
        three_z.append(reduced_X3D[i][2])

    elif y[i] == 4:
        four_x.append(reduced_X3D[i][0])
        four_y.append(reduced_X3D[i][1])
        four_z.append(reduced_X3D[i][2])

    elif y[i] == 5:
        five_x.append(reduced_X3D[i][0])
        five_y.append(reduced_X3D[i][1])
        five_z.append(reduced_X3D[i][2])

    elif y[i] == 6:
        six_x.append(reduced_X3D[i][0])
        six_y.append(reduced_X3D[i][1])
        six_z.append(reduced_X3D[i][2])

    elif y[i] == 7:
        seven_x.append(reduced_X3D[i][0])
        seven_y.append(reduced_X3D[i][1])
        seven_z.append(reduced_X3D[i][2])

    elif y[i] == 8:
        eight_x.append(reduced_X3D[i][0])
        eight_y.append(reduced_X3D[i][1])
        eight_z.append(reduced_X3D[i][2])
    
    elif y[i] == 9:
        nine_x.append(reduced_X3D[i][0])
        nine_y.append(reduced_X3D[i][1])
        nine_z.append(reduced_X3D[i][2])


# 3- Dimensional plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(zero_x, zero_y,zero_z, c='r', marker='x',label='zero')
ax.scatter(one_x, one_y,one_z, c='g', marker='+',label='one')
ax.scatter(two_x, two_y,two_z, c='b', marker='s',label='two')

ax.scatter(three_x, three_y,three_z, c='m', marker='*',label='three')
ax.scatter(four_x, four_y,four_z, c='c', marker='h',label='four')
ax.scatter(five_x, five_y,five_z, c='r', marker='D',label='five')

ax.scatter(six_x, six_y,six_z, c='y', marker='8',label='six')
ax.scatter(seven_x, seven_y,seven_z, c='k', marker='*',label='seven')
ax.scatter(eight_x, eight_y,eight_z, c='r', marker='x',label='eight')

ax.scatter(nine_x, nine_y,nine_z, c='b', marker='D',label='nine')

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')

plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=10, bbox_to_anchor=(0, 0))

plt.show()




# Chosing number of Principal Components
max_pc = 30

pcs = []
totexp_var = []

for i in range(max_pc):
    pca = PCA(n_components=i+1)
    reduced_X = pca.fit_transform(X_scale)
    tot_var = pca.explained_variance_ratio_.sum()
    pcs.append(i+1)
    totexp_var.append(tot_var)

plt.plot(pcs,totexp_var,'r')
plt.plot(pcs,totexp_var,'bs')
plt.xlabel('No. of PCs',fontsize = 13)
plt.ylabel('Total variance explained',fontsize = 13)

plt.xticks(pcs,fontsize=13)
plt.yticks(fontsize=13)
plt.show()



# SVD
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target


from sklearn.utils.extmath import randomized_svd
U,Sigma,VT = randomized_svd(X,n_components=15,n_iter=300,random_state=42)

print ("\nShape of Original Matrix:",X.shape)
print ("\nShape of Left Singular vector:",U.shape)
print ("Shape of Singular value:",Sigma.shape)
print ("Shape of Right Singular vector",VT.shape)

import pandas as pd
VT_df = pd.DataFrame(VT)

n_comps = 15
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=n_comps, n_iter=300, random_state=42)
reduced_X = svd.fit_transform(X)

print("\nTotal Variance explained for %d singular features are %0.3f"%(n_comps,svd.explained_variance_ratio_.sum())) 


# Choosing number of Singular Values
max_singfeat = 30

singfeats = []
totexp_var = []

for i in range(max_singfeat):
    svd = TruncatedSVD(n_components=i+1, n_iter=300, random_state=42)
    reduced_X = svd.fit_transform(X)
    tot_var = svd.explained_variance_ratio_.sum()
    singfeats.append(i+1)
    totexp_var.append(tot_var)

plt.plot(singfeats,totexp_var,'r')
plt.plot(singfeats,totexp_var,'bs')
plt.xlabel('No. of Features',fontsize = 13)
plt.ylabel('Total variance explained',fontsize = 13)

#plt.xticks(singfeats,fontsize=13)
plt.yticks(fontsize=13)
plt.show()





# Deep Auto Encoders
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits


digits = load_digits()
X = digits.data
y = digits.target

print (X.shape)
print (y.shape)


x_vars_stdscle = StandardScaler().fit_transform(X)
print (x_vars_stdscle.shape)



from keras.layers import Input,Dense
from keras.models import Model

# 2-Dimensional Architecture

input_layer = Input(shape=(64,),name="input")

encoded = Dense(32, activation='relu',name="h1encode")(input_layer)
encoded = Dense(16, activation='relu',name="h2encode")(encoded)
encoded = Dense(2, activation='relu',name="h3latent_layer")(encoded)

decoded = Dense(16, activation='relu',name="h4decode")(encoded)
decoded = Dense(32, activation='relu',name="h5decode")(decoded)
decoded = Dense(64, activation='sigmoid',name="h6decode")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")


# Fitting Encoder-Decoder model
autoencoder.fit(x_vars_stdscle, x_vars_stdscle, epochs=100,batch_size=256,shuffle=True,validation_split= 0.2 )

# Extracting Encoder section of the Model for prediction of latent variables
encoder = Model(autoencoder.input,autoencoder.get_layer("h3latent_layer").output)

# Predicting latent variables with extracted Encoder model
reduced_X = encoder.predict(x_vars_stdscle)

print (reduced_X.shape)


zero_x, zero_y = [],[] ; one_x, one_y = [],[]
two_x,two_y = [],[]; three_x, three_y = [],[]
four_x,four_y = [],[]; five_x,five_y = [],[]
six_x,six_y = [],[]; seven_x,seven_y = [],[]
eight_x,eight_y = [],[]; nine_x,nine_y = [],[]




# For 2-Dimensional data
for i in range(len(reduced_X)):
    if y[i] == 0:
        zero_x.append(reduced_X[i][0])
        zero_y.append(reduced_X[i][1])
        
    elif y[i] == 1:
        one_x.append(reduced_X[i][0])
        one_y.append(reduced_X[i][1])

    elif y[i] == 2:
        two_x.append(reduced_X[i][0])
        two_y.append(reduced_X[i][1])

    elif y[i] == 3:
        three_x.append(reduced_X[i][0])
        three_y.append(reduced_X[i][1])

    elif y[i] == 4:
        four_x.append(reduced_X[i][0])
        four_y.append(reduced_X[i][1])

    elif y[i] == 5:
        five_x.append(reduced_X[i][0])
        five_y.append(reduced_X[i][1])

    elif y[i] == 6:
        six_x.append(reduced_X[i][0])
        six_y.append(reduced_X[i][1])

    elif y[i] == 7:
        seven_x.append(reduced_X[i][0])
        seven_y.append(reduced_X[i][1])

    elif y[i] == 8:
        eight_x.append(reduced_X[i][0])
        eight_y.append(reduced_X[i][1])
    
    elif y[i] == 9:
        nine_x.append(reduced_X[i][0])
        nine_y.append(reduced_X[i][1])




zero = plt.scatter(zero_x, zero_y, c='r', marker='x',label='zero')
one = plt.scatter(one_x, one_y, c='g', marker='+')
two = plt.scatter(two_x, two_y, c='b', marker='s')

three = plt.scatter(three_x, three_y, c='m', marker='*')
four = plt.scatter(four_x, four_y, c='c', marker='h')
five = plt.scatter(five_x, five_y, c='r', marker='D')

six = plt.scatter(six_x, six_y, c='y', marker='8')
seven = plt.scatter(seven_x, seven_y, c='k', marker='*')
eight = plt.scatter(eight_x, eight_y, c='r', marker='x')

nine = plt.scatter(nine_x, nine_y, c='b', marker='D')


plt.legend((zero,one,two,three,four,five,six,seven,eight,nine),
           ('zero','one','two','three','four','five','six','seven','eight','nine'),
           scatterpoints=1,
           loc='lower right',
           ncol=3,
           fontsize=10)

plt.xlabel('Latent Feature 1',fontsize = 13)
plt.ylabel('Latent Feature 2',fontsize = 13)

plt.show()




# 3-Dimensional architecture
input_layer = Input(shape=(64,),name="input")

encoded = Dense(32, activation='relu',name="h1encode")(input_layer)
encoded = Dense(16, activation='relu',name="h2encode")(encoded)
encoded = Dense(3, activation='relu',name="h3latent_layer")(encoded)

decoded = Dense(16, activation='relu',name="h4decode")(encoded)
decoded = Dense(32, activation='relu',name="h5decode")(decoded)
decoded = Dense(64, activation='sigmoid',name="h6decode")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Fitting Encoder-Decoder model
autoencoder.fit(x_vars_stdscle, x_vars_stdscle, epochs=100,batch_size=256,shuffle=True,validation_split= 0.2 )

# Extracting Encoder section of the Model for prediction of latent variables
encoder = Model(autoencoder.input,autoencoder.get_layer("h3latent_layer").output)

# Predicting latent variables with extracted Encoder model
reduced_X3D = encoder.predict(x_vars_stdscle)



zero_x, zero_y,zero_z = [],[],[] ; one_x, one_y,one_z = [],[],[]
two_x,two_y,two_z = [],[],[]; three_x, three_y,three_z = [],[],[]
four_x,four_y,four_z = [],[],[]; five_x,five_y,five_z = [],[],[]
six_x,six_y,six_z = [],[],[]; seven_x,seven_y,seven_z = [],[],[]
eight_x,eight_y,eight_z = [],[],[]; nine_x,nine_y,nine_z = [],[],[]



for i in range(len(reduced_X3D)):
    
    if y[i]==10:
        continue
    
    elif y[i] == 0:
        zero_x.append(reduced_X3D[i][0])
        zero_y.append(reduced_X3D[i][1])
        zero_z.append(reduced_X3D[i][2])
        
    elif y[i] == 1:
        one_x.append(reduced_X3D[i][0])
        one_y.append(reduced_X3D[i][1])
        one_z.append(reduced_X3D[i][2])

    elif y[i] == 2:
        two_x.append(reduced_X3D[i][0])
        two_y.append(reduced_X3D[i][1])
        two_z.append(reduced_X3D[i][2])

    elif y[i] == 3:
        three_x.append(reduced_X3D[i][0])
        three_y.append(reduced_X3D[i][1])
        three_z.append(reduced_X3D[i][2])

    elif y[i] == 4:
        four_x.append(reduced_X3D[i][0])
        four_y.append(reduced_X3D[i][1])
        four_z.append(reduced_X3D[i][2])

    elif y[i] == 5:
        five_x.append(reduced_X3D[i][0])
        five_y.append(reduced_X3D[i][1])
        five_z.append(reduced_X3D[i][2])

    elif y[i] == 6:
        six_x.append(reduced_X3D[i][0])
        six_y.append(reduced_X3D[i][1])
        six_z.append(reduced_X3D[i][2])

    elif y[i] == 7:
        seven_x.append(reduced_X3D[i][0])
        seven_y.append(reduced_X3D[i][1])
        seven_z.append(reduced_X3D[i][2])

    elif y[i] == 8:
        eight_x.append(reduced_X3D[i][0])
        eight_y.append(reduced_X3D[i][1])
        eight_z.append(reduced_X3D[i][2])
    
    elif y[i] == 9:
        nine_x.append(reduced_X3D[i][0])
        nine_y.append(reduced_X3D[i][1])
        nine_z.append(reduced_X3D[i][2])



# 3- Dimensional plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(zero_x, zero_y,zero_z, c='r', marker='x',label='zero')
ax.scatter(one_x, one_y,one_z, c='g', marker='+',label='one')
ax.scatter(two_x, two_y,two_z, c='b', marker='s',label='two')

ax.scatter(three_x, three_y,three_z, c='m', marker='*',label='three')
ax.scatter(four_x, four_y,four_z, c='c', marker='h',label='four')
ax.scatter(five_x, five_y,five_z, c='r', marker='D',label='five')

ax.scatter(six_x, six_y,six_z, c='y', marker='8',label='six')
ax.scatter(seven_x, seven_y,seven_z, c='k', marker='*',label='seven')
ax.scatter(eight_x, eight_y,eight_z, c='r', marker='x',label='eight')

ax.scatter(nine_x, nine_y,nine_z, c='b', marker='D',label='nine')

ax.set_xlabel('Latent Feature 1',fontsize = 13)
ax.set_ylabel('Latent Feature 2',fontsize = 13)
ax.set_zlabel('Latent Feature 3',fontsize = 13)

ax.set_xlim3d(0,60)

plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=10, bbox_to_anchor=(0, 0))

plt.show()



ax.set_xlim3d(left = 0,right = 30)
ax.set_ylim3d(left = 0,right = 30)
ax.set_zlim3d(left = 0,right = 30)




















