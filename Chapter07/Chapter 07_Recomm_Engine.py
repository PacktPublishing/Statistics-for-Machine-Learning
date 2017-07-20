

import os
""" First change the following directory link to where all input files do exist """
os.chdir("D:\\Book writing\\Codes\\Chapter 7\\ml-latest-small\\ml-latest-small")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ratings = pd.read_csv("ratings.csv")
print (ratings.head())

movies = pd.read_csv("movies.csv")
print (movies.head())


#Combining movie ratings & movie names
ratings = pd.merge(ratings[['userId','movieId','rating']],movies[['movieId','title']],
                   how='left',left_on ='movieId' ,right_on = 'movieId')


rp = ratings.pivot_table(columns = ['movieId'],index = ['userId'],values = 'rating')
rp = rp.fillna(0)

# Converting pandas dataframe to numpy for faster execution in loops etc.
rp_mat = rp.as_matrix()


from scipy.spatial.distance import cosine


#The cosine of the angle between them is about 0.822.
a= np.asarray( [2, 1, 0, 2, 0, 1, 1, 1])
b = np.asarray( [2, 1, 1, 1, 1, 0, 1, 1])

print("\n\n")
print ("Cosine similarity between A and B is",round(1-cosine(a,b),4))


m, n = rp.shape

# User similarity matrix
mat_users = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        if i != j:
            mat_users[i][j] = (1- cosine(rp_mat[i,:], rp_mat[j,:]))
        else:
            mat_users[i][j] = 0.
            
pd_users = pd.DataFrame(mat_users,index =rp.index ,columns= rp.index )


# Finding similar users
def topn_simusers(uid = 16,n=5):
    users = pd_users.loc[uid,:].sort_values(ascending = False)
    topn_users = users.iloc[:n,]
    topn_users = topn_users.rename('score')    
    print ("Similar users as user:",uid)
    return pd.DataFrame(topn_users)

print (topn_simusers(uid=17,n=10))   


# Finding most rated movies of a user
def topn_movieratings(uid = 355,n_ratings=10):    
    uid_ratings = ratings.loc[ratings['userId']==uid]
    uid_ratings = uid_ratings.sort_values(by='rating',ascending = [False])
    print ("Top",n_ratings ,"movie ratings of user:",uid)
    return uid_ratings.iloc[:n_ratings,]    

print (topn_movieratings(uid=596,n_ratings=10))


# Movie similarity matrix
import time
start_time = time.time()
mat_movies = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i!=j:
            mat_movies[i,j] = (1- cosine(rp_mat[:,i], rp_mat[:,j]))
        else:
            mat_movies[i,j] = 0.
print("--- %s seconds ---" % (time.time() - start_time))


pd_movies = pd.DataFrame(mat_movies,index =rp.columns ,columns= rp.columns )


#pd_movies.to_csv('pd_movies.csv',sep=',')
pd_movies = pd.read_csv("pd_movies.csv",index_col='movieId')


# Finding similar movies
def topn_simovies(mid = 588,n=15):
    mid_ratings = pd_movies.loc[mid,:].sort_values(ascending = False)
    topn_movies = pd.DataFrame(mid_ratings.iloc[:n,])
    topn_movies['index1'] = topn_movies.index
    topn_movies['index1'] = topn_movies['index1'].astype('int64')
    topn_movies = pd.merge(topn_movies,movies[['movieId','title']],how = 'left',left_on ='index1' ,right_on = 'movieId')
    print ("Movies similar to movie id:",mid,",",movies['title'][movies['movieId']==mid].to_string(index=False),",are")
    del topn_movies['index1']
    return topn_movies


print (topn_simovies(mid=589,n=15))




#Collaborative filtering

import os
""" First change the following directory link to where all input files do exist """
os.chdir("D:\\Book writing\\Codes\\Chapter 7\\ml-latest-small\\ml-latest-small")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ratings = pd.read_csv("ratings.csv")
print (ratings.head())

movies = pd.read_csv("movies.csv")
print (movies.head())

rp = ratings.pivot_table(columns = ['movieId'],index = ['userId'],values = 'rating')
rp = rp.fillna(0)

A = rp.values

print ("\nShape of Original Sparse Matrix",A.shape)


W = A>0.5
W[W==True]=1
W[W==False]=0
W = W.astype(np.float64,copy=False)


W_pred = A<0.5
W_pred[W_pred==True]=1
W_pred[W_pred==False]=0
W_pred = W_pred.astype(np.float64,copy=False)
np.fill_diagonal(W_pred,val=0)



# Parameters
m,n = A.shape

n_iterations = 200
n_factors = 100
lmbda = 0.1

X = 5 * np.random.rand(m,n_factors)
Y = 5* np.random.rand(n_factors,n)

def get_error(A, X, Y, W):
    return np.sqrt(np.sum((W * (A - np.dot(X, Y)))**2)/np.sum(W))

errors = []
for itr in range(n_iterations):
    X = np.linalg.solve(np.dot(Y,Y.T)+ lmbda * np.eye(n_factors),np.dot(Y,A.T)).T
    Y = np.linalg.solve(np.dot(X.T,X)+ lmbda * np.eye(n_factors),np.dot(X.T,A))
    
    if itr%10 == 0:
        print(itr," iterations completed","RMSError value is:",get_error(A,X,Y,W))
                 
    errors.append(get_error(A,X,Y,W))

A_hat = np.dot(X,Y)
print ("RMSError of rated movies: ",get_error(A,X,Y,W))
    

plt.plot(errors);
plt.ylim([0, 3.5]);
plt.xlabel("Number of Iterations");plt.ylabel("RMSE")
#plt.title("No.of Iterations vs. RMSE")
plt.show()



def print_recommovies(uid=315,n_movies=15,pred_mat = A_hat,wpred_mat = W_pred ):
    pred_recos = pred_mat*wpred_mat
    pd_predrecos = pd.DataFrame(pred_recos,index =rp.index ,columns= rp.columns )
    pred_ratings = pd_predrecos.loc[uid,:].sort_values(ascending = False)
    pred_topratings = pred_ratings[:n_movies,]
    pred_topratings = pred_topratings.rename('pred_ratings')  
    pred_topratings = pd.DataFrame(pred_topratings)
    pred_topratings['index1'] = pred_topratings.index
    pred_topratings['index1'] = pred_topratings['index1'].astype('int64')
    pred_topratings = pd.merge(pred_topratings,movies[['movieId','title']],how = 'left',left_on ='index1' ,right_on = 'movieId')
    del pred_topratings['index1']    
    print ("\nTop",n_movies,"movies predicted for the user:",uid," based on collaborative filtering\n")
    return pred_topratings


predmtrx = print_recommovies(uid=355,n_movies=10,pred_mat=A_hat,wpred_mat=W_pred)
print (predmtrx)




# Grid Search on Collaborative Filtering
def get_error(A, X, Y, W):
    return np.sqrt(np.sum((W * (A - np.dot(X, Y)))**2)/np.sum(W))

niters = [20,50,100,200]
factors = [30,50,70,100]
lambdas = [0.001,0.01,0.05,0.1]

init_error = float("inf")


print("\n\nGrid Search results of ALS Matrix Factorization:\n")
for niter in niters:
    for facts in factors:
        for lmbd in lambdas:
                
            X = 5 * np.random.rand(m,facts)
            Y = 5* np.random.rand(facts,n)
            
            for itr in range(niter):
                X = np.linalg.solve(np.dot(Y,Y.T)+ lmbd * np.eye(facts),np.dot(Y,A.T)).T
                Y = np.linalg.solve(np.dot(X.T,X)+ lmbd * np.eye(facts),np.dot(X.T,A))
            
            error = get_error(A,X,Y,W)
            
            if error<init_error:
                print ("No.of iters",niter,"No.of Factors",facts,"Lambda",lmbd,"RMSE",error)
                init_error = error
                
            
            












