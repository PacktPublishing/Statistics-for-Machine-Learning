

rm(list = ls())

# First change the following directory link to where all the input files do exist
setwd("D:\\Book writing\\Codes\\Chapter 7\\ml-latest-small\\ml-latest-small")

ratings = read.csv("ratings.csv")
movies = read.csv("movies.csv")

ratings = ratings[,!names(ratings) %in% c("timestamp")]

library(reshape2)

# Creating Pivot table
ratings_mat = acast(ratings,userId~movieId)
ratings_mat[is.na(ratings_mat)] =0


# Content based filtering
library(lsa)

a = c(2, 1, 0, 2, 0, 1, 1, 1)
b = c(2, 1, 1, 1, 1, 0, 1, 1)

print (paste("Cosine similarity between A and B is", round(cosine(a,b),4)))


m = nrow(ratings_mat);n = ncol(ratings_mat)


# User similarity matrix
mat_users = matrix(nrow = m, ncol = m)

for (i in 1:m){
  for (j in 1:m){
    if (i != j){
      mat_users[i,j] = cosine(ratings_mat[i,],ratings_mat[j,])
    }
    else {
      mat_users[i,j] = 0.0
    }
  }
}


colnames(mat_users) = rownames(ratings_mat); rownames(mat_users) = rownames(ratings_mat)
df_users = as.data.frame(mat_users)



# Finiding similar users
topn_simusers <- function(uid=16,n=5){
  sorted_df = sort(df_users[uid,],decreasing = TRUE)[1:n]
  print(paste("Similar users as user:",uid))
  return(sorted_df)
}

print(topn_simusers(uid = 17,n=10))


# Finiding most rated movies of a user
library(sqldf)

ratings_withmovie = sqldf(" select a.*,b.title from 
                          ratings as a left join movies as b 
                          on a.movieId = b.movieId")


# Finding most rated movies of a user
topn_movieratings <- function(uid=355,n_ratings=10){
  uid_ratings = ratings_withmovie[ratings_withmovie$userId==uid,]
  sorted_uidrtng = uid_ratings[order(-uid_ratings$rating),]
  return(head(sorted_uidrtng,n_ratings))
}
  
print( topn_movieratings(uid = 596,n=10))



# Movies similarity matrix
mat_movies = matrix(nrow = n, ncol = n)

for (i in 1:n){
  for (j in 1:n){
    if (i != j){
      mat_movies[i,j] = cosine(ratings_mat[,i],ratings_mat[,j])
    }
    else {
      mat_movies[i,j] = 0.0
    }
  }
}

colnames(mat_movies) = colnames(ratings_mat); rownames(mat_movies) = colnames(ratings_mat)
df_movies = as.data.frame(mat_movies)

write.csv(df_movies,"df_movies.csv")

df_movies = read.csv("df_movies.csv")
rownames(df_movies) = df_movies$X
colnames(df_movies) = c("aaa",df_movies$X)
df_movies = subset(df_movies, select=-c(aaa))



# Finiding similar movies
topn_simovies <- function(mid=588,n_movies=5){
  sorted_df = sort(df_movies[mid,],decreasing = TRUE)[1:n_movies]
  sorted_df_t = as.data.frame(t(sorted_df))
  colnames(sorted_df_t) = c("score")
  sorted_df_t$movieId = rownames(sorted_df_t)
  
  print(paste("Similar",n_movies, "movies as compared to the movie",mid,"are :"))
  sorted_df_t_wmovie = sqldf(" select a.*,b.title from sorted_df_t as a left join movies as b 
                          on a.movieId = b.movieId")
  return(sorted_df_t_wmovie)
}

print(topn_simovies(mid = 589,n_movies=15))




# Collaborative filtering
ratings = read.csv("ratings.csv")
movies = read.csv("movies.csv")

library(sqldf)
library(reshape2)
library(recommenderlab)

ratings_v2 = ratings[,-c(4)]

ratings_mat = acast(ratings_v2,userId~movieId)
ratings_mat2 =  as(ratings_mat, "realRatingMatrix")

getRatingMatrix(ratings_mat2)

#Plotting user-item complete matrix
image(ratings_mat2, main = "Raw Ratings")

# Fitting ALS method on Data
rec=Recommender(ratings_mat2[1:nrow(ratings_mat2)],method="UBCF", param=list(normalize = "Z-score",method="Cosine",nn=5, minRating=1))
rec_2=Recommender(ratings_mat2[1:nrow(ratings_mat2)],method="POPULAR")

print(rec)
print(rec_2)

names(getModel(rec))
getModel(rec)$nn


# Create predictions for all the users
recom_pred = predict(rec,ratings_mat2[1:nrow(ratings_mat2)],type="ratings")

# Putting predicitons into list
rec_list<-as(recom_pred,"list")
head(summary(rec_list))

print_recommendations <- function(uid=586,top_nmovies=10){
  recoms_list = rec_list[[uid]]
  sorted_df = as.data.frame(sort(recoms_list,decreasing = TRUE)[1:top_nmovies])
  colnames(sorted_df) = c("score")

  sorted_df$movieId = rownames(sorted_df)
  print(paste("Movies recommended for the user",uid,"are follows:"))
  sorted_df_t_wmovie = sqldf(" select a.*,b.title from sorted_df as a left join movies as b 
                             on a.movieId = b.movieId")
  
  return(sorted_df_t_wmovie)
}

print(print_recommendations(uid = 580,top_nmovies = 15))







