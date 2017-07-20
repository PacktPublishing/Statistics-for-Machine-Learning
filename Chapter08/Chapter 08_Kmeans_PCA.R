


rm(list = ls())

# First change the following directory link to where all the input files do exist
setwd("D:\\Book writing\\Codes\\Chapter 8")

iris_data = read.csv("iris.csv")
x_iris = iris_data[,!names(iris_data) %in% c("class")]
y_iris = iris_data$class

km_fit = kmeans(x_iris,centers = 3,iter.max = 300 )

print(paste("K-Means Clustering- Confusion matrix"))
table(y_iris,km_fit$cluster)

mat_avgss = matrix(nrow = 10, ncol = 2)

# Average within the cluster sum of square
print(paste("Avg. Within sum of squares"))
for (i in (1:10)){
  km_fit = kmeans(x_iris,centers = i,iter.max = 300 )
  mean_km = mean(km_fit$withinss)
  print(paste("K-Value",i,",Avg.within sum of squares",round(mean_km,2)))
  mat_avgss[i,1] = i
  mat_avgss[i,2] = mean_km
}

plot(mat_avgss[,1],mat_avgss[,2],type = 'o',xlab = "K_Value",ylab = "Avg. within sum of square")
title("Avg. within sum of squares vs. K-value")


mat_varexp = matrix(nrow = 10, ncol = 2)
# Percentage of Variance explained
print(paste("Percent. variance explained"))
for (i in (1:10)){
  km_fit = kmeans(x_iris,centers = i,iter.max = 300 )
  var_exp = km_fit$betweenss/km_fit$totss
  print(paste("K-Value",i,",Percent var explained",round(var_exp,4)))
  mat_varexp[i,1]=i
  mat_varexp[i,2]=var_exp
}

plot(mat_varexp[,1],mat_varexp[,2],type = 'o',xlab = "K_Value",ylab = "Percent Var explained")
title("Avg. within sum of squares vs. K-value")


# PCA
digits_data = read.csv("digitsdata.csv")

remove_cols = c("target")
x_data = digits_data[,!(names(digits_data) %in% remove_cols)]
y_data = digits_data[,c("target")]

# Normalizing the data
normalize <- function(x) {return((x - min(x)) / (max(x) - min(x)))}
data_norm <- as.data.frame(lapply(x_data, normalize))
data_norm <- replace(data_norm, is.na(data_norm), 0.0)


# Extracting Principal Components
pr_out =prcomp(data_norm)
pr_components_all = pr_out$x


# 2- Dimensional PCA
K_prcomps = 2

pr_components = pr_components_all[,1:K_prcomps]

pr_components_df = data.frame(pr_components)
pr_components_df = cbind(pr_components_df,digits_data$target)
names(pr_components_df)[K_prcomps+1] = "target"

out <- split( pr_components_df , f = pr_components_df$target )
zero_df = out$`0`;one_df = out$`1`;two_df = out$`2`; three_df = out$`3`; four_df = out$`4`
five_df = out$`5`;six_df = out$`6`;seven_df = out$`7`;eight_df = out$`8`;nine_df = out$`9`


library(ggplot2)
# Plotting 2-dimensional PCA
ggplot(pr_components_df, aes(x = PC1, y = PC2, color = factor(target,labels = c("zero","one","two",
        "three","four","five","six","seven","eight","nine")))) + 
        geom_point()+ggtitle("2-D PCA on Digits Data") +
        labs(color = "Digtis")



# 3- Dimensional PCA
# Plotting 3-dimensional PCA
K_prcomps = 3

pr_components = pr_components_all[,1:K_prcomps]
pr_components_df = data.frame(pr_components)
pr_components_df = cbind(pr_components_df,digits_data$target)
names(pr_components_df)[K_prcomps+1] = "target"

pr_components_df$target = as.factor(pr_components_df$target)

out <- split( pr_components_df , f = pr_components_df$target )
zero_df = out$`0`;one_df = out$`1`;two_df = out$`2`; three_df = out$`3`; four_df = out$`4`
five_df = out$`5`;six_df = out$`6`;seven_df = out$`7`;eight_df = out$`8`;nine_df = out$`9`

library(scatterplot3d)

colors <- c("darkred", "darkseagreen4", "deeppink4", "greenyellow", "orange"
            , "navyblue", "red", "tan3", "steelblue1", "slateblue")
colors <- colors[as.numeric(pr_components_df$target)]
s3d = scatterplot3d(pr_components_df[,1:3], pch = 16, color=colors,
              xlab = "PC1",ylab = "PC2",zlab = "PC3",col.grid="lightblue",main = "3-D PCA on Digits Data")
legend(s3d$xyz.convert(3.1, 0.1, -3.5), pch = 16, yjust=0,
       legend = levels(pr_components_df$target),col =colors,cex = 1.1,xjust = 0)
       
# Chosing number of Principal Components
pr_var =pr_out$sdev ^2
pr_totvar = pr_var/sum(pr_var)
plot(cumsum(pr_totvar), xlab="Principal Component", ylab ="Cumilative Prop. of Var.",
     ylim=c(0,1),type="b",main = "PCAs vs. Cum prop of Var Explained")



#SVD 
library(svd)

digits_data = read.csv("digitsdata.csv")

remove_cols = c("target")
x_data = digits_data[,!(names(digits_data) %in% remove_cols)]
y_data = digits_data[,c("target")]



sv2 <- svd(x_data,nu=15)

sv_check = sv2$d

# Computing the square of the singular values, which can be thought of as the vector of matrix energy
# in order to pick top singular values which preserve at least 80% of variance explained
energy <- sv2$d ^ 2
tot_varexp = data.frame(cumsum(energy) / sum(energy))

names(tot_varexp) = "cum_var_explained"
tot_varexp$K_value = 1:nrow(tot_varexp)

plot(tot_varexp[,2],tot_varexp[,1],type = 'o',xlab = "K_Value",ylab = "Prop. of Var Explained")
title("SVD - Prop. of Var explained with K-value")

