The MNIST dataset consists of handwritten digits from 0 to 9. Variations in the writing styles of these digits exploded 
the number of clusters that had the best purity scores. 
A program was written in R to identify the best cluster size while using different algorithms.

## Clear environment
rm(list = ls())

## Library
library(kernlab)
library(cluster)
library(mclust)
library(dbscan)

## Input the test and train data
nrows = 800
train.full <- read.csv("train.csv", header = F)
set.seed(62)
train <- train.full[sample(nrow(train.full),nrows,replace = FALSE),]

## Convert last column to factor variable and store it seperately
train$V785 = as.factor(train$V785)
y_train = train$V785
train$V785 = NULL
clusters = 10:25

## Calculate purity
find_purity = function(bigTable, noOfClusters)
{
  cluster_purity = numeric(noOfClusters)
  sumForPurity = 0
  for(i in 1:noOfClusters)
  {
    cluster_purity[i] = max(bigTable[i,])/sum(bigTable[i,])
    sumForPurity = sumForPurity + max(bigTable[i,])
  }
  purity = sumForPurity/nrows
  return(purity)
}

## K-means
purity_kmeans = numeric(length(clusters))
forElbow = numeric(length(clusters))
for(i in clusters)
{
  bigTable = matrix(0,nrow = i,ncol = 10)
  km_res = kmeans(train,  centers = i, nstart = 50, iter.max = 30)
  for(j in 1:i)
  {
    bigTable[j,] = t(as.matrix(summary(y_train[which(km_res$cluster == j)])))
  }
  forElbow[i-clusters[1]+1] = sum(km_res$withinss)
  purity_kmeans[i-clusters[1]+1] = find_purity(bigTable, i)
}
plot(clusters,forElbow,xlab="Number of Clusters",ylab="Sum of Squared Errors",main = "Elbow Plot")
lines(clusters,forElbow)
purity_kmeans
max(purity_kmeans)
clusters[which(purity_kmeans == max(purity_kmeans))]

## Kernel K-means
# RBFDOT
purity_kkmeans_rbf = numeric(length(clusters))
for(i in clusters)
{
  bigTable = matrix(0,nrow = i,ncol = 10)
  km_res = kkmeans(as.matrix(train),  centers = i, kernel = "rbfdot")
  for(j in 1:i)
  {
    bigTable[j,] = t(as.matrix(summary(y_train[which(km_res@.Data == j)])))
  }
  purity_kkmeans_rbf[i-clusters[1]+1] = find_purity(bigTable, i)
}

purity_kkmeans_rbf
max(purity_kkmeans_rbf)
clusters[which(purity_kkmeans_rbf == max(purity_kkmeans_rbf))]

# tanhdot
purity_kkmeans_tanh = numeric(length(clusters))
for(i in clusters)
{
  bigTable = matrix(0,nrow = i,ncol = 10)
  km_res = kkmeans(as.matrix(train),  centers = i, kernel = "tanhdot")
  for(j in 1:i)
  {
    bigTable[j,] = t(as.matrix(summary(y_train[which(km_res@.Data == j)])))
  }
  purity_kkmeans_tanh[i-clusters[1]+1] = find_purity(bigTable, i)
}

purity_kkmeans_tanh
max(purity_kkmeans_tanh)
clusters[which(purity_kkmeans_tanh == max(purity_kkmeans_tanh))]

# besselDot
purity_kkmeans_bessel = numeric(length(clusters))
for(i in clusters)
{
  bigTable = matrix(0,nrow = i,ncol = 10)
  km_res = kkmeans(as.matrix(train),  centers = i, kernel = "besseldot")
  for(j in 1:i)
  {
    bigTable[j,] = t(as.matrix(summary(y_train[which(km_res@.Data == j)])))
  }
  purity_kkmeans_bessel[i-clusters[1]+1] = find_purity(bigTable, i)
}

purity_kkmeans_bessel
max(purity_kkmeans_bessel)
clusters[which(purity_kkmeans_bessel == max(purity_kkmeans_bessel))]

## Hierarchial Clustering
distance = dist(train)
# Complete Linkage
purity_hclust_comp = numeric(length(clusters))
hc_complete = hclust(distance, method = "complete")
for(i in clusters){
  bigTable = matrix(0,nrow = i,ncol = 10)
  clusterIDs = cutree(hc_complete, i)
  for(j in 1:i) {
    bigTable[j,] = t(as.matrix(summary(y_train[which(clusterIDs == j)])))
  }
  purity_hclust_comp[i-clusters[1]+1] = find_purity(bigTable, i)
}

purity_hclust_comp
max(purity_hclust_comp)
clusters[which(purity_hclust_comp == max(purity_hclust_comp))]

# Average Linkage
purity_hclust_avg = numeric(length(clusters))
hc_avg = hclust(distance, method = "average")
for(i in clusters){
  bigTable = matrix(0,nrow = i,ncol = 10)
  clusterIDs = cutree(hc_avg, i)
  for(j in 1:i) {
    bigTable[j,] = t(as.matrix(summary(y_train[which(clusterIDs == j)])))
  }
  purity_hclust_avg[i-clusters[1]+1] = find_purity(bigTable, i)
}

purity_hclust_avg
max(purity_hclust_avg)
clusters[which(purity_hclust_avg == max(purity_hclust_avg))]

# Single Linkage
purity_hclust_single = numeric(length(clusters))
hc_single = hclust(distance, method = "single")
for(i in clusters){
  bigTable = matrix(0,nrow = i,ncol = 10)
  clusterIDs = cutree(hc_single, i)
  for(j in 1:i) {
    bigTable[j,] = t(as.matrix(summary(y_train[which(clusterIDs == j)])))
  }
  purity_hclust_single[i-clusters[1]+1] = find_purity(bigTable, i)
}

purity_hclust_single
max(purity_hclust_single)
clusters[which(purity_hclust_single == max(purity_hclust_single))]

## EM Clustering
purity_EM = numeric(length(clusters))

em_res = densityMclust(train)
bigTable = matrix(0,nrow = em_res$G,ncol = 10)
for(j in 1:em_res$G)
{
  bigTable[j,] = t(as.matrix(summary(y_train[which(em_res$classification == j)])))
}
purity_EM = find_purity(bigTable, em_res$G)

purity_EM

## DBScan
kNNdistplot(train,k = 100)
abline(h = 1630, col = "red", lty=2)
purity_DB = numeric(length(clusters))
db_out = dbscan(train,eps = 1630)
for(j in 1:max(db_out$cluster))
{
  bigTable[j,] = t(as.matrix(summary(y_train[which(db_out$cluster == j)])))
}
purity_DB = find_purity(bigTable, max(db_out$cluster))
purity_DB

## Function to show image
show_an_image = function(n, x)
{
  v  = as.numeric(x[n,])
  im = matrix(v,28,28)
  im = im[,nrow(im):1]
  image(im, col = gray((0:255)/255))
}

## Show the cluster images
bigTable = matrix(0,nrow = 15,ncol = 10)
km_res = kmeans(train,  centers = 15, nstart = 50, iter.max = 30)
for(j in 1:15)
{
  bigTable[j,] = t(as.matrix(summary(y_train[which(km_res$cluster == j)])))
}
par(mfrow = c(5,5))
for(i in 1:25)
{
  show_an_image(i,train[which(km_res$cluster == j),])
}
