In this study, different classification methods such as LDA, QDA, SVM, … were applied to recognize handwritten digits.
While some of them such as SVM and Random forest did well, the others like single tree came up with high test errors. 
For some of methods, data needs to be manipulated. For example, for LDA and gradient boosting, I removed the columns with zero 
variation to be able to apply LDA. In addition, for QDA, I removed columns with small variation. 
In the other methods, it is possible to play with parameters in the classification function to get smaller error. 

rm(list=ls())

## read data
train= read.csv("train.csv", header = FALSE )

### randomely pick 1600 rows
set.seed(1)
traindata= train[sample(nrow(train), 1600), ] 
dim(traindata)


### v1 to V784(784 = 28 * 28) represent the pixels.  
### v785 is the actual digit.
### convert the last column to a factor variable.
traindata$V785 = as.factor(traindata$V785)


## test data
### randomely pick 400 rows
test= read.csv("C:/Users/Sareh/Google Drive/PhD/Term 6 - Fall 2018/Statistical Modeling 1/Assignment/HW6/test.csv", header = FALSE )
set.seed(1)
testdata= test[sample(nrow(test), 400), ]
testdata$V785 = as.factor(testdata$V785)

#############################################################################################

##get_error function

get_error = function(predicted, observed)
{
  x = table( predicted, observed)
  print(x)
}

#############################################################################################

### LDA-train data

library(MASS)

## remove columns with zero standard deviation 

i = apply(traindata[,-785],2,sd)

u = which(i ==0)
u
new_train_data = traindata[,-u]
n=ncol(new_train_data)

lda_model = lda(V785 ~ ., data = new_train_data)
lda_model
lda_predictions = predict(lda_model, data=new_train_data)$class

get_error(lda_predictions , traindata$V785)   #train error
agreement = lda_predictions==traindata$V785   #diagonal of confusion matrix
prop.table(table(agreement)) 

### LDA - test data

new_test_data = testdata[,-u]

lda_predictions_test = predict(lda_model, new_test_data)
table(lda_predictions_test$class , testdata$V785)    #confusion matrix
agreement = lda_predictions_test$class==testdata$V785
prop.table(table(agreement))

#############################################################################################

### qda- train data

i = apply(traindata[,-785],2,sd)

u = which(i <110 )
u
new_train_data = traindata[,-u]


qda_model = qda(V785 ~ ., data = new_train_data)
qda_model

qda_predictions = predict(qda_model)$class
get_error(qda_predictions , traindata$V785)
agreement = qda_predictions==traindata$V785
prop.table(table(agreement))

### qda- test data
new_test_data = testdata[,-u]

qda_predictions_test = predict(qda_model, new_test_data)
table(qda_predictions_test$class , testdata$V785)
agreement = qda_predictions_test$class==testdata$V785
prop.table(table(agreement))

#############################################################################################

## naive bayes - train data
library (e1071)
nb_model = naiveBayes(traindata$V785 ~ ., data = traindata)
nb_predictions = predict(nb_model, newdata = traindata)

get_error( nb_predictions , traindata$V785)
agreement = nb_predictions==traindata$V785
prop.table(table(agreement))


## naive bayes - test data

nb_predictions_test = predict(nb_model, newdata = testdata)

get_error( nb_predictions_test , testdata$V785)
agreement = nb_predictions_test==testdata$V785
prop.table(table(agreement))

#############################################################################################

### SVM - train data
library (e1071)
#svmfit =svm(V785 ~ ., data = traindata)
#svmfit = svm(V785 ~ ., data = traindata, kernel = 'sigmoid', gamma='0.1', type='C-classification' )
svmfit = svm(V785 ~ ., data = traindata, kernel = 'polynomial', degree = '2', type='C-classification' )
#svmfit = svm(V785 ~ ., data = traindata, kernel = 'polynomial', degree = '3', type='C-classification' )
#svmfit = svm(V785 ~ ., data = traindata, kernel = 'polynomial', degree = '4', type='C-classification' )
#svmfit = svm(V785 ~ ., data = traindata, kernel = 'polynomial', degree = '6', type='C-classification' )

svm_predictions = predict(svmfit, newdata = traindata)

get_error( svm_predictions , traindata$V785)   # train error
agreement = svm_predictions == traindata$V785  # diagnol of confusion matrix
prop.table(table(agreement))

## SVM - test data

svm_predictions_test = predict(svmfit, newdata = testdata)

get_error( svm_predictions_test , testdata$V785)   # test error
agreement = svm_predictions_test == testdata$V785  # diagnol of confusion matrix
prop.table(table(agreement))

###Single tree - train data
install.packages("tree")
library(tree)
tree_model = tree(V785 ~ ., data = traindata)
tree_model

plot(tree_model)
text(tree_model)

tree.pred=predict(tree_model, newdata = traindata, type="class")
table(tree.pred ,traindata$V785)
agreement = tree.pred==traindata$V785
prop.table(table(agreement))


#############################################################################################

## single - test data
tree.pred_test=predict(tree_model, newdata = testdata, type="class")
table(tree.pred_test ,testdata$V785)
agreement = tree.pred_test==testdata$V785
prop.table(table(agreement))


## different models of single tree
install.packages("DAAG")
library(DAAG)
install.packages("rpart")
library(rpart)

tree_model2 = rpart(V785 ~ ., data = traindata)
tree.pred=predict(tree_model2, newdata = traindata, type="class")
table(tree.pred ,traindata$V785)
agreement = tree.pred==traindata$V785
prop.table(table(agreement))

## changing method and rpart.control
## tree - model 3
tree_model3 = rpart(V785 ~ ., data = traindata, method =  "class", control = rpart.control( cp = 0.05 ))
tree.pred=predict(tree_model3, newdata = traindata, type="class")
table(tree.pred ,traindata$V785)
agreement = tree.pred==traindata$V785
prop.table(table(agreement))

tree.pred_test=predict(tree_model3, newdata = testdata, type="class")
table(tree.pred_test ,testdata$V785)
agreement = tree.pred_test==testdata$V785
prop.table(table(agreement))

## tree-model 4

tree_model4 = rpart(V785 ~ ., data = traindata, method =  "class", control = rpart.control( cp = 0.01 ))
tree.pred=predict(tree_model4, newdata = traindata, type="class")
table(tree.pred ,traindata$V785)
agreement = tree.pred==traindata$V785
prop.table(table(agreement))

tree.pred_test=predict(tree_model4, newdata = testdata, type="class")
table(tree.pred_test ,testdata$V785)
agreement = tree.pred_test==testdata$V785
prop.table(table(agreement))

##tree model 5

tree_model5 = rpart(V785 ~ ., data = traindata, method =  "class", control = rpart.control( cp = 0.1 ) )
tree.pred=predict(tree_model5, newdata = traindata, type="class")
table(tree.pred ,traindata$V785)
agreement = tree.pred==traindata$V785
prop.table(table(agreement))


## tree model 6

tree_model6 = rpart(V785 ~ ., data = traindata, method =  "class", control = rpart.control( cp = 1 ))
tree.pred=predict(tree_model6, newdata = traindata, type="class")
table(tree.pred ,traindata$V785)
agreement = tree.pred==traindata$V785
prop.table(table(agreement))

#############################################################################################

##random forest train data

install.packages("ranger")
library(ranger)

# chnage of importance
rf_model1 = ranger(V785 ~ ., data = traindata, importance = 'none')
pred_error = rf_model1$prediction.error
rf_model2 = ranger(V785 ~ ., data = traindata, importance = 'impurity')
pred_error = rf_model2$prediction.error
rf_model3 = ranger(V785 ~ ., data = traindata, importance = 'permutation')
pred_error = rf_model3$prediction.error


set.seed(1)

mtry_vals   = 1:6
pred_error = numeric(6)

for (i in mtry_vals)
{
  rf_model = ranger(V785 ~ ., data = traindata, importance = 'permutation', mtry = mtry_vals[i])
  pred_error[i] =  rf_model$prediction.error
}

plot(pred_error ~ mtry_vals, pch = 19, type = "b")

rf_model = ranger(V785 ~ ., data = traindata, importance = 'permutation', mtry = 5)
pred_error =  rf_model$prediction.error

## rf-model-test data
rf.pred_test=predict(rf_model, data = testdata)
table(rf.pred_test$predictions ,testdata$V785)
agreement = rf.pred_test$predictions==testdata$V785
prop.table(table(agreement))

#############################################################################################

## Gradient Boosting train data

library(gbm)
library(DAAG)

i = apply(traindata[,-785],2,sd)

u = which(i ==0)
u
new_train_data = traindata[,-u]

new_test_data = testdata[,-u]


## train error
gb_model=  gbm(V785 ~ ., new_train_data  , distribution="multinomial", n.trees=500)
gbm_predictions = apply(predict(gb_model, new_train_data, n.trees=gb_model$n.trees),1,which.max) - 1L
gbm_predictions
agreement = gbm_predictions == traindata$V785
prop.table(table(agreement))


## test error
test_gb_prediction = apply(predict(gb_model, new_test_data, n.trees=gb_model$n.trees),1,which.max) - 1L

agreement = test_gb_prediction == testdata$V785
prop.table(table(agreement))


##tuning



gbm_grid = expand.grid(interaction.depth = c(1, 3), 
                       n.trees = c(50, 75, 100,500), 
                       shrinkage = c(0.01, 0.1, 1), bag.fraction = c(0.5,1))

head(gbm_grid)
tail(gbm_grid)

m = dim(gbm_grid)[1]
m

gbm_auc  = rep(0, m)
gbm_auc


set.seed(1)

system.time({
  
  for (i in 1:m)
  {
    
      gb_model     = gbm(V785 ~ ., new_train_data, 
                           dist              = "multinomial",
                           interaction.depth = gbm_grid$interaction.depth[i], 
                           shrinkage         = gbm_grid$shrinkage[i], 
                           n.trees           = gbm_grid$n.trees[i],
                           bag.fraction      = gbm_grid$bag.fraction[i])     
      
      gb_pred     = apply(predict(gb_model, newdata = new_train_data, n.trees = gbm_grid$n.trees[i]),1,which.max) - 1L   
      
      agreement = gb_pred==traindata$V785
      prop.table(table(agreement))
      
      gbm_auc[i]   = (NROW(traindata)-sum(agreement))/NROW(traindata)      ### train error
      
    
  }
  
})

results     = cbind(gbm_grid, gbm_auc)

best_result = results[which.min(gbm_auc),]         ## min train error
best_result

## gb model based on best result - tran error
gb_model_final=  gbm(V785 ~ ., new_train_data  , distribution="multinomial", n.trees=100, interaction.depth = 3, shrinkage = 0.1, bag.fraction = 0.5)

gbm_pred = apply(predict(gb_model_final, new_train_data, n.trees=gb_model_final$n.trees),1,which.max) - 1L
gbm_pred
agreement = gbm_pred == traindata$V785
prop.table(table(agreement))

## test error
gbm_pred_test = apply(predict(gb_model_final, new_test_data, n.trees=gb_model_final$n.trees),1,which.max) - 1L
gbm_pred_test
agreement = gbm_pred_test == testdata$V785
prop.table(table(agreement))


#############################################################################################

## Your Creation
## first find Principle components
set.seed(101)
traindata_x=traindata[,-785]
Pca_train = prcomp (traindata_x, scale.=F, center = F)

# variance
pr_var = ( Pca_train$sdev )^2 
# % of variance
prop_varex = 100* (pr_var / sum( pr_var ))

# Plot
plot( prop_varex, xlab = "Principal Component", 
      ylab = "Proportion of Variance Explained")

## cumulative variance
plot(cumsum(Pca_train$sdev)/sum(Pca_train$sdev)*100,main='Cumulative proportion of variance explained')

Pca_train$x[,1:100]

## now find PCA by using SVD (train data)

set.seed(101)
svd_train = svd(traindata_x)
svd_train 

# Compare PCA scores with the SVD's U*Sigma
sigma <- matrix(0,784,784)
diag(sigma) <- svd_train$d       ##converting d vector to a matrix
theoreticalScores <- (svd_train$u %*% sigma)
theoreticalScores[,1:100]

##is the PCA scores the same as svd's theoreticalScores

all(round(Pca_train$x,3) == round(theoreticalScores,3))   

## use principle component (obtained from SVD) + SVM
dataSVD = data.frame(cbind(traindata$V785,theoreticalScores[,1:100]))
library (e1071)
svmfit_SVD = svm(dataSVD[,1] ~ ., data = dataSVD, kernel = 'polynomial', degree = '2', type='C-classification' )

svm_svd_predictions = predict(svmfit_SVD, newdata = dataSVD)

## get error function
get_error = function(predicted, observed)
{
  x = table( predicted, observed)
  print(x)
}

get_error( svm_svd_predictions , dataSVD[,1])   # train error
agreement = svm_svd_predictions ==  dataSVD[,1]  # diagnol of confusion matrix
prop.table(table(agreement))


## test error for svd+svm


## PCA of test data using SVD

set.seed(101)
testdata_x=testdata[,-785]
Pca_test = prcomp (testdata_x, scale.=F, center = F)

## plot of cumulative variance
plot(cumsum(Pca_test$sdev)/sum(Pca_test$sdev)*100,main='Cumulative proportion of variance explained')

Pca_train$x[,1:50]


## now find PCA by using SVD (test data)

set.seed(101)
svd_test = svd(testdata_x)
svd_test 

# Compare PCA scores with the SVD's U*Sigma
sigma <- matrix(0,400,400)
diag(sigma) <- svd_test$d       ##converting d vector to a matrix
theoreticalScores <- (svd_test$u %*% sigma)
theoreticalScores[,1:50]

##is the PCA scores the same as svd's theoreticalScores

all(round(Pca_test$x,3) == round(theoreticalScores,3))   

## test error

testdataSVD = data.frame(cbind(testdata$V785,theoreticalScores[,1:50]))

svmfit_SVD_test = svm(testdataSVD[,1] ~ ., data = testdataSVD, kernel = 'polynomial', degree = '2', type='C-classification' )

svm_svd_predictions = predict(svmfit_SVD_test, newdata = testdataSVD)

get_error( svm_svd_predictions , testdataSVD[,1])   # test error
agreement = svm_svd_predictions == testdataSVD[,1]  # diagnol of confusion matrix
prop.table(table(agreement))
