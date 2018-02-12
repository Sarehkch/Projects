
Developing different models to predict insurance claims.


##train data
train= read.csv("train.csv", header = TRUE )

model1 = lm(loss ~ ., data = train)
summary(model1)
###
### Get the test data and make the predictions!
###
test = read.csv("test.csv", header = TRUE)
y_hat = predict(model1, newdata = test)

Id = 1:nrow(test)
my_submission_1 = data.frame(Id = Id, y = y_hat)
###
### Quick check
###
head(my_submission_1)
dim(my_submission_1)
write.csv(my_submission_1 , file = "my_submission_1.csv" , row.names = FALSE)

### model2-multiple linear regression with variables selected by BIC
###
library(MASS)
model2= stepAIC(model1, direction = "backward")
summary(model2)
y_hat = predict(model2, newdata = test)
Id = 1:nrow(test)
my_submission_2 = data.frame(Id = Id, y = y_hat)
head(my_submission_2)
dim(my_submission_2)
write.csv(my_submission_2 , file = "my_submission_2.csv" , row.names = FALSE)

##Multiple regression with interactions
Model5 = stepAIC(lm(loss ~(.)^2, data = train), direction = "backward")
summary(model5) 
y_hat = predict(model5, newdata = test)

Id = 1:nrow(test)
my_submission_5 = data.frame(Id = Id, y = y_hat)
###
### Quick check
###

head(my_submission_5)
dim(my_submission_5)
write.csv(my_submission_5 , file = "my_submission_5.csv" , row.names = FALSE)

### Multiple regression with interaction terms
model6 = lm(loss ~ cat79 + cat101 + (cont12*cont6)+ (cont11*cont7)+cat57+cat12+(cont2*cont3)+(cont10*cont1)+(cont13*cont9)+cat114+cat9+cat10+cat72, data = train)
summary(model6) 

y_hat = predict(model6, newdata = test)

Id = 1:nrow(test)
my_submission_6 = data.frame(Id = Id, y = y_hat)
###
### Quick check
###
head(my_submission_6)
dim(my_submission_6)
write.csv(my_submission_6 , file = "my_submission_6.csv" , row.names = FALSE)

## Multiple regression with interactions
model7 = stepAIC(lm(loss ~ cat79 + cat101 + (cont12+cont6+cont11+cont7+cont2+cont3+cont10+cont1+cont13+cont9)^2+cat57+cat12+cat114+cat9+cat10+cat72, data = train), direction = "backward")
summary(model7) 

y_hat = predict(model7, newdata = test)

Id = 1:nrow(test)
my_submission_7 = data.frame(Id = Id, y = y_hat)
###
### Quick check
###

head(my_submission_7)
dim(my_submission_7)
write.csv(my_submission_7 , file = "my_submission_7.csv" , row.names = FALSE)


### model 3 – single tree
library(rpart)
final_tree = rpart(loss ~ ., data = train)
y_train_tree = predict(final_tree, train)

y_test_tree = predict(final_tree, test)
Id = 1:nrow(test)
my_submission_3 = data.frame(Id = Id, y = y_test_tree)
###
### Quick check
###
head(my_submission_3)
dim(my_submission_3)
write.csv(my_submission_3 , file = "my_submission_3.csv" , row.names = FALSE)
###


############ Random Forest
library(ranger)

mtry_vals = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15)
rf_error = numeric(length(mtry_vals))
set.seed(10000)
system.time({
  for (i in 1:length(mtry_vals))
  {
    rf_error[i] = ranger(loss ~ ., mtry = mtry_vals[i], data = train)$prediction.error
  }
})
plot(rf_error ~ mtry_vals, pch = 19, type = "b")


### mtry = 5

set.seed(1)
final_rf = ranger(loss ~ ., mtry = 3, data = train)
y_test_rf = predict(final_rf, test)$predictions
Id = 1:nrow(test)
my_submission_4 = data.frame(Id = Id, y = y_test_rf)
###
### Quick check
###
head(my_submission_4)
dim(my_submission_4)
write.csv(my_submission_4 , file = "my_submission_4.csv" , row.names = FALSE)

### gbm
gb_model=  gbm(loss ~ ., data = train  , distribution="gaussian", interaction.depth = 5,
               shrinkage =0.05 ,
               n.trees = 500,
               bag.fraction = 0.5)
gbm_predictions = predict(gb_model, test, n.trees=500, type = "response")
Id = 1:nrow(test)
my_submission_8 = data.frame(Id = Id, y = gbm_predictions)
head(my_submission_8)
dim(my_submission_8)
write.csv(my_submission_8 , file = "my_submission_8.csv" , row.names = FALSE)

################ GBM

compute_error = function(a,b)
{
  tmp = table(a,b)
  out = ( sum(tmp) - sum(diag(tmp)) ) / sum(tmp)
  out
}

#install.packages("gbm")
#install.packages("DAAG")


library(gbm)
###
### This is a small grid:
###
gbm_grid = expand.grid(interaction.depth = c(5, 7, 10),
                       n.trees = c(500, 750, 1000,1500),
                       shrinkage = c(0.05, 0.01),
                       bag.fraction = c(0.50, 0.75) )
m = dim(gbm_grid)[1]
gbm_error = rep(0, m)
###
### Keep number of folds small
###
no_of_folds = 2
set.seed(1)
index_values = sample(1:no_of_folds, size = dim(train)[1], replace = TRUE)
system.time({
  for (i in 1:m)
  {
    tmp_error = rep(0, no_of_folds)
    for (j in 1:no_of_folds)
    {
      index_out = which(index_values == j)
      left_out_data = train[ index_out, ]
      left_in_data = train[ -index_out, ]
      
      tmp_model = gbm( loss ~ ., data = left_in_data,
                       dist = "gaussian",
                       interaction.depth = gbm_grid$interaction.depth[i],
                       shrinkage = gbm_grid$shrinkage[i],
                       n.trees = gbm_grid$n.trees[i],
                       bag.fraction = gbm_grid$bag.fraction[i])
      
      tmp_pred = predict(tmp_model, newdata = left_out_data, type="response",
                         n.trees = gbm_grid$n.trees[i])


      tmp_error[j] = compute_error(tmp_pred, left_out_data$loss)
    }
    gbm_error[i] = mean(tmp_error)
  }
})

gbm_results = cbind(gbm_grid, gbm_error)
gbm_results[which.min(gbm_error),]


set.seed(100)
final_gbm = gbm( loss ~ ., data = train,
                 dist = "gaussian",
                 interaction.depth = 5,
                 shrinkage = 0.01,
                 n.trees = 1000,
                 bag.fraction = 0.50)

y_test_gbm = predict(final_gbm, newdata = test, type="response", n.trees = 750) 

Id = 1:nrow(test)
my_submission_9 = data.frame(Id = Id, y = y_test_gbm)
head(my_submission_9)
dim(my_submission_9)
write.csv(my_submission_9 , file = "my_submission_9.csv" , row.names = FALSE)
y_test_gbm = predict(final_gbm, newdata = test, type="response", n.trees = 1000) 

Id = 1:nrow(test)
my_submission_10 = data.frame(Id = Id, y = y_test_gbm)
head(my_submission_10)
dim(my_submission_10)
write.csv(my_submission_10 , file = "my_submission_10.csv" , row.names = FALSE)


##### another try on gbm
train= read.csv("C:/Users/Sareh/Google Drive/PhD/Term 6 - Fall 2017/Statistical Modeling 1/Final project/train.csv", header = TRUE )

test = read.csv("C:/Users/Sareh/Google Drive/PhD/Term 6 - Fall 2017/Statistical Modeling 1/Final project/test.csv", header = TRUE)


compute_error = function(a,b)
{
  tmp = table(a,b)
  out = ( sum(tmp) - sum(diag(tmp)) ) / sum(tmp)
  out
}

#install.packages("gbm")
#install.packages("DAAG")

library(gbm)
###
### This is a small grid:
###
gbm_grid = expand.grid(interaction.depth = c(1,2,3,4,5),
                       n.trees = c(850,900,925,950,975, 1000),
                       shrinkage = c(0.02, 0.01, 0.005),
                       bag.fraction = c(0.1, 0.50) )
m = dim(gbm_grid)[1]
gbm_error = rep(0, m)
###
### Keep number of folds small
###
no_of_folds = 2
set.seed(1)
index_values = sample(1:no_of_folds, size = dim(train)[1], replace = TRUE)
system.time({
  for (i in 1:m)
  {
    tmp_error = rep(0, no_of_folds)
    for (j in 1:no_of_folds)
    {
      index_out = which(index_values == j)
      left_out_data = train[ index_out, ]
      left_in_data = train[ -index_out, ]
      
      tmp_model = gbm( loss ~ ., data = left_in_data,
                       dist = "gaussian",
                       interaction.depth = gbm_grid$interaction.depth[i],
                       shrinkage = gbm_grid$shrinkage[i],
                       n.trees = gbm_grid$n.trees[i],
                       bag.fraction = gbm_grid$bag.fraction[i])
      
      tmp_pred = predict(tmp_model, newdata = left_out_data, type="response",
                         n.trees = gbm_grid$n.trees[i])
      
      
      tmp_error[j] = compute_error(tmp_pred, left_out_data$loss)
    }
    gbm_error[i] = mean(tmp_error)
  }
})

gbm_results = cbind(gbm_grid, gbm_error)
gbm_results[which.min(gbm_error),]


set.seed(100)
final_gbm = gbm( loss ~ ., data = train,
                 dist = "gaussian",
                 interaction.depth = 3,
                 shrinkage = 0.02,
                 n.trees = 950,
                 bag.fraction = 0.50)

y_test_gbm = predict(final_gbm, newdata = test, type="response", n.trees = 900) 

Id = 1:nrow(test)
my_submission_16 = data.frame(Id = Id, y = y_test_gbm)
head(my_submission_16)
dim(my_submission_16)
write.csv(my_submission_16 , file = "my_submission_16.csv" , row.names = FALSE)



## Gradient Boosting 
compute_error = function(a,b)
{
  tmp = table(a,b)
  out = ( sum(tmp) - sum(diag(tmp)) ) / sum(tmp)
  out
}

#install.packages("gbm")
#install.packages("DAAG")


library(gbm)
###
### This is a small grid:
###
gbm_grid = expand.grid(interaction.depth = c(3,4,5),
                       n.trees = c(925,950,975, 1000),
                       shrinkage = c(0.02, 0.01, 0.005),
                       bag.fraction = c(0.1, 0.50) )
m = dim(gbm_grid)[1]
gbm_error = rep(0, m)
###
### Keep number of folds small
###
no_of_folds = 5
set.seed(1)
index_values = sample(1:no_of_folds, size = dim(train)[1], replace = TRUE)
system.time({
  for (i in 1:m)
  {
    tmp_error = rep(0, no_of_folds)
    for (j in 1:no_of_folds)
    {
      index_out = which(index_values == j)
      left_out_data = train[ index_out, ]
      left_in_data = train[ -index_out, ]
      
      tmp_model = gbm( loss ~ ., data = left_in_data,
                       dist = "gaussian",
                       interaction.depth = gbm_grid$interaction.depth[i],
                       shrinkage = gbm_grid$shrinkage[i],
                       n.trees = gbm_grid$n.trees[i],
                       bag.fraction = gbm_grid$bag.fraction[i])
      
      tmp_pred = predict(tmp_model, newdata = left_out_data, type="response",
                         n.trees = gbm_grid$n.trees[i])
      
      
      tmp_error[j] = compute_error(tmp_pred, left_out_data$loss)
    }
    gbm_error[i] = mean(tmp_error)
  }
})

gbm_results = cbind(gbm_grid, gbm_error)
gbm_results[which.min(gbm_error),]

set.seed(100)
final_gbm = gbm( loss ~ ., data = train,
                 dist = "gaussian",
                 interaction.depth = 5,
                 shrinkage = 0.01,
                 n.trees = 975,
                 bag.fraction = 0.1)

y_test_gbm = predict(final_gbm, newdata = test, type="response", n.trees = 900) 

Id = 1:nrow(test)
my_submission_18 = data.frame(Id = Id, y = y_test_gbm)
head(my_submission_18)
dim(my_submission_18)
write.csv(my_submission_18 , file = "my_submission_18.csv" , row.names = FALSE)


y_test_gbm = predict(final_gbm, newdata = test, type="response", n.trees = 975) 

Id = 1:nrow(test)
my_submission_19 = data.frame(Id = Id, y = y_test_gbm)
head(my_submission_19)
dim(my_submission_19)
write.csv(my_submission_19 , file = "my_submission_19.csv" , row.names = FALSE)

## Neural Network

install.packages("NeuralNetTools")

library(nnet)
library(pROC)
library(DAAG)
library(NeuralNetTools)

?nnet
set.seed(10000)

nn_reg = nnet(loss ~ ., size = 18, data = train, linout = TRUE, maxit = 1000)
nn_reg

## look ar train rmse
sqrt(mean((train$loss - nn_reg$fitted.values)^2))

y_hat = predict(nn_reg, test)
Id = 1:nrow(test)
my_submission_11 = data.frame(Id = Id, y = y_hat)
head(my_submission_11)
dim(my_submission_11)
write.csv(my_submission_11 , file = "my_submission_11.csv" , row.names = FALSE)

### Your creation Parts
## PCA
## conver all factors to numeric
train= read.csv("C:/Users/Sareh/Google Drive/PhD/Term 6 - Fall 2017/Statistical Modeling 1/Final project/train.csv", header = TRUE )

train$cat79 = as.numeric(train$cat79)
train$cat101=as.numeric(train$cat101)
train$cat57=as.numeric(train$cat57)
train$cat12=as.numeric(train$cat12)
train$cat114=as.numeric(train$cat114)
train$cat10=as.numeric(train$cat10)
train$cat9=as.numeric(train$cat9)
train$cat72=as.numeric(train$cat72)

#### PCA 
### pca of train data
newtrain = train[,-19]
pca_result = prcomp(newtrain)
cumulative_variation = cumsum((pca_result$sdev^2)/sum(pca_result$sdev^2))

plot(cumulative_variation, xlab = "Number of Components", type = "l")
max(which(cumulative_variation < 0.75))
max(which(cumulative_variation < 0.80))
max(which(cumulative_variation < 0.90))
max(which(cumulative_variation < 0.95))

## lets keep 2
newtrain_pca = as.data.frame(pca_result$x[1:5000,1:2])

## PCA for test data
## conver all factors to numerci
test = read.csv("C:/Users/Sareh/Google Drive/PhD/Term 6 - Fall 2017/Statistical Modeling 1/Final project/test.csv", header = TRUE)

test$cat79 = as.numeric(test$cat79)
test$cat101=as.numeric(test$cat101)
test$cat57=as.numeric(test$cat57)
test$cat12=as.numeric(test$cat12)
test$cat114=as.numeric(test$cat114)
test$cat10=as.numeric(test$cat10)
test$cat9=as.numeric(test$cat9)
test$cat72=as.numeric(test$cat72)

pca_result = prcomp(test)
cumulative_variation = cumsum((pca_result$sdev^2)/sum(pca_result$sdev^2))

plot(cumulative_variation, xlab = "Number of Components", type = "l")
max(which(cumulative_variation < 0.75))
max(which(cumulative_variation < 0.80))
max(which(cumulative_variation < 0.90))
max(which(cumulative_variation < 0.95))

## lets keep 2
newtest_pca = as.data.frame(pca_result$x[1:4998,1:2])

###
newtrain_pca$loss = train$loss

## lets try linear regression on pca

model1 = lm(loss ~ ., data = newtrain_pca)
summary(model1)
###
### Get the test data and make the predictions!
###

y_hat = predict(model1, newdata = newtest_pca)

Id = 1:nrow(test)
my_submission_12 = data.frame(Id = Id, y = y_hat)
###
### Quick check
###
head(my_submission_12)
dim(my_submission_12)
write.csv(my_submission_12 , file = "my_submission_12.csv" , row.names = FALSE)

####################################################
#### gbm on pca
compute_error = function(a,b)
{
  tmp = table(a,b)
  out = ( sum(tmp) - sum(diag(tmp)) ) / sum(tmp)
  out
}

#install.packages("gbm")
#install.packages("DAAG")


library(gbm)
###
### This is a small grid:
###
gbm_grid = expand.grid(interaction.depth = c(5, 7, 10),
                       n.trees = c(500, 750, 1000,1500),
                       shrinkage = c(0.05, 0.01),
                       bag.fraction = c(0.50, 0.75) )
m = dim(gbm_grid)[1]
gbm_error = rep(0, m)
###
### Keep number of folds small
###
no_of_folds = 2
set.seed(1)
index_values = sample(1:no_of_folds, size = dim(newtrain_pca)[1], replace = TRUE)
system.time({
  for (i in 1:m)
  {
    tmp_error = rep(0, no_of_folds)
    for (j in 1:no_of_folds)
    {
      index_out = which(index_values == j)
      left_out_data = newtrain_pca[ index_out, ]
      left_in_data = newtrain_pca[ -index_out, ]
      
      tmp_model = gbm( loss ~ ., data = left_in_data,
                       dist = "gaussian",
                       interaction.depth = gbm_grid$interaction.depth[i],
                       shrinkage = gbm_grid$shrinkage[i],
                       n.trees = gbm_grid$n.trees[i],
                       bag.fraction = gbm_grid$bag.fraction[i])
      
      tmp_pred = predict(tmp_model, newdata = left_out_data, type="response",
                         n.trees = gbm_grid$n.trees[i])
      
      
      tmp_error[j] = compute_error(tmp_pred, left_out_data$loss)
    }
    gbm_error[i] = mean(tmp_error)
  }
})

gbm_results = cbind(gbm_grid, gbm_error)
gbm_results[which.min(gbm_error),]


set.seed(100)
final_gbm = gbm( loss ~ ., data = newtrain_pca,
                 dist = "gaussian",
                 interaction.depth = 7,
                 shrinkage = 0.05,
                 n.trees = 750,
                 bag.fraction = 0.50)

y_test_gbm = predict(final_gbm, newdata = newtest_pca, type="response", n.trees = 500) 

Id = 1:nrow(test)
my_submission_13 = data.frame(Id = Id, y = y_test_gbm)
head(my_submission_13)
dim(my_submission_13)
write.csv(my_submission_13 , file = "my_submission_13.csv" , row.names = FALSE)


## nnet on pca
library(nnet)
nn_reg = nnet(loss ~ ., size = 7, data = newtrain_pca, linout = TRUE, maxit = 1000)
nn_reg

## look ar train rmse
sqrt(mean((newtrain_pca$loss - nn_reg$fitted.values)^2))

y_hat = predict(nn_reg, newtest_pca)
Id = 1:nrow(newtest_pca)
my_submission_14 = data.frame(Id = Id, y = y_hat)
head(my_submission_14)
dim(my_submission_14)
write.csv(my_submission_14 , file = "my_submission_14.csv" , row.names = FALSE)

### SVM
library(e1071)
set.seed(100)
system.time({
  tuned.svm = tune(svm , loss ~., data = train ,
                   ranges = list( cost=c(0.001 , 1), gamma = c(0.001, 0.01)) )
})
tuned.svm
###
### Best was cost = 1, gamma = 0.01
###
final_svm = svm(loss ~ ., data = train, cost = 1, gamma = 0.01)
y_train_svm = predict(final_svm, train, type = "class")
svm_train_error = compute_error(train$loss, y_train_svm )
svm_train_error

y_test_svm = predict(final_svm, test, type = "class")
Id = 1:nrow(test)
my_submission_15 = data.frame(Id = Id, y = y_test_svm)
head(my_submission_15)
dim(my_submission_15)
write.csv(my_submission_15 , file = "my_submission_15.csv" , row.names = FALSE)
