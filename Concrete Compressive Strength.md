# Projects
Exploring the different models that can be used to identify the relation between the compressive strength of concrete based on its constituents and time taken to cure it. 
Modeling the Concrete strength
## Libraries required

library(boot)
library(ggplot2)
library(leaps)
library(glmnet)

## Import Data

data = read.csv("Concrete_Data.csv")
names(data) = strtrim(names(data),13)

## Data Visualisation

install.packages("ggplot2")
ggplot(data, aes(Concrete.comp)) +
        geom_histogram()
ggplot(data, aes(x = Coarse.Aggreg, y =Concrete.comp )) +
        geom_point()
ggplot(data, aes(x = Fine.Aggregat, y =Concrete.comp )) +
        geom_point()
ggplot(data, aes(x = Age..day., y =Concrete.comp )) +
        geom_point()
ggplot(data, aes(x = Superplastici, y =Concrete.comp )) +
        geom_point()

## Split data to train and test

set.seed(202)

sample <- sample.int(n = nrow(data), size = floor(.50*nrow(data)), replace = F)
train <- data[sample, ]
test  <- data[-sample, ]

## Initialize values

models = 5
training.rmse <- rep(0,models)
cv.rmse <- rep(0,models)
testing.rmse <- rep(0,models)

## RMSE function

rmse = function(a,b) { sqrt(mean((a-b)^2)) }

## Create linear model with all predictors

linear = glm(Concrete.comp ~ ., data = train)
training.rmse[1] = sqrt(sum(residuals(linear)^2)/nrow(train))
cv.rmse[1] = sqrt(cv.glm(train,linear, K = 10)$delta[1])
testing.rmse[1] = rmse(test$Concrete.comp, predict(linear,newdata = test))

## Variable selection using library leaps

 linear_select = regsubsets(Concrete.comp ~ ., data = train, nbest = 1)
 plot(linear_select, scale="adjr2")
 plot(linear_select,scale="bic")

## Variable selection using step
 
null = glm(Concrete.comp ~ 1, data = train)
full = glm(Concrete.comp ~ ., data = train)
linear_select = step(null,scope = list(lower = null, upper = full), direction = "both", k = log(nrow(train)))

training.rmse[2] = sqrt(sum(residuals(linear_select)^2)/nrow(train))
cv.rmse[2] = sqrt(cv.glm(train,linear_select, K = 10)$delta[1])
testing.rmse[2] = rmse(test$Concrete.comp, predict(linear_select,newdata = test))

## Lasso regression

lasso_model = cv.glmnet(x = data.matrix(train[, 1:8]), y = train$Concrete.comp, alpha = 1)
training.rmse[3] = sqrt(lasso_model$cvm[lasso_model$lambda == lasso_model$lambda.min])
testing.rmse[3] = rmse(test$Concrete.comp , predict(lasso_model, newx = data.matrix(test[,1:8]), s = lasso_model$lambda.min))

## Ridge regression

ridge_model = cv.glmnet(x = data.matrix(train[, 1:8]), y = train$Concrete.comp, alpha = 0)
training.rmse[4] = sqrt(ridge_model$cvm[ridge_model$lambda == ridge_model$lambda.min])
testing.rmse[4] = rmse(test$Concrete.comp , predict(ridge_model, newx = data.matrix(test[,1:8]), s = ridge_model$lambda.min))


## Our Model

full_inter = glm(Concrete.comp ~ .*., data = train)
our_model = step(null,scope = list(lower = null, upper = full_inter), direction = "both", k = log(nrow(train)))
summary(our_model)

training.rmse[5] = sqrt(sum(residuals(our_model)^2)/nrow(train))
cv.rmse[5] = sqrt(cv.glm(train,our_model, K = 10)$delta[1])
testing.rmse[5] = rmse(test$Concrete.comp, predict(our_model,newdata = test))
