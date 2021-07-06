if (!require(tidyverse)) install.packages('tidyverse')     # Data manipulation
if (!require(caret)) install.packages('caret')             # Data split
if (!require(Metrics)) install.packages('Metrics')         # Evaluate results
if (!require(rstudioapi)) install.packages('rstudioapi')   # for rstudio working dir
if (!require(Rtsne)) install.packages('Rtsne')             # for tsne plotting
if (!require(smotefamily)) install.packages('smotefamily') # for smote implementation
if (!require(rpart)) install.packages('rpart')             # for decision tree model
if (!require(Rborist)) install.packages('Rborist')         # for random forest model
if (!require(corrplot)) install.packages('corrplot')       # for data visualization
if (!require(ROSE)) install.packages('ROSE')               # for ROC curves
if (!require(xgboost)) install.packages('xgboost')         # for xgboost curves
if (!require(knitr)) install.packages("knitr")             # Latex tables
if (!require(kableExtra)) install.packages('kableExtra')   # for tables


# -----------------------------------------------
# Unzipping data located in /data/creditCard.zip
# -----------------------------------------------
# Setting working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
if (!file.exists("data/creditcard.csv")){
  unzip("data/creditCard.zip", exdir = "data")
}

if(!exists("fraud_data")){
  # Load into memory
  fraud_data <- read.csv(file = 'data/creditcard.csv')
}


# -------------------------------------------------
# Data verification, check na, empty and proportion
# -------------------------------------------------

# Take a look to our data
glimpse(fraud_data)

# Check na in the data set
no_na <- sum(is.na(fraud_data))

# Check amount col if has negative values
no_negative <- sum(fraud_data$Amount < 0)

# Check time variable
minTime <- min(fraud_data$Time)
maxTime <- max(fraud_data$Time)


# Check Class type
class(fraud_data$Class)  # Is integer
fraud_data$Class <- as.factor(fraud_data$Class)
class(fraud_data$Class)  # Is factor

# Check data balance
fraud_data %>% group_by(Class) %>% count() # Highly unbalanced 492 frauds vs 284315 no-fraud

# -------------------------------------------------
# Data Visualization,
# -------------------------------------------------

# Visualizing features
sample <- fraud_data[1:10000,]

# Checking all but time values
ggplot(stack(fraud_data[,-c(1,30, 31)]), aes(ind, values)) + 
  geom_boxplot(color = "black", fill = I("deepskyblue4")) +
  labs(
    x = "Feature",
    y = "Value")

# Checking time values
fraud_data %>%
  ggplot(aes(Time)) +
  geom_histogram(bins = 100, color = "black", fill = I("deepskyblue4")) + 
  labs(
     x = "Time (Seconds since the first transaction in the dataset)",
     y = "Count with that time")

# Visualization of time values
fraud_data %>%
  ggplot(aes(x = Time, fill = factor(Class))) + geom_histogram(bins = 100)+
  labs(x = 'Time in seconds since first transaction', y = 'Number of transactions') +
  ggtitle('Distribution of time of transaction by class') +
  facet_grid(Class ~ ., scales = 'free_y')

# Checking Amount values
fraud_data %>%
  ggplot(aes(Amount)) +
  geom_histogram(binwidth = 10, color = "black", fill = I("deepskyblue4")) + 
  labs(
    x = "Amount (Dollars for trasaction)",
    y = "Count") +
  xlim(-101,1000)

# Visualization of Amount values
fraud_data %>%
  ggplot(aes(x = Amount, fill = factor(Class))) + geom_histogram(binwidth = 5)+
  labs(x = 'Amount transaction', y = 'Number of transactions') +
  ggtitle('Distribution of transaction amount by class') +
  facet_grid(Class ~ ., scales = 'free_y') + 
  xlim(-101,500)


# Correlations plot
correlations <- cor(fraud_data[,-which(names(fraud_data) %in% c("Time", "Class"))], method="pearson")
corrplot(correlations, number.cex = .9, method = "circle", type = "full", tl.cex=0.8,tl.col = "black")

# T-SNE plot
tsne_subset <- 1:as.integer(0.1*nrow(fraud_data))
tsne <- Rtsne(fraud_data[tsne_subset,-c(1, 31)], perplexity = 20, theta = 0.5, pca = F, verbose = T, max_iter = 500, check_duplicates = F)

classes <- if_else(fraud_data$Class[tsne_subset] == 0, "No fraud", "Fraud")
tsne_mat <- as.data.frame(tsne$Y)
ggplot(tsne_mat, aes(x = V1, y = V2)) + 
  geom_point(aes(color = classes)) + 
  theme_minimal() + 
  ggtitle("t-SNE visualisation of transactions") + 
  scale_color_manual(values = c("#CA160A", "#439DEC"))


# -------------------------------------------------
# Data Partition
# -------------------------------------------------

# A we have a very unbalance dataset the procedure will consist in split data into train and test, to have a 80 to 20
# relation between train and test data.

# Training data will be upsampled in the fraud cases in order to have a balance training dataset. Test data will 
# remain untouch
# The upsample method will be SMOTE (synthetic minority oversampling technique)

# Getting our test samples
set.seed(2021)
index <- createDataPartition(y = fraud_data$Class, p = .8, list = FALSE)

train_data <- fraud_data[index,]
test_data <- fraud_data[-index,]


# Applying SMOTE
train <- SMOTE(train_data[,-which(names(train_data) %in% c("Class"))],
              train_data[,which(names(train_data) %in% c("Class"))])
train <- train$data
train %>% group_by(class) %>% count()

# Now train data has around 227,452 cases for no fraud and 227,338 for fraud

# --------------------------------------------------------------
#                           EVALUATION
# --------------------------------------------------------------

# As we are working with highly unbalanced we need a method to properly measure our result,
# for this particular case using accuracy is not a good metric as its formula (TP + TN / Total)
# will give a good metric even if we do not make a prediction at all (we will see it on the
# dummy approach section), therefore we will diferent metrics to evaluate our models, these models
# will be:
# F1 score - (2 * Precision * Recall) / (Precision + Recall)
# RO - Area Under the Curve, As a relationship between TRUE Positives rate and FALSE Positive rates
# Just as a reminder, 
# precission = TruePositive / (TruePositive + FalsePositive)
# recall = TruePositive / (TruePositive + FalseNegative)
# These 2 metrics will allow us to evaluate our method if is better or worse than other

# ------ Dummy Attemps

# Our first dummy attemp will consist in 3 types off prediction
# 1.- Random: Choose randomly between 1 and cero to predict our class
# 2.- No-fraud: Choose always 0, do not make any prediction at all, as is very small all fraud 
# cases we can say is never fraud
# 3.- Dumy statistic: Based on the probability between fraud and no fraud, chose one of them based
# on their ocurrance i.e. i if relationship is 10 to 1 predict 10 to 1 ocurrance of the class

# ------------ 1. Random
size <- length(test_data$Class)
y_hat_random <- sample(c(0:1), size = size, replace = TRUE)
random_acc <-  accuracy(test_data$Class, y_hat_random)
random_roc <- ROSE::roc.curve(test_data$Class, y_hat_random, plotit = TRUE)
random_roc_val <- random_roc$auc
random_f1 <- F_meas(data = as.factor(y_hat_random), reference = test_data$Class)

# ----------- 2. No-Fraud
y_hat_noFraud <- as.integer(replicate(0, n = size))
noFraud_acc <-  accuracy(test_data$Class, y_hat_noFraud)
noFraud_roc <- ROSE::roc.curve(test_data$Class, y_hat_noFraud, plotit = TRUE)
noFraud_roc_val <- noFraud_roc$auc

y_hat_noFraud[1] <- 1
noFraud_f1 <- F_meas(data = as.factor(y_hat_noFraud), reference = test_data$Class)


# ----------- 3. Dumy Statistics
counts <- train_data %>% group_by(Class) %>% count()
noFraudC <- counts$n[1]
fraudC <- counts$n[2]
relationship <- fraudC / (fraudC + noFraudC)
y_hat_statistic <- sample(c(0,1), size = size, replace = TRUE, prob = c(1-relationship,relationship))

statistic_acc <-  accuracy(test_data$Class, y_hat_statistic)
statistic_roc <- ROSE::roc.curve(test_data$Class, y_hat_statistic, plotit = TRUE)
statistic_roc_val <- statistic_roc$auc
statistic_f1 <- F_meas(data = as.factor(y_hat_statistic), reference = test_data$Class)

# Based on these preliminary metrics we can see that using ROC AUC is much better metric than using
# F1 or accuracy, we will use this one so on

# We will try several aproaches in order to determine the best method, this methods will be
# 1.- CART, one decision Tree, no upsampling
# 2.- CART, one decision Tree, upsampling
# 3.- Random Forest
# 4.- Xboost

# ------ 1.- CART
CART_noSm_fit <- rpart(Class ~ ., data = train_data)

#Evaluate model performance on test set
CART_noSm_pred <- predict(CART_noSm_fit, newdata = test_data, method = "Class")

CART_noSM_roc <- ROSE::roc.curve(test_data$Class, CART_noSm_pred[,2], plotit = TRUE)
CART_noSM_roc_val <- CART_noSM_roc$auc


# ------ 2.- CART with balance output
CART_Sm_fit <- rpart(class ~ ., data = train)

#Evaluate model performance on test set
CART_Sm_pred <- predict(CART_Sm_fit, newdata = test_data, method = "class")

CART_SM_roc <- ROSE::roc.curve(test_data$Class, CART_Sm_pred[,2], plotit = TRUE)
CART_SM_roc_val <- CART_SM_roc$auc


# ------ 3.- Random Forest with balanced train data
x <- train[,-which(names(train) %in% c("class"))]
y <- as.factor(train$class)
rf_fit <- Rborist(x, y, ntree = 1000, minNode = 20, maxLeaf = 13)

rf_pred <- predict(rf_fit, test_data[,-which(names(test_data) %in% c("Class"))], ctgCensus = "prob")
rf_prob <- rf_pred$prob

rf_roc <- ROSE::roc.curve(test_data$Class, rf_prob[,2], plotit = TRUE)
rf_roc_val <- rf_roc$auc


# ------ 4.- XGBoost with balance train
xgb <- xgboost::xgboost(data = data.matrix(train[,-31]), 
               label = as.numeric(train[,31]),
               eta = 0.1,
               gamma = 0.1,
               max_depth = 10, 
               nrounds = 300, 
               objective = "binary:logistic",
               colsample_bytree = 0.6,
               verbose = 0,
               nthread = 7)

xgb_pred <- predict(xgb, data.matrix(test_data[,-31]))

xgb_roc <- ROSE::roc.curve(test_data$Class, xgb_pred, plotit = TRUE)
xgb_roc_val <- xgb_roc$auc

# Importance plot
names <- dimnames(data.matrix(train[,-31]))[[2]]

# Compute feature importance matrix
importance_matrix <- xgboost::xgb.importance(names, model = xgb)

# Nice graph
xgboost::xgb.plot.importance(importance_matrix[1:7,])  

# Train with only top 7 variables
train_t7 <- train[,which(names(train) %in% importance_matrix$Feature[1:7])]
xgb_t7 <- xgboost::xgboost(data = data.matrix(train_t7), 
                        label = as.numeric(train$class),
                        eta = 0.1,
                        gamma = 0.1,
                        max_depth = 10, 
                        nrounds = 300, 
                        objective = "binary:logistic",
                        colsample_bytree = 0.6,
                        verbose = 0,
                        nthread = 7)

xgb_pred_t7 <- predict(xgb_t7,data.matrix(test_data[,which(names(test_data) %in% importance_matrix$Feature[1:7])]))

xgb_roc_t7 <- ROSE::roc.curve(test_data$Class, xgb_pred_t7, plotit = TRUE)
xgb_roc_val_t7 <- xgb_roc_t7$auc

# Conclussions

# Summarizing ROC Scores
results <- data.frame(Method = c("Random","noFraud","dummy Statistics",
                                 "CART no UpSample","CART Upsample","Random Forest","XGBoost","XGBoost top 7"),
                      ROC_value=c(random_roc_val,noFraud_roc_val, statistic_roc_val,
                                  CART_noSM_roc_val, CART_SM_roc_val, rf_roc_val, xgb_roc_val, xgb_roc_val_t7))
results




