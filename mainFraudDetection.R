if (!require(tidyverse)) install.packages('tidyverse')     # Data manipulation
if (!require(caret)) install.packages('caret')             # Data split
if (!require(Metrics)) install.packages('Metrics')         # Evaluate results
if (!require(rstudioapi)) install.packages('rstudioapi')   # for rstudio working dir
if (!require(Rtsne)) install.packages('Rtsne')             # for tsne plotting
if (!require(smotefamily)) install.packages('smotefamily') # for smote implementation
if (!require(rpart)) install.packages('rpart')             # for decision tree model
if (!require(Rborist)) install.packages('Rborist')         # for random forest model
if (!require(corrplot)) install.packages('corrplot')       # for data visualization
if (!require(ROSE)) install.packages('ROSE')       # for ROC curves

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

# Check na in the data set
no_na <- sum(is.na(fraud_data))

# Take a look to our data
glimpse(fraud_data)


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

# Visualization of time values
fraud_data %>%
  ggplot(aes(x = Time, fill = factor(Class))) + geom_histogram(bins = 100)+
  labs(x = 'Time in seconds since first transaction', y = 'Number of transactions') +
  ggtitle('Distribution of time of transaction by class') +
  facet_grid(Class ~ ., scales = 'free_y')

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

# A we have a very unbalance dataset the procedure will consist in split data into train and test 5 times
# in order to perform cross validation. For this we will divide the data in 5 sections, each one of this sections
# will be the test data in each cross validation set and the rest for training, allowing us to have a 80 to 20
# relation between train and test data, and having different test data for each one of the sets.

# Each training data will be upsampled in the fraud cases in order to have a balance training dataset.
# The upsample method will be SMOTE (synthetic minority oversampling technique)

# Getting our 5 test samples
set.seed(2021)
index <- createDataPartition(y = fraud_data$Class, times = 5, p = .8, list = FALSE)

train_data <- fraud_data[index[,1],]
test_data <- fraud_data[-index[,1],]


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
# 3.- Dumy stadistic: Based on the probability between fraud and no fraud, chose one of them based
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


# ----------- 3. Dumy Stadistics
counts <- train_data %>% group_by(Class) %>% count()
noFraudC <- counts$n[1]
fraudC <- counts$n[2]
relationship <- fraudC / (fraudC + noFraudC)
y_hat_stadistic <- sample(c(0,1), size = size, replace = TRUE, prob = c(1-relationship,relationship))

stadistic_acc <-  accuracy(test_data$Class, y_hat_stadistic)
stadistic_roc <- ROSE::roc.curve(test_data$Class, y_hat_stadistic, plotit = TRUE)
stadistic_roc_val <- stadistic_roc$auc
stadistic_f1 <- F_meas(data = as.factor(y_hat_stadistic), reference = test_data$Class)

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


# ------ 3.- Random Forest with balance output
x <- train[,-31]
y <- as.factor(train[,31])
rf_fit <- Rborist(x, y, ntree = 1000, minNode = 20, maxLeaf = 13)

rf_pred <- predict(rf_fit, test_data[,-31], ctgCensus = "prob")
rf_prob <- rf_pred$prob

rf_roc <- ROSE::roc.curve(test_data$Class, rf_prob[,2], plotit = TRUE)
rf_roc_val <- rf_roc$auc


