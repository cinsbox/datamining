# K-means Cluster
# Separate data into k clusters in a way that data points in the same 
# cluster are similar and data points in the different clusters are farther apart
# The set of codes below demonstrate how to run K-means cluster using R

# Install the needed packages 
install.packages("cluster") # package for cluster 
install.packages("factoextra") # package for visualizing multivariates 

# Import the packages 
library(cluster)
library(factoextra)

# Read the dataset
my_data <- read.csv(file.choose()) # use customers.csv

# Check the first few rows of data
head(my_data)

# Normalize the data set so bring all variables to same range
kmean_data <- scale(my_data)
head(kmean_data)

# Gap statistic method: use to determine the best K (# of clusters)
# fviz_nbclust is from the factoextra package
# nstart: generate n initial random centroids and choose the best one for the algorithm
# method: name of the method
# nboot: n of bootstrapping
fviz_nbclust(kmean_data, kmeans, nstart = 25, method = "gap_stat", nboot = 50) + labs(subtitle = "Gap statistic method")

# Run K-means clustering 
set.seed(123) # random number generation for reproducibility
km.res <- kmeans(kmean_data, 3, nstart = 25) # 3 is optimal number of clusters

# Visualize the k-means clusters 
# fviz_cluster: for clustering visualization
# km.res: plot k-means cluster
# data: normalized dataset 
# palette: color scheme
# ggtheme = minimalist theme
fviz_cluster(km.res, data = kmean_data, palette = "jco", 
             ggtheme = theme_minimal()) # https://nanx.me/ggsci/articles/ggsci.html


# DBScan (Density-Based Spatial Clustering of Applications with Noise)
# Clustering method based upon densities
# Identifies low density region as outliers
install.packages("dbscan")
library(dbscan)

customers_mat <- as.matrix(my_data[,1:4]) # transform data into matrix 
dbs <- dbscan(customers_mat,eps = 40) # epoch or eps is # of passes the data go through algorithm
dbs
dbs$cluster


# Hierarchical Clustering
cust_clusters <- hclust(dist(my_data[,1:4])) # hclust to perform hierarchial clustering
cust_clusters
plot(cust_clusters) # dendrogram to decide number of clusters 

cust_clusters1 = cutree(cust_clusters, 4) # set 4 clusters 
cust_clusters1 
table(cust_clusters1) # summary table of # of observerations per cluster


# kNN Classification
# KNN is a distance-based algorithm which predicts value based on the number 
# of class observations found in its neighbourhood
# Data from UCI Machine Learning
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# Data set is Breast Cancer Wisconsin Diagnostics (bcwd)
# The data set contains 32 variables (V1 to V32)
# Each variable is a measurable clinical indication related to
# breast cancer.  
# Total observations= 569
# The data is availabe in .dat form, so the read file is changed from 
# the .csv to table during the data import
# The codes below demonstrate the use of kNN Classification

# Install package
install.packages("class") # for classification

# Import library
library(class)

# Import data table
# use read.table for .data
# data is separated by comma ','
bcwd <- read.table(file.choose(), sep =',') # use bcwd.data 

# View data: bcwd
View(bcwd)

# Get rid of first column
bcwd <- bcwd[, -1] # this gets rid of V1 which is ID column

# Normalize data 
data_norm <- function(x) { ((x-min(x))/ (max(x)- min(x)))} # scale between min max range

# as.data.frame: check dataframe or force it
# lapply: apply function over list or vector
bcwd_norm <- as.data.frame(lapply(bcwd[, -1], data_norm)) 

# Check normalized data
summary(bcwd[,2:5])
summary(bcwd_norm[,1:4]) # position changed to 1:4

# Note that normalized data is between 1 and -1

# Set up training and test samples for kNN
# Note that there are 569 observations
# Typical training to test sample ratio is 90:10 or 80:20
# This kNN classification is set at 80:20

# training samples: from observation 1 to 450
bcwd_train <- bcwd_norm[1:450,]

# test samples: from 451 to 569
bcwd_test <-bcwd_norm[451:569,]

# Set up kNN classification
# Estimate k by looking at square root of total observation:
bcwd_pred <- knn(bcwd_train, bcwd_test, bcwd[1:450,1], k=23)

# Display kNN output table
table(bcwd_pred, bcwd[451:569, 1])

# The kNN output result is:
# bcwd_pred  B  M
#         B 92  2
#         M  0 25
#
# The output result can also be treated as a Confusion Matrix where: 
# B-B: True Positive (TP)  # TP = 92
# M-M: True Negative (TN)  # TN = 25
# B-M: False Negative (FN) # FN = 0
# M-B: False Positive (FP) # FP = 2 
# Using confusion matrix, calculate Accuracy, Precision and Recall values:
# Accuracy = (TN + TP)/(TN + FP + TP+ FN)
# Precision = TP/(TP + FP) # specificity
# Recall = TP/(TP+FN)      # sensitivity
# Then based upon Precision and Recall values, calculate F1 score:
# F1 = 2*(Precision * Recall / Precision + Recall) 

# Designate numbers for TP, TN, FN and FP
tp <- 92 # Switch the numbers when different data is used 
tn <- 25
fn <- 0
fp <- 2

# Calculate accuracy
con_accu <- (tn+tp)/(tn+fp+tp+fn)
con_accu

# Calculate precision
con_prec <- tp/(tp+fp)
con_prec

# Calculate recall
con_recall <- tp/(tp+fn)
con_recall

# Calculate F1 score
con_f1 <- 2*(con_prec*con_recall)/(con_prec+con_recall)
con_f1 # F1 score range from 0 to 1, score closer to 1 is better




