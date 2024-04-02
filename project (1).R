#import important libraries 
library("OpenMx")
library("lavaan")
library("semPlot")
library("tidyverse")
library("mice")
library("MoEClust")
library("performance")
library("mediation")
library("Hmisc")
# Load the data from the CSV file
customer_shopping_data <- read.csv("D:/work/So/customer_shopping_data_1695379411426.csv")
str(customer_shopping_data)
head(customer_shopping_data,5)

#Data cleaning and check missing values
data_var <-
  customer_shopping_data %>%
  dplyr::select(1:9) 
md.pattern(data_var)

#Frequency Distribution of categorical variables
dataframe_categorical_variables=data.frame(customer_shopping_data$gender,customer_shopping_data$category
                                           ,customer_shopping_data$payment_method)
describe(dataframe_categorical_variables)


#Task 1.1

library(ggplot2)
library(corrplot)
#For input
customer_shopping_data$invoice_date <- as.Date(customer_shopping_data$invoice_date, format = "%d/%m/%Y")

# Extract month from invoice date
customer_shopping_data$invoice_month <- format(customer_shopping_data$invoice_date, "%Y-%m")
agg_data <- aggregate(cbind(age, price, quantity) ~ invoice_month, data = customer_shopping_data, sum)
# Convert aggregated data into a time series object
ts_data <- ts(agg_data [, c("age", "price", "quantity")], 
              start = c(2021, 1), frequency = 12)
# Plot the time series
plot.ts(ts_data, 
        main = "Time Series Plot of Age, Price, and Quantity",
        ylab = "Values",
        col = 1:3) # Different colors for each variable
dev.off()
#For output
total_sales <- aggregate(quantity ~ invoice_month, data = customer_shopping_data, sum)
sales_ts <- ts(total_sales$quantity, start = c(2021, 1), frequency = 12)
plot(sales_ts, main = "Time Series Plot of Total Sales Quantity by Month", xlab = "Date", ylab = "Total Sales Quantity")

#Task 1.2
age_hist <- hist(customer_shopping_data$age, plot = FALSE)
age_density <- density(customer_shopping_data$age)

# Histogram density for price
price_hist <- hist(customer_shopping_data$price, plot = FALSE)
price_density <- density(customer_shopping_data$price)

# Histogram density for quantity
quantity_hist <- hist(customer_shopping_data$quantity, plot = FALSE)
quantity_density <- density(customer_shopping_data$quantity)

# Setting up the graphical parameters for multiple plots
par(mfrow = c(1, 3))  # 1 row, 3 columns

# Plotting histogram densities
plot(age_density, col = "skyblue", main = "Density of Age", xlab = "Age", ylab = "Density")
lines(age_density, col = "blue")

plot(price_density, col = "salmon", main = "Density of Price", xlab = "Price", ylab = "Density")
lines(price_density, col = "red")

plot(quantity_density, col = "lightgreen", main = "Density of Quantity", xlab = "Quantity", ylab = "Density")
lines(quantity_density, col = "darkgreen")


# Bar plot for category
category_freq <- table(customer_shopping_data$gender)
barplot(category_freq, main = "Distribution of gender", xlab = "Category", ylab = "Frequency", col = "lightgreen")


# Bar plot for payment method
payment_method_freq <- table(customer_shopping_data$payment_method)
barplot(payment_method_freq, main = "Distribution of Payment Method", xlab = "Payment Method", ylab = "Frequency", col = "lightblue")

# Bar plot for category
category_freq <- table(customer_shopping_data$category)
barplot(category_freq, main = "Distribution of Category", xlab = "Category", ylab = "Frequency", col = "lightgreen")


#Task 1.3
cor(customer_shopping_data[, c("age", "quantity", "price")])

par(mfrow = c(1, 2))  # 1 row, 3 columns

# Scatter plots
with(customer_shopping_data, {
  plot(age, quantity, main = "Age vs. Quantity", xlab = "Age", ylab = "Quantity")
  plot(price, quantity, main = "Price vs. Quantity", xlab = "Price", ylab = "Quantity")
 })


#Task 2.1
# Extracting predictor variables
X2 <- as.numeric(factor(customer_shopping_data$category))
X4 <- as.numeric(factor(customer_shopping_data$payment_method))
X1 <- customer_shopping_data$age
X3 <- customer_shopping_data$price

# Creating polynomial features
X_poly <- cbind(X1^2, X1^3, X1^4, X2^4, X3^4, X3^3, X4)

# Adding intercept column
X_poly <- cbind(1, X_poly)

# Convert target variable to matrix
y <- as.matrix(customer_shopping_data$quantity)
X <- cbind(X1,X2,X3,X4)
# Fit ridge regression models
library(glmnet)

alpha <- 0  # poly regression
lambda <- 1  # Regularization parameter

# Define design matrices for each model
Y1 <- cbind(1, X4, X1^2, X1^3, X2^4, X1^4)
Y2 <- cbind(1, X4, X1^3, X3^4)
Y3 <- cbind(1, X3^3, X3^4)
Y4 <- cbind(1, X2, X1^3, X3^4)
Y5 <- cbind(1, X4, X1^2, X1^3, X3^4)

# Fit poly regression models
poly_model1 <- glmnet(Y1, y, alpha = alpha, lambda = lambda)
poly_model2 <- glmnet(Y2, y, alpha = alpha, lambda = lambda)
poly_model3 <- glmnet(Y3, y, alpha = alpha, lambda = lambda)
poly_model4 <- glmnet(Y4, y, alpha = alpha, lambda = lambda)
poly_model5 <- glmnet(Y5, y, alpha = alpha, lambda = lambda)

# Print coefficients
print(coef(poly_model1))
print(coef(poly_model2))
print(coef(poly_model3))
print(coef(poly_model4))
print(coef(poly_model5))


#2.2
# Predictions for each model
y_pred_1 <- predict(poly_model1, newx = Y1)
y_pred_2 <- predict(poly_model2, newx = Y2)
y_pred_3 <- predict(poly_model3, newx = Y3)
y_pred_4 <- predict(poly_model4, newx = Y4)
y_pred_5 <- predict(poly_model5, newx = Y5)

# Compute residuals for each model
residuals_1 <- y - y_pred_1
residuals_2 <- y - y_pred_2
residuals_3 <- y - y_pred_3
residuals_4 <- y - y_pred_4
residuals_5 <- y - y_pred_5

# Compute RSS for each model
RSS_1 <- sum(residuals_1^2)
RSS_2 <- sum(residuals_2^2)
RSS_3 <- sum(residuals_3^2)
RSS_4 <- sum(residuals_4^2)
RSS_5 <- sum(residuals_5^2)

# Print RSS for each model
cat("RSS for Model 1:", RSS_1, "\n")
cat("RSS for Model 2:", RSS_2, "\n")
cat("RSS for Model 3:", RSS_3, "\n")
cat("RSS for Model 4:", RSS_4, "\n")
cat("RSS for Model 5:", RSS_5, "\n")

#2.3 Compute log-likelihood for each model
log_likelihood_1 <- sum(dnorm(y, mean = y_pred_1, sd = sqrt(var(residuals_1)), log = TRUE))
log_likelihood_2 <- sum(dnorm(y, mean = y_pred_2, sd = sqrt(var(residuals_2)), log = TRUE))
log_likelihood_3 <- sum(dnorm(y, mean = y_pred_3, sd = sqrt(var(residuals_3)), log = TRUE))
log_likelihood_4 <- sum(dnorm(y, mean = y_pred_4, sd = sqrt(var(residuals_4)), log = TRUE))
log_likelihood_5 <- sum(dnorm(y, mean = y_pred_5, sd = sqrt(var(residuals_5)), log = TRUE))

# Print log-likelihood for each model
cat("Log-Likelihood for Model 1:", log_likelihood_1, "\n")
cat("Log-Likelihood for Model 2:", log_likelihood_2, "\n")
cat("Log-Likelihood for Model 3:", log_likelihood_3, "\n")
cat("Log-Likelihood for Model 4:", log_likelihood_4, "\n")
cat("Log-Likelihood for Model 5:", log_likelihood_5, "\n")

#2.4
# Number of observations
n <- length(y)

# Count number of parameters for each model
num_params_1 <- sum(coef(poly_model1) != 0)
num_params_2 <- sum(coef(poly_model2) != 0)
num_params_3 <- sum(coef(poly_model3) != 0)
num_params_4 <- sum(coef(poly_model4) != 0)
num_params_5 <- sum(coef(poly_model5) != 0)

# Compute AIC for each model
AIC_1 <- -2 * log_likelihood_1 + 2 * num_params_1
AIC_2 <- -2 * log_likelihood_2 + 2 * num_params_2
AIC_3 <- -2 * log_likelihood_3 + 2 * num_params_3
AIC_4 <- -2 * log_likelihood_4 + 2 * num_params_4
AIC_5 <- -2 * log_likelihood_5 + 2 * num_params_5

# Compute BIC for each model
BIC_1 <- -2 * log_likelihood_1 + log(n) * num_params_1
BIC_2 <- -2 * log_likelihood_2 + log(n) * num_params_2
BIC_3 <- -2 * log_likelihood_3 + log(n) * num_params_3
BIC_4 <- -2 * log_likelihood_4 + log(n) * num_params_4
BIC_5 <- -2 * log_likelihood_5 + log(n) * num_params_5

# Print AIC and BIC for each model
cat("AIC for Model 1:", AIC_1, "\n")
cat("AIC for Model 2:", AIC_2, "\n")
cat("AIC for Model 3:", AIC_3, "\n")
cat("AIC for Model 4:", AIC_4, "\n")
cat("AIC for Model 5:", AIC_5, "\n")

cat("\n")

cat("BIC for Model 1:", BIC_1, "\n")
cat("BIC for Model 2:", BIC_2, "\n")
cat("BIC for Model 3:", BIC_3, "\n")
cat("BIC for Model 4:", BIC_4, "\n")
cat("BIC for Model 5:", BIC_5, "\n")


#2.5
# Load necessary libraries
library(ggplot2)

# Function to create Q-Q plot
create_qq_plot <- function(residuals, model_name) {
  qq_data <- data.frame(Theoretical = quantile(residuals, probs = seq(0, 1, 0.01)),
                        Sample = quantile(rnorm(length(residuals), mean = mean(residuals), sd = sd(residuals)), probs = seq(0, 1, 0.01)))
  
  ggplot(qq_data, aes(x = Theoretical, y = Sample)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    geom_point(color = "blue") +
    labs(title = paste("Q-Q Plot for Model", model_name),
         x = "Theoretical Quantiles",
         y = "Sample Quantiles") +
    theme_minimal()
}

# Create Q-Q plots for each model
qq_plot_1 <- create_qq_plot(residuals_1, "1")
qq_plot_2 <- create_qq_plot(residuals_2, "2")
qq_plot_3 <- create_qq_plot(residuals_3, "3")
qq_plot_4 <- create_qq_plot(residuals_4, "4")
qq_plot_5 <- create_qq_plot(residuals_5, "5")

# Display Q-Q plots
library(gridExtra)
grid.arrange(qq_plot_1, qq_plot_2, qq_plot_3, qq_plot_4, qq_plot_5, ncol = 3)
dev.off()
#2.7
# Load necessary libraries
library(glmnet)
library(boot)
library(ggplot2)
# Set seed for reproducibility
set.seed(123)

# Split the dataset into training and testing sets (70% training, 30% testing)
train_indices <- sample(nrow(X), 0.7 * nrow(X))
X_train <- Y3[train_indices, ]
y_train <- y[train_indices]
X_test <- Y3[-train_indices, ]
y_test <- y[-train_indices]

# Convert X_train matrix to data frame
X_train_df <- as.data.frame(X_train)

# Train your selected "best" model using the training dataset
best_model <- poly_model3  # Replace with your best model
best_model_fit <- lm(y_train ~ ., data = X_train_df)  # Fit linear regression model

# Make predictions on the testing data
predictions <- predict(best_model_fit, newdata = as.data.frame(X_test), interval = "confidence")

# Plot the actual testing data and the model predictions
plot(y_test, col = "blue", pch = 16, xlab = "Index", ylab = "Sales", main = "Actual vs. Predicted Sales")
points(predictions[, 1], col = "red", pch = 16)  # Model predictions
lines(predictions[, 1], col = "red")  # Connect predictions with a line
segments(1:length(y_test), predictions[, 2], 1:length(y_test), predictions[, 3], col = "green")  # Confidence intervals
legend("topright", legend = c("Actual", "Predicted", "95% CI"), col = c("blue", "red", "green"), pch = c(16, 16, NA), lty = c(NA, 1, NA))


#3.1
# Assuming the estimated values are:
V1_estimated <- 2.954292e+00  # Intercept
V4_estimated <- 0  # No coefficient for V4 in Model 3

# Step 3: Define Prior Distributions
V2_estimated <- 8.594136e-12
V3_estimated <- 9.581656e-16

prior_range_V2 <- c(V2_estimated - 0.1 * abs(V2_estimated), V2_estimated + 0.1 * abs(V2_estimated))
prior_range_V3 <- c(V3_estimated - 0.1 * abs(V3_estimated), V3_estimated + 0.1 * abs(V3_estimated))

# Step 4: Draw Samples
n_samples <- 100  # Number of samples
V2_samples <- runif(n_samples, min = prior_range_V2[1], max = prior_range_V2[2])
V3_samples <- runif(n_samples, min = prior_range_V3[1], max = prior_range_V3[2])
# Joint posterior distribution for V2 and V3
plot(V2_samples, V3_samples, main = "Joint Posterior Distribution for V2 and V3", xlab = "V2", ylab = "V3", col = "orange")

# Scatterplot with marginal histograms for V2 and V3
pairs(data.frame(V2 = V2_samples, V3 = V3_samples), main = "Joint Posterior Distribution with Marginals", col = "purple")


