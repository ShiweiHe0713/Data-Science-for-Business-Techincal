# Data-Science-for-Business-Techincal

**Table  of contents**
- [Data-Science-for-Business-Techincal](#data-science-for-business-techincal)
- [Topics 4](#topics-4)
  - [Cross validation](#cross-validation)
    - [Grid search](#grid-search)
- [Topic 6 Linear Classification \& Regression](#topic-6-linear-classification--regression)
- [Topic 7 Feature Engineering \& Variable Selection](#topic-7-feature-engineering--variable-selection)
- [Topic 8 Similarity, Neighbors and Clustering](#topic-8-similarity-neighbors-and-clustering)

# Topics 4
## Cross validation

**ROC**: A Receiver Operating Characteristic (ROC) curve is a graph that shows the performance of a binary classification method. It plots TPR vs. FPR.
<img alt="ROC curve" src="assets/ROC.png" style="zoom: 25%;">

**AUC**: "Area under the ROC Curve"

### Grid search


# Topic 6 Linear Classification & Regression

Linear classification is achieved via Linear Discriminant: Class(x) = 1._ 2._ 

`from sklearn.discriminant_analysis import LinearDiscriminantAnalysis` 

**SVM**

The optimal SVM is the boundary that maximizes the margin. 

`from sklearn import svm`

Complexity parameters of SVM:

1. C: The cost. Higher C fits training data more closely
2. Kernel: Linear or non-linear with degree of polynomial

**Regression Models**

Prediction variable is numeric - continuous. Decision Tree is a classification models, NOT regression.

**Regression Trees**

`from sklearn.tree import DecisionTreeRegressor`

**Linear Regression**

`from sklearn.linear.model import LinearRegression`

- Linear Regression is to predict a numeric variable from one or multiple variables(which can be numeric or categorical)

**Linear Relationships**  $Y = \beta_0 + \beta_1 X$

Correlation coefficient `r_xy` :

- 0 means no relation, -1 means negative, 1 means postive relationship
- Least squares regression line is the line that minimizes the sum of **squared** residuals (Make errors don’t cancel out)
- use `statmodels` to get regression output
- $\beta_0$ is the Y-intercept. It is not reliable if it’s outside the range of data (Extrapolation)
- The interpretation of beta_1 can’t be casual

**Regression concerns**

- Violations of assumption: linear relationship, normality of residuals, constant variance
- Regression can be sensitive to: outliers, leverage points, highly correlated predictors

**Logistic Regression**

`from sklearn.linear_model import LogisticRegression`

**Loss function**

Linear Regression - Squared-error loss, Logistic Regression - Logistic Loss

**Complexity** 

More attributes means more complex relationships. Categorical variables can explore dimensionality. 

**Reduce complexity:** 1. Reduce the number of attributes(Variable selection); 2. Use a penalty in the objective function(Regularization).

**Regularized Regression:** The least squares regression line is the line that minimizes the sum of the **squared** residuals.

Ridge/L2 regression: Estimates are squared.

Lasso/L1 regression: Estimates are absolute value.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/9118a158-2790-4ffc-bcd6-b7a05fc9272d/b7e95cd2-7f4d-473e-8d6b-55495f002bf0/Untitled.png)

**λ is the penalty parameter**, protect us from overfitting.

Regularization is also referred to as **shrinkage.**

λ = 0 means no regularization, larger is stronger penalty, can be determined by cross-validation.

**Lasso Regression:** the lasso will “shrink” parameter estimates towards zero, has same effect as variable selection.

# Topic 7 Feature Engineering & Variable Selection

- Add attributes: feature construction, feature engineering
- Remove attributes: dimension reduction, variable selection

**Example1**: Predict whether a customer will respond to a special offer, based on previous purchases

- We want every entry to be every people, and to avoid redundancy
- Eg. Turn 10 purchases entry into 1 purchase sum amount

**Example2 Netflix prize:** Predict if someone’s gonna like Jurassic Park

**Things to do before modeling**

- Categorical → dummies, numeric → categorical: highest 10%, or H/M/L categories, binning of target or attribute for long tail
- strings → date variables, extract year/month/day from data
- Combine columns: mean, max, min or total might be more relevant
- Standardize features:
    - `StandardScaler` : Converts to Z-Scorel, MinMaxScaler, to [0,1]

**Example3: Network data: Top 2 buddies were more predictive than anything else**

- Why removing unimportant attributes?
    
    Model accuracy, overfitting, efficiency, interpretability, unintended data leakage
    

**Feature selection by Addition**

Identify each feature’s impact: correlation, ROC curve, information gain, regression table - find all of the significant features and remove non-sigs

- Can also do this iteratively, if there is collinearity

**Iterative feature subtraction**

Backward elimination(Regression): Improve $R^2$ every step til none of the remaining variables when removed increase $R^2_{adj}$

- In every iteration, when the process suggests removing a categorical variable, it means removing all its dummy variables (levels) at once, not just one(e.g. Removing is_red, is_blue, is_fellow all together for color)
- L1 Regularization is another way to reduce the number of features
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/9118a158-2790-4ffc-bcd6-b7a05fc9272d/1905ad6b-dbb3-4b9f-abfd-09d4749b7dc0/Untitled.png)
    
    Use CV to find the best value of lambda, fit data with optimal lambda, remove features w/ coefficient = 0
    

**Example online advertising: Dstillery,** ad targeting inc in NYC

**How to solve cold start?** where we don’t have data yet to train on? What to do if there are very few positive examples? Reducing attribute space? 

- **Feature reduction - Clustering:** Dstillery bought data that allowed them to categorize all URLs into topics

**Dimension Reduction - Principle Components**

- Principal Components Analysis(PCA): High dimensional space → lower
- Take original data and reduct to top principle components, fit regression on PC instead of original data(Variable should be normalized for PCA)

# Topic 8 Similarity, Neighbors and Clustering

Use case of similarity in business: Find **similar products** to existing products to identify recommendations; Identify **items/customers** similar to your known. best customers

- Unsupervised learning: Use similarities. to group similar items into clusters

**Distance**

We use normally Euclidean distance to define distance, and we turn non-numeric data into numeric data also for creating distances

**Different types of similarity/distances**

- Numeric Distance: Euclidean, Manhattan, Cosine distance
- Binary vectors: Hamming distance (# of changes from A to B))
- Jaccard similarity: Intersection over union