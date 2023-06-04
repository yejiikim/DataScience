# DataScience_TermProject 9조
## Open Source SW Contribution

### Predicting Hospital Readmissions: An Analysis of Risk Factors and Model Development
This project aims to analyze a dataset from Kaggle titled "Predicting Hospital Readmissions", focusing on understanding the risk factors associated with hospital readmission.

### Objective
The primary goal of this project is to ascertain whether diabetes, amongst other factors, is a significant predictor of hospital readmissions. To achieve this, we aim to construct a predictive model leveraging the available features in the dataset. This model can potentially uncover hidden patterns and provide valuable insights into the factors that contribute to hospital readmission rates.

The insights drawn from this analysis are expected to have substantial implications for both patient care and healthcare systems. By understanding the factors influencing hospital readmissions, healthcare providers can tailor treatment and intervention programs to mitigate these risks. Consequently, this could lead to improved patient outcomes and a decrease in the demand on healthcare services due to preventable readmissions.


### Hospital Readmission Data Analysis
The provided Python script is intended to perform data preprocessing and set the foundation for machine learning model implementation.

The following Python libraries are utilized in this code:

* numpy: A library for efficient array computations.
* pandas: A library providing various functionalities necessary for data analysis.
* sklearn: A library to implement and train machine learning models.
* matplotlib, seaborn: Libraries for data visualization.

This Python script provides a comprehensive preprocessing and exploration approach to a dataset named hospital_readmissions.csv. The following are the steps that are performed:
#### Overall Process
#### 1. Data Inspection & Data Preprocessing
1)  Data loading
2)  Handling missing values
3)  Data encoding
4)  Feature selection
5)  Data splitting
6)  Feature scaling
7)  Visualizing data distribution after scaling.

#### 2. Data Analyis & Evaluation
This code provides an overview of using different machine learning algorithms for prediction tasks. The given data is split into training and test datasets. 
This code represents a complete machine learning pipeline, from importing necessary libraries to preprocessing, building, evaluating, and optimizing various machine learning models. The models being implemented here include Regression, Classification, and Clustering algorithms.

  ##### Libraries used:
  * Numpy: Used for mathematical and logical operations on arrays.
  * Pandas: Provides data structures and data analysis tools.
  * Scikit-learn: Used for machine learning and data mining.
  * Matplotlib: A plotting library used for creating static, animated, and interactive visualizations.
  * Imblearn: Provides a set of re-sampling techniques commonly used in datasets showing strong between-class imbalance.

  ##### Workflow and Algorithms used:

  * Regression:
  The LinearRegression model is built, fitted, and used to predict the test set values.
  Performance is visualized with scatter plots comparing actual vs predicted values.
  K-fold cross-validation (with k=10) is performed using the 'r2' scoring metric.

  * Classification:
  Decision Tree Classifier and Logistic Regression models are built, fitted, and used to predict the test set values.
  For the Decision Tree Classifier, the tree structure is visualized using plot_tree.
  Confusion Matrix for the Decision Tree Classifier is visualized using heatmap.
  K-fold cross-validation (with k=10) is performed using the 'accuracy' scoring metric for both models.

  * Clustering:
  The K-Means clustering algorithm is built, fitted on the training set and used to predict the test set values.
  The clustering results are visualized using a scatter plot.

  * Handling Class Imbalance:
  The Synthetic Minority Over-sampling Technique (SMOTE) is used to balance the classes.
  Decision Tree Classifier and Logistic Regression models are built and evaluated as before but this time with the balanced data.

  * Hyperparameter Tuning:
    Grid Search Cross Validation is used to find the best parameters for Linear Regression, Decision Tree Classifier, Logistic Regression, Random Forest Classifier, and K-Nearest Neighbors     Classifier. The top 5 parameter combinations are printed.
    Randomized Search Cross Validation is also used to find the best parameters for Linear Regression, Decision Tree Classifier, and Logistic Regression.


  #####  Evaluation Metrics used:
  * Regression: R^2 score is used which provides a measure of how well unseen samples are likely to be predicted by the model.
  * Classification: Accuracy score is used which is the ratio of number of correct predictions to the total number of input samples. Also, confusion matrix is used to understand the           performance of the classification model.
  * Clustering: There is no specific evaluation metric used in this code but the clustering results are visualized.

    In the hyperparameter tuning section, the best parameters and the respective R^2 scores (for regression models) or accuracy scores (for classification models) are displayed. The models      are then re-fit with these best parameters to hopefully improve their performance.


Overall, this code gives a detailed and comprehensive overview of a typical machine learning pipeline with various algorithms and methods. This would be an excellent starting point for      any machine learning project.
