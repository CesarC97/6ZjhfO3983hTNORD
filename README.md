## Introduction
The main focus of this project is on a recent survey to a select customer cohort done by one of the fastest growing startups in the logistics and delivery domain that thrives on making their customers happy. A subset of this data is provided.
The primary objective of this project is to predict if a customer is happy or not based on the answers they have given to the questions asked based on different predictive models. The models considered for prediction are *Logistic Regression, Deep Learning, and Ensemble Learning models*.

## Data Description
- *Y* = Target attribute with values indicating 0 (unhappy) and 1 (happy) customers
- *X1* = My order was delivered on time
- *X2* = Contents of my order was as I expected
- *X3* = I ordered everything I wanted to order
- *X4* = I paid a good price for my order
- *X5* = I am satisfied with my courier
- *X6* = The app makes ordering easy for me

Attributes X1 to X6 indicate the responses for each question and have values from 1 to 5; the smallest number indicates less likely and the highest number indicates more likely towards the answer.

## Exploratory Data Analysis
The data set has 126 observations and 7 variables, it has no missing data, every variable is numerical with integer values; however, these variables can also be considered as ordinal categorical variables. The target variable *Y* has more happy customers than unhappy with proportions of 54.8% and 45.2%, respectively.

## Happiness Prediction
This step is focused on how to predict whether or not customers are happy by training different machine learning models. We train a logistic model and some deep learning models for happiness prediction. For each model, the data is split into training data and test data with sizes of 80% and 20%, respectively. Additionally, evaluation metrics will be reported.

## Logistic Regression Model
This model uses the sigmoid function to link the log-odds of a data point to the range $[0,1]$. Since the only outcomes of the target variable Y are 0(Unhappy) and 1(Happy), the logistic regression model is very useful since the sigmoid function provides a probability for the classification decision (Happy or Unhappy).

- The majority of the predicted values for Y are as equal as the test values from the test data
- The predicted probabilities obtained from the model estimated whether the customer is happy or not; using a threshold of 0.50, it can be easy to predict customers' happiness
- The confusion matrix indicates the True Negatives, True Positives, False Positives, and False Negatives proportions, and such proportions are helpful to calculate the precision, recall, f1-score, and accuracy for model evaluation
- The model's accuracy is 0.85, which indicates that the model has a little bias predicting more happy customers than unhappy
- The Receiving Operating Characteristic (ROC) curve can be obtained by calculating and plotting the True Positive Rate vs. the False Positive Rate. The Area Under the Curve (AUC) is helpful to determine how well the model can distinghish between the two classes (0 and 1). Since the Area Under the Curve is approximately equal to 0.87 and is close to 1, this indicates that the logistic regression model is almost a perfect classifier

## Other Classifiers
On this step, some other classifiers will be implemented to determine if the accuracy can be improved from the previous model.

## Random Search Hyperparameter Optimization
The accuracy for every single predictive model was significantly high. However, a random search hyperparameter optimization will be required to improve the accuracy of the happiness prediction. This method searches for a range of values to a subset of the dataset to obtain the best performance of the models. Randomized search and grid search are the most common methods for hyperparameter optimization.

## Recursive Feature Elimination
Recursive Feature Elimination is a selection method that iteratively remove less significant features, and it keeps the ones that enhance predictive accuracy. This method is significantly better to determine which features are more important.

LASSO stands for Least Absolute Shrinkage and Selection Operator and has the ability to set some coefficients to zero. This is the penalty selected for the recursive feature elimination. X2, X4, and X6 are the features (variables) that need to be eliminated to improve the accuracy of the models. Now the training and test datasets have 3 remaining features.

## Refitting Logistic Regression Model
Based on previous results, the logistic regression model can be trained with the 3 remaining features obtained from recursive feature elimination (X1, X3, and X5). The accuracy and the other model evaluation metrics are exactly the same as the previous model's.

## Conclusion
The primary objective of this project was to predict if a customer is happy or not based on the answers they have given to the questions asked based on different predictive models. Based on the different approaches performed, there are some interesting insights about the project:

- Based on the exploratory data analysis performed, the distribution for most of the variables is left-skewed, which indicates that most customers tend to be happy when their answers to the questions are in the highest levels (4 and 5). It seems to be a weak or no correlation at all between every single question, but further analysis demonstrates that there are some questions that are more relevant for happiness prediction than others.
- The Logistic Regression model has a relatively high accuracy score (0.85), and so for the other scores for each class (Happy and Unhappy) based on the confusion matrix. Also, the Receiving Operating Characteristic (ROC) Curve has an area under the curve of 0.87 (relatively close to 1), which indicates the measure of the logistic model's ability to distinguish the customers that are happy against the ones that are not.
- Some other trained models, such as Naive Bayes Classifier, Linear SVC Classifier, Bernoulli NB Classifier, and Hard Voting Classifier have as relatively high accuracy scores as the Logistic Regression's. It happens to notice that not only the accuracy score from such models, but also the precision, recall, and f1 scores for each class (Happy and Unhappy) are relatively high and close to 1 based on the confusion matrices, which means that each class has a relatively high true prediction than a false one.
- The Random Search Hyperparameter Optimization and Recursive Feature Elimination methods were considered to select a set of variables that have more relative importance than others to improve the accuracy of the trained models. However, Random Search Hyperparameter Optimization suggested to consider every single variable, and Recursive Feature Elimination determined to not consider the expected contents of the order (X2), the price of the order was good (X4), and the easiness of the app for ordering (X6).
- This logistics and delivery startup might give partial importance to such questions to determine whether a customer is happy or not because it might tend to some bias; a logistic model was trained with the remaining questions and yielded the same results on the classification report from the original trained logistic model. The remaining models might yield the same or better prediction results with the remaining three questions, but every question has either the same or higher importance for happiness prediction.
