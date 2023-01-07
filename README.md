# Credit Risk Analysis


## Overview

Machine learning is believed to make a more accurate identificaiton of good candidates for loans. Since good loans usually outnumber risky ones, understanding credit risk becomes an unbalanced classification problem. With machine learning, it's possible to build and evaluate several models with algorithms to predict credit risk.

***The purpose of this analysis is to employ different techniques to train and evaluate models with unbalanced classes to predict credit risk.***

There are several different parts in this analysis that include oversampling and undersampling the data, a combination of both using the SMOTEENN algorithm and reducing bias using **BalancedRandomForestClassifier** and **EasyEnsembleClassifier** models. 

## ReSampling Models to predict Credit Risk

The first steps into the analysis ***imbalanced-learn*** and ***scikit-learn*** are used to evaluate three machine learning models:

* *OverSampling*: using **RandomOverSampler** and **SMOTE** algorithms.
* *UnderSampling*: using  **ClusterCentroids** algorithm.

# Analysis

Initially, **get_dummies()** method is used to convert the string values into numerical values. Then, target variables are set and their balance is checked. 

<img width="1082" alt="Screenshot 2023-01-07 at 3 17 54 PM" src="https://user-images.githubusercontent.com/111609994/211173772-78ca19de-8491-43d3-a734-a7da37ca76c7.png">

<img width="425" alt="Screenshot 2023-01-07 at 3 18 32 PM" src="https://user-images.githubusercontent.com/111609994/211173776-634b5db8-4761-462a-b600-04bfde61392e.png">

Next, the ***train_test_split*** method is used to split the data:

<img width="761" alt="Screenshot 2023-01-07 at 3 20 35 PM" src="https://user-images.githubusercontent.com/111609994/211173888-c68d6ea2-d5d0-4447-a13c-9c973c4c6e14.png">

## RandomOverSampler

To oversample the data, several steps are used:

1. The training data is resampled with the RandomOversampler
<img width="511" alt="Screenshot 2023-01-07 at 3 27 05 PM" src="https://user-images.githubusercontent.com/111609994/211173936-4aed1e5f-1249-4fd1-a641-ae1698f12686.png">

2. Logistic Regression model is used to train the resampled data.
<img width="511" alt="Screenshot 2023-01-07 at 3 28 06 PM" src="https://user-images.githubusercontent.com/111609994/211173970-b6d6e18f-5aa6-4fed-aa88-db7f8bdd29bc.png">

3. The balanced accuracy score is calculated (0.64975) after setting ***y_pred = model.predict(X_test)***.
<img width="409" alt="Screenshot 2023-01-07 at 3 29 35 PM" src="https://user-images.githubusercontent.com/111609994/211174000-a0038931-bbfa-4a96-b5ae-bf347bc1b141.png">

4. The confusion matrix is displayed:
<img width="754" alt="Screenshot 2023-01-07 at 3 30 19 PM" src="https://user-images.githubusercontent.com/111609994/211174013-6e25a217-25a7-43a4-b2d8-38515ff4e698.png">

5. The imbalanced classification report is displayed:

<img width="726" alt="Screenshot 2023-01-07 at 3 31 05 PM" src="https://user-images.githubusercontent.com/111609994/211174024-3cc18838-6e99-4d60-a366-19a4d871246b.png">

***RESULTS***

The balanced accuracy score from ***RandomOverSampler*** is 0.649, nearly 65%.
However, considering the small number of low_risk credits, the precision score is 1% for high_risk and 100% for low_risk. This leads the F1 score to be extremely low for high_risk, only 2% and pretty high for low_risk: 81%.

## SMOTE

Oversampling the data using SMOTE is much like RandomOverSampler. The only difference in steps is to make sure the training data is **RESAMPLED**:

<img width="720" alt="Screenshot 2023-01-07 at 3 39 03 PM" src="https://user-images.githubusercontent.com/111609994/211174176-71c6a779-7ed5-4d12-a48d-4ce864a3930a.png">

***RESULTS***
The balanced accuracy score for SMOTE is approximately the same: 0.644372. Nearly 65%.
The results seem be the same with 1% of precision for high_risk and 100% precision for low_risk credits. F1 is slightly different for low risk compared to the other method: 79%.


### *ClusterCentroids*

Just as described above, the data is resampled using ***ClusterCentroids***:

<img width="576" alt="Screenshot 2023-01-07 at 3 40 57 PM" src="https://user-images.githubusercontent.com/111609994/211174226-29ec34e2-4e4b-4d0a-9b5c-222870be283b.png">

***RESULTS***

Undersampling resulted in lesser balanced accuracy score of 0.5291, almost 53%.
Precision is the same as in other cases: 1% and 100% respectively. Undersampling resulted in 62% of F1 for low_risk.

### Combination: Over and Under

The training data is resampled and Logistic Regression model is used to train it.


<img width="495" alt="Screenshot 2023-01-07 at 3 53 20 PM" src="https://user-images.githubusercontent.com/111609994/211174534-4411eca4-5807-4de3-b030-4d8c2548ebb9.png">

<img width="352" alt="Screenshot 2023-01-07 at 3 53 38 PM" src="https://user-images.githubusercontent.com/111609994/211174538-24b74138-dd44-4f71-9006-a5a6ad9477e7.png">











