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
However, considering the small number of low_risk population, the precision score is 1% for high_risk and 100% for low_risk. This leads the F1 score to be extremely low for high_risk, only 2% and pretty high for low_risk: 81%.
The recall rate is 62% for high, 68% for low risk.

***Low risk data outweighs high risk data. This results in 100% of precision score for low risk applicants and only 1% precision for high risk applicants.***

## SMOTE

Oversampling the data using SMOTE is much like RandomOverSampler. The only difference in steps is to make sure the training data is **RESAMPLED**:

<img width="720" alt="Screenshot 2023-01-07 at 3 39 03 PM" src="https://user-images.githubusercontent.com/111609994/211174176-71c6a779-7ed5-4d12-a48d-4ce864a3930a.png">

***RESULTS***
The balanced accuracy score for SMOTE is approximately the same: 0.644372. Nearly 65%.
The results seem be the same with 1% of precision for high_risk and 100% precision for low_risk credits. F1 is slightly different for low risk compared to the other method: 79%.

***In addition to RandomOverSampler, SMOTE returned a precision rate of 100% for low risk data and a 1% precision for high risk data. This, again, has to do with a larger low risk population size which forces precision rate to be much lower for high risk and the recall rate of 62%.***

## *ClusterCentroids*

Just as described above, the data is resampled using ***ClusterCentroids***:

<img width="576" alt="Screenshot 2023-01-07 at 3 40 57 PM" src="https://user-images.githubusercontent.com/111609994/211174226-29ec34e2-4e4b-4d0a-9b5c-222870be283b.png">

***RESULTS***

Undersampling resulted in lesser balanced accuracy score of 0.5291, almost 53%.
Precision is the same as in other cases: 1% and 100% respectively. Undersampling resulted in 62% of F1 for low_risk.
Recall for low risk 61% and 45% for high risk. There are too many false positives and the sensitivity results in only 45% for low risk population. 

### Combination: Over and Under

The training data is resampled and Logistic Regression model is used to train it.


<img width="495" alt="Screenshot 2023-01-07 at 3 53 20 PM" src="https://user-images.githubusercontent.com/111609994/211174534-4411eca4-5807-4de3-b030-4d8c2548ebb9.png">

<img width="352" alt="Screenshot 2023-01-07 at 3 53 38 PM" src="https://user-images.githubusercontent.com/111609994/211174538-24b74138-dd44-4f71-9006-a5a6ad9477e7.png">


***RESULTS***

Balanced accuracy score is 64%.
Recall here is 71% for high risk and 55% for low risk. 
Precision is the same as in all other cases: 1% vs 100%.
F1 is slightly higher than other results for low risk: 71%, and the 2% for high_risk.

## Ensemble Learners

This section of the analysis uses 2 ensemble algorithms to determine which one performs better. The algorithms are ***Balanced Random Forest Classifier*** and ***Easy Ensemble AdaBoost Classifier***. 

To work with ensemble learners, number of estimators should be specified. The next steps follow the same process we've seen in the other models.

The training data is resampled with the BalancedRandomForestClassifier:

<img width="640" alt="Screenshot 2023-01-07 at 4 02 48 PM" src="https://user-images.githubusercontent.com/111609994/211174718-faea40b2-092e-4a02-9da1-369b4e4b8aed.png">


Balanced accuracy score is calculated:

<img width="426" alt="Screenshot 2023-01-07 at 4 02 55 PM" src="https://user-images.githubusercontent.com/111609994/211174721-a456cb37-8f72-4796-b2ae-9f01717d67f4.png">


Classification Report is displayed:

<img width="695" alt="Screenshot 2023-01-07 at 4 03 06 PM" src="https://user-images.githubusercontent.com/111609994/211174726-105ce1c5-33b1-4a4f-ace3-c15202564761.png">

***Balanced Random Forest Classifier*** resulted in balance accuracy score of 79%.
Precision here is slightly higher for high risk credits: 4%. Low risk credits have precision of 100% again.
Recall is 67% for high risk and 91% for low risk.
F1, thus, 7% for high risk and 95% for low risk.

So far, compared to the rest of the models, this method returned the smallest number of False Positives, increasing the sensitivity for low risk to 91% and precision 100%. The precision for high risk is still low: 4%


***Easy Ensemble AdaBoost Classifier*** resulted in balance accuracy score of 93%.
Precision here is even higher for high risk credits: 7%. Low risk credits have precision of 100% again.
Recall is 91% for high risk and 94% for low risk.
F1, thus, 14% for high risk and 97% for low risk.

Easy Ensemble AdaBoost returned the highest rate for low risk population sensitivity: 94%. This model resulted in a very low false positives. Even precision for high risk data is higher than the rest of the models: 7%


## Summary

Since the data had a very small number of high credit risk, all these models somehow failed to return a high precision rate. This means that even if the sensitivity for high credit risk is high, the precision rate still fails to identify the correct low and high risk credits. Balanced Random Forest Classifie and Easy Ensemble AdaBoost Classifier models came close with high recall rates for both low and high risk. Still, even with a 7% of precision for high risk credits is not ideal to get rid of a good chunk of false positives.




