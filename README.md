## Cross Validation
<img src="https://www.dummies.com/wp-content/uploads/9781119245513-fg1104.jpg" class="inline"/><br>

_______________________________________________________________________________________________________________________________________
## What is Cross Validation?

### Statistical Theory    
In statistics, Cross-Validation, is also known as rotation estimation, and is used primarily to assay how well your predictive model can be generalized and successfully deployed to a previously unseen set of data.

### Machine Learning Basis   
In machine learning, it's used as methodology for validating that your generated model might -actually- work in the proper sense, i.e, isn't fatally flawed due to training errors in the form of over/under fitting or selection bias, and if it is potentially capable of being generalized and used for a set of unseen data. 

## How Does It Work?    
Cross-validation is simply taking a large dataset, breaking it into smaller chunks, training on all but one chunk, evaluating the un-trained chunk by testing it to get an accuracy metric, then repeating this process with another chunk. You continue until each possible chunk has been trained on, and them sum the average scores you've obtained. 

Another way to think about this, is if you had a list of 10 numbers, 1-10. You select #1 to be tested, then train on the remaining 9, and compute a score. You repeat this with position 2, and train on 1 and 3-10, then you select position 3 to be tested, and you train on 1-2, 4-10, until every number has been used as a test subject, and compute the average result. 

### K-Fold Cross Validation   
K-Fold Cross Validtion is an approach where you partition the dataset into a comprehensive set of `k` permutations, with each unique set referred to as a **fold**, each fold contains one unique set as the **test** data. We run the algorithm through each fold, returning a score, and then we average the set of results for our final metric. 

_______________________________________________________________________________________________________________________________________
## What problem does it solve?
Cross-Validation is used to identify and prevent overfitting. An emphasis is placed on the scenario when the amount of data is restrictively small. 

When used properly, we can maximize the volume of data used in training our model.

_______________________________________________________________________________________________________________________________________
# Code
There are many ways to replicate a CV approach with programming. This is simply one approach using Sci-Kit Learn's `cross_val_score` module to compute the CV score, and we'll use it by creating a set of Python methods to automate the process for us. 

It's a simple, but useful example program to help learn how cross validation and k-folds can be used to augment and test your model. 

## How To Use
Generate a model! Then, simply pass the model, your features, your target variable, and two k-fold values into our `CrossVal()` method, to compare the accuracy and effect of differing k-fold values.  
_______________________________________________________________________________________________________________________________________

## Import
```Python3
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
```

_______________________________________________________________________________________________________________________________________
## Instantiate LinearRegression() Model
```Python3
def Regression():
    reg = LinearRegression()
    return reg
```

_______________________________________________________________________________________________________________________________________
## Deploy Cross Validation
```Python3
def CrossVal(model, X, y, cv1, cv2):
    cvscores_A = cross_val_score(model, X, y, cv=cv1)
    print(np.mean(cvscores_A))
    cvscores_B = cross_val_score(model, X, y, cv=cv2)
    print(np.mean(cvscores_B))
```
