## Cross Validation
<img src="https://www.dummies.com/wp-content/uploads/9781119245513-fg1104.jpg" class="inline"/><br>

## What is Cross Validation?
Cross-validation is a method to help evaluate a model. Specifically, it partitions the dataset into a comprehensive set of permutations, with each unique set referred to as a **fold**, each fold contains one unique set as the **test** data. We run the algorithm through each fold, returning a score, and then we average the set of results for our final metric. 

## What problem does it solve?
Cross-Validation is used to identify and prevent overfitting. An emphasis is placed on the scenario when the amount of data is restrictively small. 

When used properly, we can maximize the volume of data used in training our model.

## Import
```Python3
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
```

## Instantiate LinearRegression() Model
```Python3
def Regression():
    reg = LinearRegression()
    return reg
```

## Deploy Cross Validation
```Python3
def CrossVal(model, X, y, cv1, cv2):
    cvscores_A = cross_val_score(model, X, y, cv=cv1)
    print(np.mean(cvscores_A))
    cvscores_B = cross_val_score(model, X, y, cv=cv2)
    print(np.mean(cvscores_B))
```
