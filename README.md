## Cross Validation
<img src="https://www.dummies.com/wp-content/uploads/9781119245513-fg1104.jpg" class="inline"/><br>
Cross-validation is a method to help evaluate a model. 

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
