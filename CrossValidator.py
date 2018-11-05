from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

def Regression():
    reg = LinearRegression()
    return reg

def CrossVal(model, X, y, cv1, cv2):
    cvscores_A = cross_val_score(model, X, y, cv=cv1)
    print(np.mean(cvscores_A))
    cvscores_B = cross_val_score(model, X, y, cv=cv2)
    print(np.mean(cvscores_B))
