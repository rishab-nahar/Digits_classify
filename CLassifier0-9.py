import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
Train_D=pd.read_csv("train.csv")
Test_D=pd.read_csv("test.csv")
labels=Train_D.iloc[:,0].values
pixel_data=Train_D.iloc[:,1:].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(pixel_data,labels)
from sklearn.svm import SVC
classifier=SVC(C=0.5)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
