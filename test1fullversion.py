# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 10:56:00 2019

@author: dell-pc
"""
####################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.model_selection import KFold
# Question 1:

test1=np.loadtxt('../data/Test1_features.dat',delimiter=',')
y=np.loadtxt('../data/Test1_labels.dat',delimiter=',')
param = {"learning_rate_init":list(np.arange(0.525,1.025,0.025)), "alpha":list(np.arange(0.025,0.525,0.025))}
std= StandardScaler()
auc_score=0
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(test1):
    train_X,train_y=test1[train_index],y[train_index]
    test_X,test_y=test1[test_index],y[test_index]
    std.fit(train_X)
    train_X=std.transform(train_X)
    test_X=std.transform(test_X)
    ann_model = MLPClassifier(hidden_layer_sizes=15, activation='relu',alpha=0.2,learning_rate_init=0.25, solver='sgd', random_state=0,verbose=False)
    ann_model.fit(train_X,train_y)
    y_pred=ann_model.predict(test_X)
    auc_score+=roc_auc_score(y_true=test_y,y_score=y_pred)
auc_score/=5
print('Mean AUC:%.3f'%auc_score)
if auc_score>0.9:
    print('Mean AUC is satisfied')
###############################################################################
###############################################################################
# Question 2:

std.fit(test1)
test1_std=std.transform(test1)
grid_search = GridSearchCV(MLPClassifier(hidden_layer_sizes=15,activation='relu',solver='sgd',random_state=0),n_jobs=-1,param_grid=param,cv=3,refit='AUC',return_train_score=True)
grid_search.fit(test1_std, y.values.ravel())
result=pd.DataFrame(grid_search.cv_results_)
x = np.arange(0.525, 1.025, 0.025)
y = np.arange(0.025, 0.525, 0.025)
x, y = np.meshgrid(x, y)
z=result['mean_test_score'].reshape(x.shape)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, z, rstride = 1,cstride = 1,cmap = plt.get_cmap('rainbow') ) 
ax.set_zlim(0.5, 1)
ax.set_xlabel('learning_rate')
ax.set_ylabel('alpha')
ax.set_zlabel('AUC_score')
plt.show()