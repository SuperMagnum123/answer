# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:32:52 2019

@author: dell-pc
"""
#####################################
import pandas as pd
import numpy as np
test2=pd.read_csv('../data/Test2_Data.csv',engine='python')
c=[]
d=[]
a=0
for i in list(test2.columns):
    a=sum(test2[test2[i]==1]['Label']==1)
    if a>120:
        d.append(i)
    if a>120 and (a/sum(test2[i]==1))>0.7:
        c.append(i)
c.pop()
d.pop()
b=0
total=0
for i in range(len(d)):
    for j in range(i+1,len(d),1):
        b=test2[(test2[d[i]]==1) & (test2[d[j]]==1) & (test2['Label']==True)].shape[0]
        total=test2[(test2[d[i]]==1) & (test2[d[j]]==1)].shape[0]
        if b>120 and b>0.7*total:
            c.append([d[i],d[j]])
f=open(r'./规则.txt','w')
for i in range(len(c)):
    if type(c[i])==str:
        print(c[i]+'->'+'Label',file=f)
    else:
        for j in c[i]:
            print(j+',',end=' ',file=f)
        print('->'+'Label ',file=f)