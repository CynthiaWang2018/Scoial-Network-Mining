# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 12:40:23 2018

@author: Administrator
"""

import pandas as pd
import numpy as np

movie = pd.read_csv('movie.csv')
user = pd.read_csv('user.csv')
score_1000 = pd.read_csv('score_1000.csv',dtype=int)
list2 = pd.read_csv('list2.csv',header=None,dtype=int)
d1 = score_1000

# 两个数据框进行拼接，类似于cbind()
df = pd.concat([d1,list2],axis=1)

# movie[['rate','id']]选取两列元素而不是movie['rate','id']
df1 = pd.merge(df,movie[['rate','id']],left_on='movie_id',right_on='id',how='outer')
df1 = df1.drop('movie_id',axis=1)

df2 = d1.groupby('user_id').agg('mean').reset_index()
df2.rename(columns={'rates':'user_rate'}, inplace = True)
df2 = pd.merge(df1,df2[['user_rate','user_id']],on='user_id',how='outer')

# 转化成libsvm格式
libsvm = list()
def getlibsvm(df2):
    for i in range(40381):
        libsvm.append(int(df2.rates[i]))
        libsvm.append(str(int(df2.user_id[i])-1)+':'+str(1))
        libsvm.append(str(int(df2.order[i])+999)+':'+str(1))
        libsvm.append(str(1997)+':'+str(df2.rate[i]/2))
        libsvm.append(str(1998)+':'+str(df2.user_rate[i]))
    return libsvm
libsvm = getlibsvm(df2)

libsvm2 = pd.DataFrame(np.array(libsvm).reshape(40381,5))
libsvm2.columns = ['rate','user','movie','movie_bias','user_bias']
libsvm2.to_csv('libsvm2.csv',header=None,index=None)



# 划分训练集合测试集
from sklearn.model_selection import train_test_split
train31, test31, = train_test_split(libsvm2, test_size=0.1, random_state=8)
train32, test32, = train_test_split(libsvm2, test_size=0.1, random_state=9)

# train3_1,test3_1
fw = open("train31.txt", 'w') 
for i in range(len(train31)):
        fw.write(train31.iloc[i,0]+' '+train31.iloc[i,1]+' '+train31.iloc[i,2]
        +' '+train31.iloc[i,3]+' '+train31.iloc[i,4])
        fw.write('\n')
fw.close()

fw = open("test31.txt", 'w') 
for i in range(len(test31)):
        fw.write(test31.iloc[i,0]+' '+test31.iloc[i,1]+' '+test31.iloc[i,2]
        +' '+test31.iloc[i,3]+' '+test31.iloc[i,4])
        fw.write('\n')
fw.close()

# train3_2,test3_2
fw = open("train32.txt", 'w') 
for i in range(len(train32)):
        fw.write(train32.iloc[i,0]+' '+train32.iloc[i,1]+' '+train32.iloc[i,2]
        +' '+train32.iloc[i,3]+' '+train32.iloc[i,4])
        fw.write('\n')
fw.close()

fw = open("test32.txt", 'w') 
for i in range(len(test32)):
        fw.write(test32.iloc[i,0]+' '+test32.iloc[i,1]+' '+test32.iloc[i,2]
        +' '+test32.iloc[i,3]+' '+test32.iloc[i,4])
        fw.write('\n')
fw.close()