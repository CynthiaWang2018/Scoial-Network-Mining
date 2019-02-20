# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 10:06:53 2018

@author: Administrator
"""

import pandas as pd
import numpy as np

movie_vec = pd.read_csv('movie_vec1.csv',header=None)
del movie_vec[0]
del movie_vec[15]

movie_vec.columns = movie_vec.ix[0,:]
movie_vec = movie_vec.drop(0,axis=0)
movie_vec = movie_vec.reset_index()

'''0-999是用户ID，1000-1996是电影ID，1997是电影平均评分，1998是用户平均评分，1999-2025是电影类型'''
'''2026-5168是演员信息'''

# 将演员信息one-hot化
def actor(x):
    n = 0
    m = list()
    l = np.array(eval(x))
    for k in range(len(l)):
        if l[k]==1:
            m.append(str(2026+k)+':'+str(1))
        else:
            n = n+1
    return m
movie_vec['actor_libsvm'] = movie_vec['actor_vec'].apply(actor)

actor = movie_vec[['number','actor_libsvm']]
actor['number'] = actor['number'].astype('int')

df4 = pd.read_csv('df4.csv',header=None)
df4.columns = ['user_id','rate','movie_num','movie_bias','movie_id','user_bias','type']

df6 = pd.merge(df4,actor,how='left',left_on='movie_num',right_on='number')
df6 = df6.dropna(axis=0)


# 转化成libsvm格式
libsvm6 = []
def getlibsvm(d):
    for i in range(len(d)):
        libsvm6.append(d.rate[i])
        libsvm6.append(str(int(d.user_id[i])-1)+':'+str(1))
        libsvm6.append(str(int(d.movie_num[i])+999)+':'+str(1))
        libsvm6.append(str(1997)+':'+str(d.movie_bias[i]/2))
        libsvm6.append(str(1998)+':'+str(d.user_bias[i]))
        libsvm6.append(d.type[i])
        libsvm6.append(str(d['actor_libsvm'][i]))
    return libsvm6
libsvm6 = getlibsvm(df6)

libsvm = pd.DataFrame(np.array(libsvm6).reshape(len(df6),7))
libsvm.columns = ['rate','user','movie','movie_bias','user_bias','type','actor']

# 去除type两端的双引号
l = list()
for i in range(len(libsvm)):
    l.append(eval(libsvm['type'][i]))
libsvm['type'] = l

# 去除actor两端的双引号
l = list()
for i in range(len(libsvm)):
    l.append(eval(libsvm['actor'][i]))
libsvm['actor'] = l

libsvm.to_csv('libsvm6.csv',header=True,index=None)



# 划分训练集合测试集
from sklearn.model_selection import train_test_split
train61, test61 = train_test_split(libsvm, test_size=0.1, random_state=10)
train62, test62 = train_test_split(libsvm, test_size=0.1, random_state=11)
train61 = train61.reset_index()
test61 = test61.reset_index()
train62 = train62.reset_index()
test62 = test62.reset_index()

# 解决类别是-1带来的麻烦
def transpose(list1):
    if -1 in list1:
        l = ''
    else:
        l= ' '+" ".join(i for i in list1)
    return l

def transpose1(list1):
    l= ' '+" ".join(i for i in list1)
    return l


# train61,test61
fw = open("train61.txt", 'w') 
for i in range(len(train61)):
    fw.write(train61['rate'][i]+' '+train61['user'][i]+' '+train61['movie'][i]
        +' '+train61['movie_bias'][i]+' '+train61['user_bias'][i]+transpose(train61['type'][i])
        +transpose1(train61['actor'][i]))
    fw.write('\n')
fw.close()

fw = open("test61.txt", 'w') 
for i in range(len(test61)):
     fw.write(test61['rate'][i]+' '+test61['user'][i]+' '+test61['movie'][i]
        +' '+test61['movie_bias'][i]+' '+test61['user_bias'][i]+transpose(test61['type'][i])
        +transpose1(test61['actor'][i]))
     fw.write('\n')
fw.close()

# train62,test62
fw = open("train62.txt", 'w') 
for i in range(len(train62)):
     fw.write(train62['rate'][i]+' '+train62['user'][i]+' '+train62['movie'][i]
        +' '+train62['movie_bias'][i]+' '+train62['user_bias'][i]+transpose(train62['type'][i])
        +transpose1(train62['actor'][i]))
     fw.write('\n')
fw.close()

fw = open("test62.txt", 'w') 
for i in range(len(test62)):
     fw.write(test62['rate'][i]+' '+test62['user'][i]+' '+test62['movie'][i]
        +' '+test62['movie_bias'][i]+' '+test62['user_bias'][i]+transpose(test62['type'][i])
        +transpose1(test62['actor'][i]))
     fw.write('\n')
fw.close()