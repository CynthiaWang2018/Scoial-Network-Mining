# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 11:51:49 2018

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

movie_vec.columns = ['index', 'number', 'rate', 'title', 'url', 'id', 'directors', 'year',
       'actors', 'type', 'countries', 'summary', 'type_vec', 'actor_vec',
       'director_vec', 'summary_vec', 'actor_libsvm']

'''0-999是用户ID，1000-1996是电影ID，1997是电影平均评分，1998是用户平均评分，1999-2025是电影类型'''
'''2026-5168是演员信息，5169-5521是summary信息,共353列'''

def summary(x):
    l = list()
    for i in range(353):
        l.append(str(5169+i)+':'+str(eval(x)[i]))
    return l
movie_vec['summary_libsvm'] = movie_vec['summary_vec'].apply(summary)

summary = movie_vec[['number','summary_libsvm','actor_libsvm']]
summary['number'] = summary['number'].astype('int')

df4 = pd.read_csv('df4.csv',header=None)
df4.columns = ['user_id','rate','movie_num','movie_bias','movie_id','user_bias','type']

df7 = pd.merge(df4,summary,how='left',left_on='movie_num',right_on='number')
df7 = df7.dropna(axis=0)



# 转化成libsvm格式
libsvm7 = []
def getlibsvm(d):
    for i in range(len(d)):
        libsvm7.append(d.rate[i])
        libsvm7.append(str(int(d.user_id[i])-1)+':'+str(1))
        libsvm7.append(str(int(d.movie_num[i])+999)+':'+str(1))
        libsvm7.append(str(1997)+':'+str(d.movie_bias[i]/2))
        libsvm7.append(str(1998)+':'+str(d.user_bias[i]))
        libsvm7.append(d.type[i])
        libsvm7.append(str(d['actor_libsvm'][i]))
        libsvm7.append(str(d['summary_libsvm'][i]))
    return libsvm7
libsvm7 = getlibsvm(df7)

# 解决内存问题，选取小数据集做的
libsvm71 = libsvm7[0:40384]

libsvm71 = pd.DataFrame(np.array(libsvm71).reshape(5048,8))
libsvm71.columns = ['rate','user','movie','movie_bias','user_bias','type','actor','summary']

# 去除type两端的双引号
l = list()
for i in range(len(libsvm71)):
    l.append(eval(libsvm71['type'][i]))
libsvm71['type'] = l

# 去除actor两端的双引号
l = list()
for i in range(len(libsvm71)):
    l.append(eval(libsvm71['actor'][i]))
libsvm71['actor'] = l

# 去除summary两端的双引号
l = list()
for i in range(len(libsvm71)):
    l.append(eval(libsvm71['summary'][i]))
libsvm71['summary'] = l

libsvm71.to_csv('libsvm71.csv',header=True,index=None)



# 划分训练集合测试集
from sklearn.model_selection import train_test_split
train71, test71 = train_test_split(libsvm71, test_size=0.1, random_state=10)
train72, test72 = train_test_split(libsvm71, test_size=0.1, random_state=11)
train71 = train71.reset_index()
test71 = test71.reset_index()
train72 = train72.reset_index()
test72 = test72.reset_index()

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


# train71,test71
fw = open("train71.txt", 'w') 
for i in range(len(train71)):
    fw.write(train71['rate'][i]+' '+train71['user'][i]+' '+train71['movie'][i]
        +' '+train71['movie_bias'][i]+' '+train71['user_bias'][i]+transpose(train71['type'][i])
        +transpose1(train71['actor'][i])+transpose1(train71['summary'][i]))
    fw.write('\n')
fw.close()

fw = open("test71.txt", 'w') 
for i in range(len(test71)):
     fw.write(test71['rate'][i]+' '+test71['user'][i]+' '+test71['movie'][i]
        +' '+test71['movie_bias'][i]+' '+test71['user_bias'][i]+transpose(test71['type'][i])
        +transpose1(test71['actor'][i])+transpose1(test71['summary'][i]))
     fw.write('\n')
fw.close()

# train72,test72
fw = open("train72.txt", 'w') 
for i in range(len(train72)):
     fw.write(train72['rate'][i]+' '+train72['user'][i]+' '+train72['movie'][i]
        +' '+train72['movie_bias'][i]+' '+train72['user_bias'][i]+transpose(train72['type'][i])
        +transpose1(train72['actor'][i])+transpose1(train72['summary'][i]))
     fw.write('\n')
fw.close()

fw = open("test72.txt", 'w') 
for i in range(len(test72)):
     fw.write(test72['rate'][i]+' '+test72['user'][i]+' '+test72['movie'][i]
        +' '+test72['movie_bias'][i]+' '+test72['user_bias'][i]+transpose(test72['type'][i])
        +transpose1(test72['actor'][i])+transpose1(test72['summary'][i]))
     fw.write('\n')
fw.close()




















train62 = pd.read_table('train62.txt',header=None)
train, test = train_test_split(train62, test_size=0.11111111, random_state=10)
train = train.reset_index()
test = test.reset_index()

fw = open("train62_new.txt", 'w') 
for i in range(len(train)):
    fw.write(train[0][i])
    fw.write('\n')
fw.close()

fw = open("valid62_new.txt", 'w') 
for i in range(len(test)):
    fw.write(test[0][i])
    fw.write('\n')
fw.close()