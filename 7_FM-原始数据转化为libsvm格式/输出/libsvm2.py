# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 19:54:45 2018

@author: Administrator
"""

# movie.info()
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

# 电影的年份有91种,有nan，90个为1921-2017

# 电影的类型,共有27种
# pd.Series(l2).value_counts()查看各种类型电影的数目
movie['type'] = movie['type'].fillna(-1)
l1 = []
l2 = []
def gettype(movie):
    for i in range(len(movie)):
        if movie.loc[i,'type'] != -1:
            l1.append(movie.loc[i,'type'].split(','))
            l2.extend(movie.loc[i,'type'].split(','))
        else:
            l1.append(-1)
    return l1,l2
l1,l2 = gettype(movie)

# 画图分析电影类型
plt1 = pd.Series(l2).value_counts()[:16]
import seaborn as sns
sns.set_style("whitegrid")
sns.barplot(x=plt1.index,y=plt1.values)



# 转化成libsvm格式
libsvm = list()
def getlibsvm(df1):
    for i in range(40381):
        libsvm.append(int(df1.rates[i]))
        libsvm.append(str(int(df1.user_id[i])-1)+':'+str(1))
        libsvm.append(str(int(df1.order[i])+999)+':'+str(1))
        libsvm.append(str(1997)+':'+str(df1.rate[i]/2))
    return libsvm
libsvm = getlibsvm(df1)

libsvm1 = pd.DataFrame(np.array(libsvm).reshape(40381,4))
libsvm1.columns = ['rate','user','movie','movie_bias']
libsvm1.to_csv('libsvm1.csv',header=None,index=None)



# 划分训练集合测试集
from sklearn.model_selection import train_test_split
train21, test21, = train_test_split(libsvm1, test_size=0.1, random_state=6)
train22, test22, = train_test_split(libsvm1, test_size=0.1, random_state=7)

# train21,test21
fw = open("train21.txt", 'w') 
for i in range(len(train21)):
        fw.write(train21.iloc[i,0]+' '+train21.iloc[i,1]+' '+train21.iloc[i,2]+' '+train21.iloc[i,3])
        fw.write('\n')
fw.close()

fw = open("test21.txt", 'w') 
for i in range(len(test21)):
        fw.write(test21.iloc[i,0]+' '+test21.iloc[i,1]+' '+test21.iloc[i,2]+' '+test21.iloc[i,3])
        fw.write('\n')
fw.close()

# train22,test22
fw = open("train22.txt", 'w') 
for i in range(len(train22)):
        fw.write(train22.iloc[i,0]+' '+train22.iloc[i,1]+' '+train22.iloc[i,2]+' '+train22.iloc[i,3])
        fw.write('\n')
fw.close()

fw = open("test22.txt", 'w') 
for i in range(len(test22)):
        fw.write(test22.iloc[i,0]+' '+test22.iloc[i,1]+' '+test22.iloc[i,2]+' '+test22.iloc[i,3])
        fw.write('\n')
fw.close()




