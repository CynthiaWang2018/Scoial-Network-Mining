# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:49:58 2018

@author: Administrator
"""

import pandas as pd
import numpy as np

str_dict={
    '剧情':'1999'+':'+'1',
    '喜剧':'2000'+':'+'1',
    '爱情':'2001'+':'+'1',
    '犯罪':'2002'+':'+'1',
    '动作':'2003'+':'+'1',
    '动画':'2004'+':'+'1',
    '惊悚':'2005'+':'+'1',
    '冒险':'2006'+':'+'1',
    '悬疑':'2007'+':'+'1',
    '科幻':'2008'+':'+'1',
    '传记':'2009'+':'+'1',
    '历史':'2010'+':'+'1',
    '战争':'2011'+':'+'1',
    '奇幻':'2012'+':'+'1',
    '家庭':'2013'+':'+'1',
    '恐怖':'2014'+':'+'1',
    '同性':'2015'+':'+'1',
    '音乐':'2016'+':'+'1',
    '歌舞':'2017'+':'+'1',
    '运动':'2018'+':'+'1',
    '西部':'2019'+':'+'1',
    '情色':'2020'+':'+'1',
    '古装':'2021'+':'+'1',
    '儿童':'2022'+':'+'1',
    '灾难':'2023'+':'+'1',
    '武侠':'2024'+':'+'1',
    '黑色电影':'2025'+':'+'1'
  }

# 把type转换成列数
def changetype(x):
    res_list = []
    if x == -1:
        res_list.append(-1)
    else:
        for one_str in x:
            for key in str_dict:
                one_str = one_str.replace(key, str_dict[key])
            res_list.append(one_str)
    return res_list

t1 = []
for i in range(1000):
    t = changetype(l1[i])
    t1.append(t)

movie['type1'] = t1

# merge的时候加入 right_index=True 使索引不变,这样可以按照索引拼接了
df4 = pd.merge(df2,movie[['id','type1']],on='id',how='outer',right_index=True)

# 转化成libsvm格式
libsvm2 = pd.read_csv('libsvm2.csv',header=None)
libsvm4 = libsvm2
libsvm4['type1'] = df4.type1
libsvm4.columns = ['rate','user','movie','movie_bias','user_bias','type1']
libsvm4.to_csv('libsvm4.csv',header=None,index=None)



# 划分训练集合测试集
from sklearn.model_selection import train_test_split
train41, test41, = train_test_split(libsvm4, test_size=0.1, random_state=10)
train42, test42, = train_test_split(libsvm4, test_size=0.1, random_state=11)


'''list的元素中间加上空格'''
# train4_1.iloc[0,5]
# ['2008:1', '2005:1', '2014:1']
# print(" ".join(i for i in train4_1.iloc[0,5]))
# 2008:1 2005:1 2014:1


# 解决类别是-1带来的麻烦
def transpose(list1):
    if -1 in list1:
        l = ''
    else:
        l= ' '+" ".join(i for i in list1)
    return l


# train41,test41
fw = open("train41.txt", 'w') 
for i in range(len(train41)):
        fw.write(train41.iloc[i,0]+' '+train41.iloc[i,1]+' '+train41.iloc[i,2]
        +' '+train41.iloc[i,3]+' '+train41.iloc[i,4]+transpose(train41.iloc[i,5]))
        fw.write('\n')
fw.close()

fw = open("test41.txt", 'w') 
for i in range(len(test41)):
        fw.write(test41.iloc[i,0]+' '+test41.iloc[i,1]+' '+test41.iloc[i,2]
        +' '+test41.iloc[i,3]+' '+test41.iloc[i,4]+transpose(test41.iloc[i,5]))
        fw.write('\n')
fw.close()

# train42,test42
fw = open("train42.txt", 'w') 
for i in range(len(train42)):
        fw.write(train42.iloc[i,0]+' '+train42.iloc[i,1]+' '+train42.iloc[i,2]
        +' '+train42.iloc[i,3]+' '+train42.iloc[i,4]+transpose(train42.iloc[i,5]))
        fw.write('\n')
fw.close()

fw = open("test42.txt", 'w') 
for i in range(len(test42)):
        fw.write(test42.iloc[i,0]+' '+test42.iloc[i,1]+' '+test42.iloc[i,2]
        +' '+test42.iloc[i,3]+' '+test42.iloc[i,4]+transpose(test42.iloc[i,5]))
        fw.write('\n')
fw.close()
    
    
df4.to_csv('df4.csv',header=None,index=None)
    
    
    
    
    
    