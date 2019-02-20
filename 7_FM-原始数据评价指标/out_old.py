# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 19:40:37 2018

@author: Administrator
"""

import pandas as pd
import numpy as np

fm1 = pd.read_table('f1.txt',header=None)
fm1_test = pd.read_table('test11.txt',header=None)
fm2 = pd.read_table('f2.txt',header=None)
fm2_test = pd.read_table('test22.txt',header=None)
fm3 = pd.read_table('f3.txt',header=None)
fm3_test = pd.read_table('test31.txt',header=None)
fm4 = pd.read_table('f4.txt',header=None)
fm4_test = pd.read_table('test42.txt',header=None)
fm5 = pd.read_table('f5.txt',header=None)
fm5_test = pd.read_table('test52.txt',header=None)
fm6 = pd.read_table('f6.txt',header=None)
fm6_test = pd.read_table('test62.txt',header=None)


# 提取出训练集中有效的评分、用户ID和电影ID信息
def getid(fm1_test):
    rate = list()
    user = list()
    movie = list()
    for i in range(len(fm1_test)):
        l1 = fm1_test[0][0].split(' ')[0]
        l2 = fm1_test[0][i].split(' ')[1].split(':')[0]
        l3 = fm1_test[0][i].split(' ')[2].split(':')[0]
        rate.append(int(l1))
        user.append(int(l2))
        movie.append(int(l3))
    return rate,user,movie

rate1, user1, movie1 = getid(fm1_test)
rate2, user2, movie2 = getid(fm2_test)
rate3, user3, movie3 = getid(fm3_test)
rate4, user4, movie4 = getid(fm4_test)

def getid(fm5_test):
    rate = list()
    user = list()
    movie = list()
    for i in range(len(fm5_test)):
        l1 = int(float(fm5_test[0][i].split(' ')[0]))
        l2 = int(fm5_test[0][i].split(' ')[1].split(':')[0])
        l3 = int(fm5_test[0][i].split(' ')[2].split(':')[0])
        rate.append(int(l1))
        user.append(int(l2))
        movie.append(int(l3))
    return rate,user,movie
rate5, user5, movie5 = getid(fm5_test)
rate6, user6, movie6 = getid(fm6_test)



# RMSE由libfm软件直接算出
# 求出MAE
def mae(fm1,rate1):
    s = 0
    for i in range(len(fm1)):
        l = abs(fm1[0][i]-rate1[i])
        s = l+s
    mae = s/len(fm1)
    return mae

mae1 = mae(fm1,rate1) # 0.5966
mae2 = mae(fm2,rate2) # 0.5887
mae3 = mae(fm3,rate3) # 0.5881
mae4 = mae(fm4,rate4) # 0.5828
mae5 = mae(fm5,rate5) # 0.5902
mae6 = mae(fm6,rate6) # 0.58825


# 求出FCP
# 将训练集和输出信息合并
fm1_test = pd.DataFrame({'rate':rate1,'user':user1,'movie':movie1})
df1 = fm1_test.join(fm1)
df1.columns = ['rate', 'user', 'movie', 'rate_fm']

fm2_test = pd.DataFrame({'rate':rate2,'user':user2,'movie':movie2})
df2 = fm2_test.join(fm2)
df2.columns = ['rate', 'user', 'movie', 'rate_fm']

fm3_test = pd.DataFrame({'rate':rate3,'user':user3,'movie':movie3})
df3 = fm3_test.join(fm3)
df3.columns = ['rate', 'user', 'movie', 'rate_fm']

fm4_test = pd.DataFrame({'rate':rate4,'user':user4,'movie':movie4})
df4 = fm4_test.join(fm4)
df4.columns = ['rate', 'user', 'movie', 'rate_fm']

fm5_test = pd.DataFrame({'rate':rate5,'user':user5,'movie':movie5})
df5 = fm5_test.join(fm5)
df5.columns = ['rate', 'user', 'movie', 'rate_fm']

fm6_test = pd.DataFrame({'rate':rate6,'user':user6,'movie':movie6})
df6 = fm6_test.join(fm6)
df6.columns = ['rate', 'user', 'movie', 'rate_fm']


# 对用户遍历算nc和nc
def ncnd(df1):
    nc = []
    nd = []
    for i in range(1000):
        # 对每个用户循环
        d = df1[df1['user']==i+1].reset_index()
        correct = 0
        defalse = 0
        other = 0
        for i in range(len(d)):
            for j in range(len(d)):
                if ((d['rate'][i] > d['rate'][j]) & (d['rate_fm'][i] > d['rate_fm'][j])):
                    correct = correct + 1
                elif ((d['rate'][i] > d['rate'][j]) & (d['rate_fm'][i] < d['rate_fm'][j])):
                    defalse = defalse + 1
                else:
                    other = other +1
        nc.append(correct)
        nd.append(defalse)
    return nc, nd

nc1, nd1 = ncnd(df1)
print(sum(nc1)/(sum(nc1)+sum(nd1))) # 0.7661
nc2, nd2 = ncnd(df2)
print(sum(nc2)/(sum(nc2)+sum(nd2))) # 0.7690
nc3, nd3 = ncnd(df3)
print(sum(nc3)/(sum(nc3)+sum(nd3))) # 0.7676
nc4, nd4 = ncnd(df4)
print(sum(nc4)/(sum(nc4)+sum(nd4))) # 0.7713
nc5, nd5 = ncnd(df5)
print(sum(nc5)/(sum(nc5)+sum(nd5))) # 0.7711
nc6, nd6 = ncnd(df6)
print(sum(nc6)/(sum(nc6)+sum(nd6))) # 0.7726