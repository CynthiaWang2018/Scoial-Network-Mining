# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 23:00:39 2018

@author: Administrator
"""

import pandas as pd
import numpy as np

# 储存着0-2025列的信息
df4 = pd.read_csv('df4.csv',header=None)
df4.columns = ['user_id','rate','movie_num','movie_bias','movie_id','user_bias','type']

'''0-999是用户ID，1000-1996是电影ID，1997是电影平均评分，1998是用户平均评分，1999-2025是电影类型'''
'''2026-2153是电影简述的文本信息'''

# summary和df4按照movie_num的主键进行合并
summary = pd.read_csv('node2vec.csv',header=None)
summary['movie_num'] = [int(i+1) for i in summary[0]]
# 去掉逗号
for j in range(1,128):
    summary[j] = [float(summary[j][i].replace(',','')) for i in range(1000)]
del summary[0]

df5 = pd.merge(df4, summary, on='movie_num', how='left')
df5 = df5.dropna(axis=0) 


# 转化成libsvm格式
libsvm5 = []
def getlibsvm(d):
    for i in range(len(d)):
        libsvm5.append(d.rate[i])
        libsvm5.append(str(int(d.user_id[i])-1)+':'+str(1))
        libsvm5.append(str(int(d.movie_num[i])+999)+':'+str(1))
        libsvm5.append(str(1997)+':'+str(d.movie_bias[i]/2))
        libsvm5.append(str(1998)+':'+str(d.user_bias[i]))
        libsvm5.append(d.type[i])
    return libsvm5
libsvm5 = getlibsvm(df5)

# 这里运行慢
for j in range(1,128):
    df5[j] = [str(2025+j)+':'+str(df5[j][i]) for i in range(len(df5))]
df5[128] = [str(2025+128)+':'+str(df5[128][i]) for i in range(len(df5))]
text = df5[[i+1 for i in range(128)]]

libsvm = pd.DataFrame(np.array(libsvm5).reshape(len(df5),6))
libsvm.columns = ['rate','user','movie','movie_bias','user_bias','type']
libsvm = libsvm.join(text)
libsvm.to_csv('libsvm5.csv',header=True,index=None)


libsvm5 = pd.read_csv('libsvm5.csv')
# 去除type两端的双引号
l = list()
for i in range(len(libsvm5)):
    l.append(eval(libsvm5['type'][i]))
libsvm5['type'] = l


# 划分训练集合测试集
from sklearn.model_selection import train_test_split
train51, test51 = train_test_split(libsvm5, test_size=0.1, random_state=10)
train52, test52 = train_test_split(libsvm5, test_size=0.1, random_state=11)
train51 = train51.reset_index()
test51 = test51.reset_index()
train52 = train52.reset_index()
test52 = test52.reset_index()

# 解决类别是-1带来的麻烦
def transpose(list1):
    if -1 in list1:
        l = ''
    else:
        l= ' '+" ".join(i for i in list1)
    return l


# train51,test51
fw = open("train51.txt", 'w') 
for i in range(len(train51)):
    fw.write(str(train51['rate'][i])+' '+train51['user'][i]+' '+train51['movie'][i]
        +' '+train51['movie_bias'][i]+' '+train51['user_bias'][i]+transpose(train51['type'][i])
        +' '+train51['1'][i]+' '+train51['2'][i]+' '+train51['3'][i]+' '+train51['4'][i]
        +' '+train51['5'][i]+' '+train51['6'][i]+' '+train51['7'][i]+' '+train51['8'][i]
        +' '+train51['9'][i]+' '+train51['10'][i]+' '+train51['11'][i]+' '+train51['12'][i]
        +' '+train51['13'][i]+' '+train51['14'][i]+' '+train51['15'][i]+' '+train51['16'][i]
        +' '+train51['17'][i]+' '+train51['18'][i]+' '+train51['19'][i]+' '+train51['20'][i]
        +' '+train51['21'][i]+' '+train51['22'][i]+' '+train51['23'][i]+' '+train51['24'][i]
        +' '+train51['25'][i]+' '+train51['26'][i]+' '+train51['27'][i]+' '+train51['28'][i]
        +' '+train51['29'][i]+' '+train51['30'][i]+' '+train51['31'][i]+' '+train51['32'][i]
        +' '+train51['33'][i]+' '+train51['34'][i]+' '+train51['35'][i]+' '+train51['36'][i]
        +' '+train51['37'][i]+' '+train51['38'][i]+' '+train51['39'][i]+' '+train51['40'][i]
        +' '+train51['41'][i]+' '+train51['42'][i]+' '+train51['43'][i]+' '+train51['44'][i]
        +' '+train51['45'][i]+' '+train51['46'][i]+' '+train51['47'][i]+' '+train51['48'][i]
        +' '+train51['49'][i]+' '+train51['50'][i]+' '+train51['51'][i]+' '+train51['52'][i]
        +' '+train51['53'][i]+' '+train51['54'][i]+' '+train51['55'][i]+' '+train51['56'][i]
        +' '+train51['57'][i]+' '+train51['58'][i]+' '+train51['59'][i]+' '+train51['60'][i]
        +' '+train51['61'][i]+' '+train51['62'][i]+' '+train51['63'][i]+' '+train51['64'][i]
        +' '+train51['65'][i]+' '+train51['66'][i]+' '+train51['67'][i]+' '+train51['68'][i]
        +' '+train51['69'][i]+' '+train51['70'][i]+' '+train51['71'][i]+' '+train51['72'][i]
        +' '+train51['73'][i]+' '+train51['74'][i]+' '+train51['75'][i]+' '+train51['76'][i]
        +' '+train51['77'][i]+' '+train51['78'][i]+' '+train51['79'][i]+' '+train51['80'][i]
        +' '+train51['81'][i]+' '+train51['82'][i]+' '+train51['83'][i]+' '+train51['84'][i]
        +' '+train51['85'][i]+' '+train51['86'][i]+' '+train51['87'][i]+' '+train51['88'][i]
        +' '+train51['89'][i]+' '+train51['90'][i]+' '+train51['91'][i]+' '+train51['92'][i]
        +' '+train51['93'][i]+' '+train51['94'][i]+' '+train51['95'][i]+' '+train51['96'][i]
        +' '+train51['97'][i]+' '+train51['98'][i]+' '+train51['99'][i]+' '+train51['100'][i]
        +' '+train51['101'][i]+' '+train51['102'][i]+' '+train51['103'][i]+' '+train51['104'][i]
        +' '+train51['105'][i]+' '+train51['106'][i]+' '+train51['107'][i]+' '+train51['108'][i]
        +' '+train51['109'][i]+' '+train51['110'][i]+' '+train51['111'][i]+' '+train51['112'][i]
        +' '+train51['113'][i]+' '+train51['114'][i]+' '+train51['115'][i]+' '+train51['116'][i]
        +' '+train51['117'][i]+' '+train51['118'][i]+' '+train51['119'][i]+' '+train51['120'][i]
        +' '+train51['121'][i]+' '+train51['122'][i]+' '+train51['123'][i]+' '+train51['124'][i]
        +' '+train51['125'][i]+' '+train51['126'][i]+' '+train51['127'][i]+' '+train51['128'][i])
    fw.write('\n')
fw.close()


fw = open("test51.txt", 'w') 
for i in range(len(test51)):
    fw.write(str(test51['rate'][i])+' '+test51['user'][i]+' '+test51['movie'][i]
        +' '+test51['movie_bias'][i]+' '+test51['user_bias'][i]+transpose(test51['type'][i])
        +' '+test51['1'][i]+' '+test51['2'][i]+' '+test51['3'][i]+' '+test51['4'][i]
        +' '+test51['5'][i]+' '+test51['6'][i]+' '+test51['7'][i]+' '+test51['8'][i]
        +' '+test51['9'][i]+' '+test51['10'][i]+' '+test51['11'][i]+' '+test51['12'][i]
        +' '+test51['13'][i]+' '+test51['14'][i]+' '+test51['15'][i]+' '+test51['16'][i]
        +' '+test51['17'][i]+' '+test51['18'][i]+' '+test51['19'][i]+' '+test51['20'][i]
        +' '+test51['21'][i]+' '+test51['22'][i]+' '+test51['23'][i]+' '+test51['24'][i]
        +' '+test51['25'][i]+' '+test51['26'][i]+' '+test51['27'][i]+' '+test51['28'][i]
        +' '+test51['29'][i]+' '+test51['30'][i]+' '+test51['31'][i]+' '+test51['32'][i]
        +' '+test51['33'][i]+' '+test51['34'][i]+' '+test51['35'][i]+' '+test51['36'][i]
        +' '+test51['37'][i]+' '+test51['38'][i]+' '+test51['39'][i]+' '+test51['40'][i]
        +' '+test51['41'][i]+' '+test51['42'][i]+' '+test51['43'][i]+' '+test51['44'][i]
        +' '+test51['45'][i]+' '+test51['46'][i]+' '+test51['47'][i]+' '+test51['48'][i]
        +' '+test51['49'][i]+' '+test51['50'][i]+' '+test51['51'][i]+' '+test51['52'][i]
        +' '+test51['53'][i]+' '+test51['54'][i]+' '+test51['55'][i]+' '+test51['56'][i]
        +' '+test51['57'][i]+' '+test51['58'][i]+' '+test51['59'][i]+' '+test51['60'][i]
        +' '+test51['61'][i]+' '+test51['62'][i]+' '+test51['63'][i]+' '+test51['64'][i]
        +' '+test51['65'][i]+' '+test51['66'][i]+' '+test51['67'][i]+' '+test51['68'][i]
        +' '+test51['69'][i]+' '+test51['70'][i]+' '+test51['71'][i]+' '+test51['72'][i]
        +' '+test51['73'][i]+' '+test51['74'][i]+' '+test51['75'][i]+' '+test51['76'][i]
        +' '+test51['77'][i]+' '+test51['78'][i]+' '+test51['79'][i]+' '+test51['80'][i]
        +' '+test51['81'][i]+' '+test51['82'][i]+' '+test51['83'][i]+' '+test51['84'][i]
        +' '+test51['85'][i]+' '+test51['86'][i]+' '+test51['87'][i]+' '+test51['88'][i]
        +' '+test51['89'][i]+' '+test51['90'][i]+' '+test51['91'][i]+' '+test51['92'][i]
        +' '+test51['93'][i]+' '+test51['94'][i]+' '+test51['95'][i]+' '+test51['96'][i]
        +' '+test51['97'][i]+' '+test51['98'][i]+' '+test51['99'][i]+' '+test51['100'][i]
        +' '+test51['101'][i]+' '+test51['102'][i]+' '+test51['103'][i]+' '+test51['104'][i]
        +' '+test51['105'][i]+' '+test51['106'][i]+' '+test51['107'][i]+' '+test51['108'][i]
        +' '+test51['109'][i]+' '+test51['110'][i]+' '+test51['111'][i]+' '+test51['112'][i]
        +' '+test51['113'][i]+' '+test51['114'][i]+' '+test51['115'][i]+' '+test51['116'][i]
        +' '+test51['117'][i]+' '+test51['118'][i]+' '+test51['119'][i]+' '+test51['120'][i]
        +' '+test51['121'][i]+' '+test51['122'][i]+' '+test51['123'][i]+' '+test51['124'][i]
        +' '+test51['125'][i]+' '+test51['126'][i]+' '+test51['127'][i]+' '+test51['128'][i])
    fw.write('\n')
fw.close()


fw = open("train52.txt", 'w') 
for i in range(len(train52)):
    fw.write(str(train52['rate'][i])+' '+train52['user'][i]+' '+train52['movie'][i]
        +' '+train52['movie_bias'][i]+' '+train52['user_bias'][i]+transpose(train52['type'][i])
        +' '+train52['1'][i]+' '+train52['2'][i]+' '+train52['3'][i]+' '+train52['4'][i]
        +' '+train52['5'][i]+' '+train52['6'][i]+' '+train52['7'][i]+' '+train52['8'][i]
        +' '+train52['9'][i]+' '+train52['10'][i]+' '+train52['11'][i]+' '+train52['12'][i]
        +' '+train52['13'][i]+' '+train52['14'][i]+' '+train52['15'][i]+' '+train52['16'][i]
        +' '+train52['17'][i]+' '+train52['18'][i]+' '+train52['19'][i]+' '+train52['20'][i]
        +' '+train52['21'][i]+' '+train52['22'][i]+' '+train52['23'][i]+' '+train52['24'][i]
        +' '+train52['25'][i]+' '+train52['26'][i]+' '+train52['27'][i]+' '+train52['28'][i]
        +' '+train52['29'][i]+' '+train52['30'][i]+' '+train52['31'][i]+' '+train52['32'][i]
        +' '+train52['33'][i]+' '+train52['34'][i]+' '+train52['35'][i]+' '+train52['36'][i]
        +' '+train52['37'][i]+' '+train52['38'][i]+' '+train52['39'][i]+' '+train52['40'][i]
        +' '+train52['41'][i]+' '+train52['42'][i]+' '+train52['43'][i]+' '+train52['44'][i]
        +' '+train52['45'][i]+' '+train52['46'][i]+' '+train52['47'][i]+' '+train52['48'][i]
        +' '+train52['49'][i]+' '+train52['50'][i]+' '+train52['51'][i]+' '+train52['52'][i]
        +' '+train52['53'][i]+' '+train52['54'][i]+' '+train52['55'][i]+' '+train52['56'][i]
        +' '+train52['57'][i]+' '+train52['58'][i]+' '+train52['59'][i]+' '+train52['60'][i]
        +' '+train52['61'][i]+' '+train52['62'][i]+' '+train52['63'][i]+' '+train52['64'][i]
        +' '+train52['65'][i]+' '+train52['66'][i]+' '+train52['67'][i]+' '+train52['68'][i]
        +' '+train52['69'][i]+' '+train52['70'][i]+' '+train52['71'][i]+' '+train52['72'][i]
        +' '+train52['73'][i]+' '+train52['74'][i]+' '+train52['75'][i]+' '+train52['76'][i]
        +' '+train52['77'][i]+' '+train52['78'][i]+' '+train52['79'][i]+' '+train52['80'][i]
        +' '+train52['81'][i]+' '+train52['82'][i]+' '+train52['83'][i]+' '+train52['84'][i]
        +' '+train52['85'][i]+' '+train52['86'][i]+' '+train52['87'][i]+' '+train52['88'][i]
        +' '+train52['89'][i]+' '+train52['90'][i]+' '+train52['91'][i]+' '+train52['92'][i]
        +' '+train52['93'][i]+' '+train52['94'][i]+' '+train52['95'][i]+' '+train52['96'][i]
        +' '+train52['97'][i]+' '+train52['98'][i]+' '+train52['99'][i]+' '+train52['100'][i]
        +' '+train52['101'][i]+' '+train52['102'][i]+' '+train52['103'][i]+' '+train52['104'][i]
        +' '+train52['105'][i]+' '+train52['106'][i]+' '+train52['107'][i]+' '+train52['108'][i]
        +' '+train52['109'][i]+' '+train52['110'][i]+' '+train52['111'][i]+' '+train52['112'][i]
        +' '+train52['113'][i]+' '+train52['114'][i]+' '+train52['115'][i]+' '+train52['116'][i]
        +' '+train52['117'][i]+' '+train52['118'][i]+' '+train52['119'][i]+' '+train52['120'][i]
        +' '+train52['121'][i]+' '+train52['122'][i]+' '+train52['123'][i]+' '+train52['124'][i]
        +' '+train52['125'][i]+' '+train52['126'][i]+' '+train52['127'][i]+' '+train52['128'][i])
    fw.write('\n')
fw.close()


fw = open("test52.txt", 'w') 
for i in range(len(test52)):
    fw.write(str(test52['rate'][i])+' '+test52['user'][i]+' '+test52['movie'][i]
        +' '+test52['movie_bias'][i]+' '+test52['user_bias'][i]+transpose(test52['type'][i])
        +' '+test52['1'][i]+' '+test52['2'][i]+' '+test52['3'][i]+' '+test52['4'][i]
        +' '+test52['5'][i]+' '+test52['6'][i]+' '+test52['7'][i]+' '+test52['8'][i]
        +' '+test52['9'][i]+' '+test52['10'][i]+' '+test52['11'][i]+' '+test52['12'][i]
        +' '+test52['13'][i]+' '+test52['14'][i]+' '+test52['15'][i]+' '+test52['16'][i]
        +' '+test52['17'][i]+' '+test52['18'][i]+' '+test52['19'][i]+' '+test52['20'][i]
        +' '+test52['21'][i]+' '+test52['22'][i]+' '+test52['23'][i]+' '+test52['24'][i]
        +' '+test52['25'][i]+' '+test52['26'][i]+' '+test52['27'][i]+' '+test52['28'][i]
        +' '+test52['29'][i]+' '+test52['30'][i]+' '+test52['31'][i]+' '+test52['32'][i]
        +' '+test52['33'][i]+' '+test52['34'][i]+' '+test52['35'][i]+' '+test52['36'][i]
        +' '+test52['37'][i]+' '+test52['38'][i]+' '+test52['39'][i]+' '+test52['40'][i]
        +' '+test52['41'][i]+' '+test52['42'][i]+' '+test52['43'][i]+' '+test52['44'][i]
        +' '+test52['45'][i]+' '+test52['46'][i]+' '+test52['47'][i]+' '+test52['48'][i]
        +' '+test52['49'][i]+' '+test52['50'][i]+' '+test52['51'][i]+' '+test52['52'][i]
        +' '+test52['53'][i]+' '+test52['54'][i]+' '+test52['55'][i]+' '+test52['56'][i]
        +' '+test52['57'][i]+' '+test52['58'][i]+' '+test52['59'][i]+' '+test52['60'][i]
        +' '+test52['61'][i]+' '+test52['62'][i]+' '+test52['63'][i]+' '+test52['64'][i]
        +' '+test52['65'][i]+' '+test52['66'][i]+' '+test52['67'][i]+' '+test52['68'][i]
        +' '+test52['69'][i]+' '+test52['70'][i]+' '+test52['71'][i]+' '+test52['72'][i]
        +' '+test52['73'][i]+' '+test52['74'][i]+' '+test52['75'][i]+' '+test52['76'][i]
        +' '+test52['77'][i]+' '+test52['78'][i]+' '+test52['79'][i]+' '+test52['80'][i]
        +' '+test52['81'][i]+' '+test52['82'][i]+' '+test52['83'][i]+' '+test52['84'][i]
        +' '+test52['85'][i]+' '+test52['86'][i]+' '+test52['87'][i]+' '+test52['88'][i]
        +' '+test52['89'][i]+' '+test52['90'][i]+' '+test52['91'][i]+' '+test52['92'][i]
        +' '+test52['93'][i]+' '+test52['94'][i]+' '+test52['95'][i]+' '+test52['96'][i]
        +' '+test52['97'][i]+' '+test52['98'][i]+' '+test52['99'][i]+' '+test52['100'][i]
        +' '+test52['101'][i]+' '+test52['102'][i]+' '+test52['103'][i]+' '+test52['104'][i]
        +' '+test52['105'][i]+' '+test52['106'][i]+' '+test52['107'][i]+' '+test52['108'][i]
        +' '+test52['109'][i]+' '+test52['110'][i]+' '+test52['111'][i]+' '+test52['112'][i]
        +' '+test52['113'][i]+' '+test52['114'][i]+' '+test52['115'][i]+' '+test52['116'][i]
        +' '+test52['117'][i]+' '+test52['118'][i]+' '+test52['119'][i]+' '+test52['120'][i]
        +' '+test52['121'][i]+' '+test52['122'][i]+' '+test52['123'][i]+' '+test52['124'][i]
        +' '+test52['125'][i]+' '+test52['126'][i]+' '+test52['127'][i]+' '+test52['128'][i])
    fw.write('\n')
fw.close()