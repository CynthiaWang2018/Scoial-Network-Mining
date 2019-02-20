# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:44:04 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
movie = pd.read_csv('movie.csv')
user = pd.read_csv('user.csv')

score = pd.read_csv('score.csv',header=None)
score_1000 = pd.read_csv('score_1000.csv',dtype=int)
score = np.matrix(score)

# 发现一共有40,381个有评分，共1,000,000。有4%的有数据
# np.sum(score[i,j]>0 for i in range(1000) for j in range(1000))
# 共有997个电影有评分
# len(set(score_1000.movie_id))

# 将电影ID转化为1,2,...,997,存在list2中
d1 = score_1000
list1=sorted(list(set(d1.movie_id)))
list2=[]
def transform(d1,list1):
    for i in range(len(d1)):
        for j in range(len(list1)):
            if d1.movie_id[i] == list1[j]:
                list2.append(j+1)
    return list2
list2 = transform(d1,list1)
list2 = pd.read_csv('list2.csv',header=None,dtype=int)
list2.columns = ['order']

list1 = list(list2.order)



'''类似table的函数'''
# list2.count(1)，list2.count(2)，list2.count(997)经过测试发现没有错误
# Series.value_counts()
# df.apply(pd.value_counts) 
# 第一次生成
# 生成SVM格式
libsvm = np.zeros((len(list2),1998),dtype=int)
def getsvm(d1,list2):
    n = len(list2)
    libsvm[:,0] = d1.rates[:n]
    for i in range(n):
        libsvm[i,d1.user_id[i]] = 1
        libsvm[i,list2[i]+1000] = 1 
    return libsvm

matrix1 = getsvm(d1,list2)

np.savetxt('list2.csv', list2, delimiter = ',')
np.savetxt('matrix1.csv', matrix1, delimiter = ',')          
            
    
# 转化成libsvm格式
libsvm = list()
def getlibsvm(d1,list2):
    for i in range(len(d1)):
        libsvm.append(d1.rates[i])
        libsvm.append(str(d1.user_id[i]-1)+':'+str(1))
        libsvm.append(str(list2.order[i]+999)+':'+str(1))
    return libsvm
list3 = getlibsvm(d1,list2)

libsvm = pd.DataFrame(np.array(list3).reshape(len(d1),3))
libsvm.columns = ['rate','user','movie']
libsvm.to_csv('libsvm.csv',header=None,index=None)


'''DataFrame的切片操作iloc'''
# data.iloc[1:3,1:3]
# df.loc[index, column_name],选取指定行和列的数据
# df.loc[0:2, ['name','age']]
# df.ix[0,[1,2]]		#第0行，第1列和第2列的数据

# 分成训练集和测试集
# 取得训练集和测试集
from sklearn.model_selection import train_test_split
train1, test1, = train_test_split(libsvm, test_size=0.1, random_state=5)
train2, test2, = train_test_split(libsvm, test_size=0.1, random_state=1)
train3, test3, = train_test_split(libsvm, test_size=0.1, random_state=2)
train4, test4, = train_test_split(libsvm, test_size=0.1, random_state=3)
train5, test5, = train_test_split(libsvm, test_size=0.1, random_state=4)

# train.to_csv('train.csv',header=None,index=None)
# test.to_csv('test.csv',header=None,index=None)

# train.to_excel('train.xlsx',header=None,index=None)
# test.to_excel('tset.xlsx',header=None,index=None)

# 求出矩阵的秩，当做K值，为849
np.linalg.matrix_rank(score)


# 谢谢百度，啊啊啊，还是自己撸代码好呀
# train11,test11
fw = open("train11.txt", 'w') 
for i in range(len(train1)):
        fw.write(train1.iloc[i,0]+' '+train1.iloc[i,1]+' '+train1.iloc[i,2])
        fw.write('\n')
fw.close()

fw = open("test11.txt", 'w') 
for i in range(len(test1)):
        fw.write(test1.iloc[i,0]+' '+test1.iloc[i,1]+' '+test1.iloc[i,2])
        fw.write('\n')
fw.close()

# train12,test12
fw = open("train12.txt", 'w') 
for i in range(len(train2)):
        fw.write(train2.iloc[i,0]+' '+train2.iloc[i,1]+' '+train2.iloc[i,2])
        fw.write('\n')
fw.close()

fw = open("test12.txt", 'w') 
for i in range(len(test2)):
        fw.write(test2.iloc[i,0]+' '+test2.iloc[i,1]+' '+test2.iloc[i,2])
        fw.write('\n')
fw.close()

# train13,test13
fw = open("train13.txt", 'w') 
for i in range(len(train3)):
        fw.write(train3.iloc[i,0]+' '+train3.iloc[i,1]+' '+train3.iloc[i,2])
        fw.write('\n')
fw.close()

fw = open("test13.txt", 'w') 
for i in range(len(test3)):
        fw.write(test3.iloc[i,0]+' '+test3.iloc[i,1]+' '+test3.iloc[i,2])
        fw.write('\n')
fw.close()

# train14,test14
fw = open("train14.txt", 'w') 
for i in range(len(train4)):
        fw.write(train4.iloc[i,0]+' '+train4.iloc[i,1]+' '+train4.iloc[i,2])
        fw.write('\n')
fw.close()

fw = open("test14.txt", 'w') 
for i in range(len(test4)):
        fw.write(test4.iloc[i,0]+' '+test4.iloc[i,1]+' '+test4.iloc[i,2])
        fw.write('\n')
fw.close()

# train15,test15
fw = open("train15.txt", 'w') 
for i in range(len(train5)):
        fw.write(train5.iloc[i,0]+' '+train5.iloc[i,1]+' '+train5.iloc[i,2])
        fw.write('\n')
fw.close()

fw = open("test15.txt", 'w') 
for i in range(len(test5)):
        fw.write(test5.iloc[i,0]+' '+test5.iloc[i,1]+' '+test5.iloc[i,2])
        fw.write('\n')
fw.close()





'''特征工程二'''
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





'''特征工程三'''
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





'''特征工程四'''
# 处理电影剧情信息
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





'''特征工程五'''
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






'''特征工程六'''
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





'''特征工程七'''
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