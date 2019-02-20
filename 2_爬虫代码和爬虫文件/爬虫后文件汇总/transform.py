# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 01:07:21 2018

@author: Administrator
"""

import pandas as pd
import numpy as np

d1 = pd.read_csv('movie_d_0_100.csv')
d2 = pd.read_csv('movie_d_100_200.csv')
d3 = pd.read_csv('movie_d_200_300.csv')
d4 = pd.read_csv('movie_d_300_400.csv')
d5 = pd.read_csv('movie_d_400_500.csv')
d51 = pd.read_csv('movie_d_460_500.csv')
d6 = pd.read_csv('movie_d_500_600.csv')
d7 = pd.read_csv('movie_d_600_700.csv')
d8 = pd.read_csv('movie_d_700_800.csv')
d9 = pd.read_csv('movie_d_800_900.csv')
d10 = pd.read_csv('movie_d_900_1000.csv')
d11 = pd.read_csv('movie_d_1000_1100.csv')
d12 = pd.read_csv('movie_d_1100_1200.csv')
d13 = pd.read_csv('movie_d_1200_1300.csv')
d14 = pd.read_csv('movie_d_1300_1400.csv')
d15 = pd.read_csv('movie_d_1400_1500.csv')
d16 = pd.read_csv('movie_d_1500_1600.csv')
d17 = pd.read_csv('movie_d_1600_1700.csv')

d31 = pd.read_csv('movie3000_3100.csv')
d32 = pd.read_csv('movie3100_3200.csv')
d33 = pd.read_csv('movie3200_3300.csv')
d34 = pd.read_csv('movie3300_3400.csv')
d35 = pd.read_csv('movie3400_3500.csv')

alldata = pd.read_csv('all_rate.csv')
movie = pd.read_csv('movie.csv')
score = pd.read_csv('score_1000.csv')

frames = [d1,d2,d3,d4,d5,d51,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,
          d17,d31,d32,d33,d34,d35]
d = pd.concat(frames)
d_new = d[d['score'].notnull()]
d_new['id'] = d_new['id'].astype('int')

df = pd.merge(alldata, d_new, how='inner', left_on='movie_id', right_on='id')
df = df.drop_duplicates(keep='first')

df.to_csv('spider.csv')


# 处理只含三列的数据集
data1 = df[['user_id','movie_id','score_x']]
data1.columns = ['user_id','movie_id','rate']
data2 = score[['user_id','movie_id','rates']]
data2.columns = ['user_id','movie_id','rate']

data = pd.concat([data1,data2]).drop_duplicates(keep='first')
len(set(data['movie_id']))
len(set(data['user_id']))

data.to_csv('usermovierate.csv')