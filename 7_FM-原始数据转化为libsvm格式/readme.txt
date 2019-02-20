libsvm 将原始数据转化为libsvm格式，共900多行代码，处理了电影ID，用户ID，评分，电影偏置项，用户偏置项，电影类型，Node2vec处理电影导演演员节点信息，Doc2vec处理电影简介等 7 个特征工程。

movie_vec1.csv 其中summary：doc2vec,是都是353的向量；
                   type_vec:用了tf-idf做了，向量；
                   actor_voc，direct_vec：都是one-hot了