# -*- coding = utf-8 -*-
# @Time : 2023/1/8 16:18
# @Author : liubaoxin
# @File : train_model.py
# @Software : PyCharm
import csv
import torch
from numpy import array
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

data = pd.read_csv("train_last.csv",encoding='UTF-8')
data = data.values.tolist()
articles = list()
labels = list()
print(data[0])
print("开始获得数据")
for d in data:
    articles.append(d[0].split(' '))
    labels.append(d[1])
print("数据完成")
del articles[10000:]
del labels[10000:]
print(len(articles),len(labels))
labels = array(labels)
allwords = set()
print("开始统计")
for sen in articles:
    for word in sen:
        allwords.add(word)
allwords = list(allwords)
print("完成！！！！")
test = pd.read_csv('test_last.csv',encoding='UTF-8')
test = test.values.tolist()
# del test[4000:]
print(test[0:10])
results = list()
vector = list()
for sen in articles:
    temp = list(map(lambda x:sen.count(x),allwords))
    vector.append(temp)

model = MultinomialNB()
# model.cuda()
model.fit(vector,labels)


def Predict(string):
    words = string.split(' ')
    currentVector = array(tuple(map(lambda x:words.count(x),allwords)))
    # currentVector.cuda()
    result = model.predict(currentVector.reshape(1,-1))
    return result
for i in test:
    result = Predict(i[0])
    results.append(result)
print(results[0:10])
test =pd.DataFrame(data = results)
test.to_csv('chuantong_result.csv',encoding='UTF-8',index=None,header=None)

result_index = 1
result_test = []
for label in results:
    x = {}
    x['ID'] = result_index
    x['Last Label'] = int(label)
    result_test.append(x)
    result_index += 1
with open("chuantong_result_last.csv",'w',newline='',encoding='UTF-8') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(['ID','Last Label'])
    for n in result_test:
        writer.writerow(n.values())


