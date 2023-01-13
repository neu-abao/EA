# -*- coding = utf-8 -*-
# @Time : 2023/1/8 11:19
# @Author : liubaoxin
# @File : data_pre.py
# @Software : PyCharm
import jieba
import pandas  as pd
import csv
import re
data = []
with open('train_data.csv', encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True):
        data.append(row)
data.pop(0)
print(len(data))

data_test = []
with open('test_data_new.csv', encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True):
        data_test.append(row)
data_test.pop(0)
print(len(data_test))

train_sentences = []
train_labels = []
for sen in data:
    temp = sen[1].split(' __eou__ ')
    if len(temp) == len(sen[2]):
        for t in temp:
            train_sentences.append(t)
        for n in sen[2]:
            train_labels.append(n)
# print(sentences[0],labels[0])
# print(len(sentences),len(labels))


#测试集处理
test_sentences = []
test_labels = []
for sen in data_test:
    temp = sen[1].split(' __eou__ ')
    test_sentences.append(temp[-1])
        # test_labels.append(sen[2][-1])
# print(len(test_sentences),len(test_labels))


#创建停用词表
def stopwordslist():
    stopwords = [line.strip() for line in open('newstopwords.txt',encoding='UTF-8').readlines()]
    return stopwords

#对每一行进行中文分词

def seg_depart(sentence):
    # 对文档中的每一行进行中文分词
    # print("正在分词")
    sentence_depart = jieba.cut(sentence.strip())
    # 创建一个停用词列表
    stopwords = stopwordslist()
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

#开始处理训练集
stopwords = stopwordslist()
pre_train_data = []
for sentence in train_sentences:
    pre_train_data.append(seg_depart(sentence))
# print(pre_data[0])

#将数据写入新的CSV文件
data_list = []
for data, label in zip(pre_train_data,train_labels):
    x = {}
    x['text'] = re.sub(' +', ' ', data)
    x['label'] = label
    data_list.append(x)

with open("train_last.csv",'w',newline='',encoding='UTF-8') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(['text', 'label'])
    for n in data_list:
        writer.writerow(n.values())
train_df = pd.read_csv('train.csv',encoding='UTF-8')
# print(train_df.shape)
# print(train_df.head())


#测试集处理
#将数据写入新的CSV文件
pre_test_data = []
for sentence in test_sentences:
    pre_test_data.append(seg_depart(sentence))
print(pre_test_data[0:10])
data_newtest = []
for data in pre_test_data:
    x = {}
    x['text'] = re.sub(' +',' ',data)
    data_newtest.append(x)
# data_test = []
# for data, label in zip(pre_test_data,test_labels):
#     x = {}
#     x['text'] = re.sub(' +', ' ', data)
#     x['label'] = label
#     data_test.append(x)
#
# with open("test.csv",'w',newline='',encoding='UTF-8') as f_csv:
#     writer = csv.writer(f_csv)
#     writer.writerow(['text', 'label'])
#     for n in data_test:
#         writer.writerow(n.values())
# test_df = pd.read_csv('test.csv',encoding='UTF-8')
# print(test_df.shape)
# print(test_df.head())
with open("test_last.csv",'w',newline='',encoding='UTF-8') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(['text'])
    for d in data_newtest:
        writer.writerow(d.values())





