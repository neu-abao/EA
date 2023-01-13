# -*- coding = utf-8 -*-
# @Time : 2023/1/11 17:44
# @Author : liubaoxin
# @File : bert_train.py
# @Software : PyCharm
#! -*- coding:utf-8 -*-
import tokenizer
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import csv
# 数据预处理
data = []
with open('train_data.csv', encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True):
        data.append(row)
data.pop(0)
train_sentences = []
train_labels = []
for sen in data:
    temp = sen[1].split(' __eou__ ')
    if len(temp) == len(sen[2]):
        for t in temp:
            train_sentences.append(t)
        for n in sen[2]:
            train_labels.append(n)
print(len(train_sentences),len(train_labels))

#测试集数据预处理
data_test = []
with open('test_data_new.csv', encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True):
        data_test.append(row)
data_test.pop(0)
#测试集处理
test_sentences = []
for sen in data_test:
    temp = sen[1].split(' __eou__ ')
    test_sentences.append(temp[-1])
print(len(test_sentences))

#将数据写入新的CSV文件
data_list = []
train_index = 1
for data, label in zip(train_sentences,train_labels):
    x = {}
    x['indexNo'] = train_index
    x['text'] = data
    x['sentiment'] = int(label) -1
    data_list.append(x)
    train_index += 1
with open("./bert_data/train_bert_last.csv",'w',newline='',encoding='UTF-8') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(['indexNo','text', 'sentiment'])
    for n in data_list:
        writer.writerow(n.values())

#测试集
data_test = []
test_index = 1
for data in test_sentences:
    x = {}
    x['indexNo'] = test_index
    x['text'] = data
    # x['sentiment'] = int(label) -1
    data_test.append(x)
    test_index += 1

with open("./bert_data/test_bert_last.csv",'w',newline='',encoding='UTF-8') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(['indexNo','text'])
    for n in data_test:
        writer.writerow(n.values())

train_df = pd.read_csv("./bert_data/train_bert.csv",encoding='UTF-8')
print(train_df['sentiment'].value_counts())

from transformers import BertTokenizer,BertModel

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 8
MAX_LEN = 128

#模型构建
import torch.nn as nn


class MyBert(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese", return_dict=False)
        self.drop_out = nn.Dropout(0.1)
        self.dense = nn.Linear(768, 6)
        torch.nn.init.normal_(self.dense.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        _, out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        out = self.drop_out(out)
        logits = self.dense(out)
        return logits

#中文数据处理
class MyDataset(object):
    def __init__(self, tweet, sentiment):
        self.tweet = tweet
        self.sentiment = sentiment
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item],
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'sentiment': torch.tensor(data["sentiment"], dtype=torch.long),
        }


def process_data(tweet, sentiment, tokenizer, max_len):
    tok_tweet = tokenizer(tweet)
    input_ids_orig = tok_tweet.input_ids[1:-1]
    input_ids = [101] + input_ids_orig + [102]
    token_type_ids = tok_tweet['token_type_ids']
    attention_mask = tok_tweet['attention_mask']

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        #         print('fit:',len(input_ids))
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
    elif padding_length <= 0:  # 截断
        input_ids = [101] + input_ids_orig[:max_len - 2] + [102]
        #         print('cut:',len(input_ids))
        attention_mask = attention_mask[:max_len]
        token_type_ids = token_type_ids[:max_len]

    return {  # 这里返回的是list格式的tensor
        'ids': input_ids,
        'mask': attention_mask,
        'token_type_ids': token_type_ids,
        'orig_tweet': tweet,
        'sentiment': sentiment,
    }

#构建训练函数
from tqdm.autonotebook import tqdm
import utils
from torch.autograd import Variable

loss_func = nn.CrossEntropyLoss()


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):
        #         print(d)
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        sentiment = d["sentiment"]
        orig_tweet = d["orig_tweet"]

        #         print(ids, type(ids), ids.shape)
        #         print(token_type_ids, type(token_type_ids), token_type_ids.shape)
        #         print(mask,type(mask), mask.shape)
        sentiment = sentiment.to(device)
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        model.zero_grad()

        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )
        #         loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
        loss = loss_func(outputs, sentiment)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # torch.softmax或torch.max 参数dim是函数索引的维度0/1，0是每列的最大值，1是每行的最大值
        # 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
        outputs = torch.softmax(outputs, dim=1).cpu().detach().numpy()

        if bi % 120 == 0:
            print('Step:', bi)
            print('train loss: %.4f' % loss.item())
            print('outputs:', outputs)
            pred_y = torch.max(torch.tensor(outputs, dtype=torch.float), dim=1)[1].data.numpy()
            #             pred_y = torch.softmax(outputs, dim=1).cpu().detach().numpy()
            print('pred_y:', pred_y)
            #             print(np.array(sentiment.data)).item()
            accuracy = (sum(pred_y == np.array(sentiment.data.cpu())).item()) / sentiment.size(0)
            print('train accurancy: %.2f' % accuracy)

# #训练流程
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np

df_train = train_df.copy()

train_dataset = MyDataset(
    tweet=df_train.text.values,
    sentiment=df_train.sentiment.values,)

train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    num_workers=0 )

# train_loader = Data.DataLoader(dataset=train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
# device = torch.device("cuda")

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
model = MyBert()
model.to(device)

num_train_steps = int(len(df_train) / TRAIN_BATCH_SIZE * EPOCHS)
param_optimizer = list(model.named_parameters())

# 这里是设置不应该添加正则化项的参数，一般是BN层的可训练参数及卷积层和全连接层的 bias
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]
optimizer = AdamW(optimizer_parameters, lr=3e-5)

# 学习率变动函数，这里使用的是预热学习率
# 在预热期间，学习率从0线性增加到优化器中的初始lr。
# 在预热阶段之后创建一个schedule，使其学习率从优化器中的初始lr线性降低到0
# 这里没使用预热，直接从初始学习率开始下降
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,  # The number of steps for the warmup phase.
    num_training_steps=num_train_steps # The total number of training steps
)

# es = utils.EarlyStopping(patience=2, mode="max")


# 训练EPOCHS个批次
for epoch in range(EPOCHS):
    print('Epoch:',epoch)
    train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)

# 模型保存
torch.save({'state_dict': model.state_dict()}, 'bert_chinese_simple_8epoch_dict.pth.bar') #只保留参数
#
# # # # 读取
model = MyBert()
checkpoint = torch.load('bert_chinese_simple_3epoch_dict.pth.bar')
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
# print(model)

#测试集构建
class MyDatasetTest:
    def __init__(self, tweet):
        self.tweet = tweet
        #         self.sentiment = sentiment
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data_test(
            self.tweet[item],
            #             self.sentiment[item],
            self.tokenizer,
            self.max_len
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"]
            #             'sentiment': torch.tensor(data["sentiment"], dtype=torch.long),
        }


def process_data_test(tweet, tokenizer, max_len):
    tok_tweet = tokenizer(tweet)
    input_ids_orig = tok_tweet.input_ids[1:-1]

    input_ids = [101] + input_ids_orig + [102]
    token_type_ids = tok_tweet['token_type_ids']
    attention_mask = tok_tweet['attention_mask']

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        #         print('fit:',len(input_ids))
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
    elif padding_length <= 0:  # 截断
        input_ids = [101] + input_ids_orig[:max_len - 2] + [102]
        #         print('cut:',len(input_ids))
        attention_mask = attention_mask[:max_len]
        token_type_ids = token_type_ids[:max_len]

    return {  # 这里返回的是list格式的tensor
        'ids': input_ids,
        'mask': attention_mask,
        'token_type_ids': token_type_ids,
        'orig_tweet': tweet,
        #         'sentiment': sentiment,
    }

#划分测试集
df_test = pd.read_csv("./bert_data/test_bert_last.csv",encoding='UTF-8')
# print(df_test['sentiment'].value_counts())

test_dataset = MyDatasetTest(
    tweet=df_test.text.values,
#     sentiment=df_test.sentiment[:2000].values,
)

test_data_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=TEST_BATCH_SIZE,
    num_workers=0 # 这个是多线程数，最好设为0
)

# 小规模测试集，用于实际观察
test_sample_dataset = MyDatasetTest(
    tweet=df_test.text[:10].values,
#     sentiment=df_test.sentiment[:10].values,
)

test_sample_loader = torch.utils.data.DataLoader(
    test_sample_dataset,
    batch_size=TEST_BATCH_SIZE,
    num_workers=0 # 这个是多线程数，最好设为0
)

#测试
def test_fn(data_loader, model):
    final_output = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))

        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            #             sentiment = d["sentiment"]
            orig_tweet = d["orig_tweet"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            outputs = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids,
            )

            outputs_res = torch.softmax(outputs, dim=1).cpu().detach().numpy()

            for px, tweet in enumerate(orig_tweet):
                #                 tweet_sentiment = sentiment[px]
                res_sentiment = np.argmax(outputs_res[px])
                final_output.append(res_sentiment)
    return final_output
# test_y = df_test.sentiment
pred_y = test_fn(test_data_loader,model)

# accuracy = (sum(pred_y == np.array(test_y))) / len(test_y)
# print('acc:%.2f'%accuracy)

result_index = 1
result_test = []
for label in pred_y:
    x = {}
    x['ID'] = result_index
    x['Last Label'] = label+1
    result_test.append(x)
    result_index += 1
with open("./bert_data/bert_result_last.csv",'w',newline='',encoding='UTF-8') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow(['ID','Last Label'])
    for n in result_test:
        writer.writerow(n.values())