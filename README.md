## Bi-LSTM-CRF for Name Entity Recognition

A PyTorch implemention of Bi-LSTM-CRF model for Chinese Named Entity Recognition. 

使用 PyTorch 实现 Bi-LSTM-CRF 模型，用来完成中文命名实体识别任务。


## 数据集

这里采用的数据集来自 [zh-NER-TF](https://github.com/Determined22/zh-NER-TF) 项目，感谢 [Determined22](https://github.com/Determined22) 幸苦处理数据。

这个数据集共有三种实体： PERSON, LOCATION, ORGANIZATION，下面是统计信息：

|  -  | sentence | PER | LOC | ORG |
|:----|:---:|:---:|:---:|:---:|
| train  | 46364 | 17615 | 36517 | 20571 |
| test   | 4365  | 1973  | 2877  | 1331  |


训练数据和测试数据存放在 `datasets` 目录下，其中文件的格式如下所示：

```
中	B-ORG
共	I-ORG
中	I-ORG
央	I-ORG
致	O
中	B-ORG
国	I-ORG
致	I-ORG
公	I-ORG
党	I-ORG
十	I-ORG
一	I-ORG
大	I-ORG
的	O
贺	O
词	O
```

其中句子与句子之间使用空行隔开，在 `data.py` 中有具体读取数据的代码。

## 模型

模型的结构大致如下，这里 Bi-LSTM 层的输入为字向量。Bi-LSTM 对每个字进行编码，然后经过 softmax 后，每个词对应一个长度为 `len(tags)` 的向量，在不使用 CRF 的方法中，就取这个向量中最大的值的位置作为预测的 tag 了，但显然这不能达到最优。

这里每个词的对应的向量作为 CRF 的输入，CRF 会最大化整个序列的概率，因此得到的结果会更优。

![image](https://user-images.githubusercontent.com/7794103/58749767-9f217980-84bc-11e9-806e-773b80ab46d3.png)

图取自: _https://aclweb.org/anthology/N16-1030_

在 PyTorch 中没有 CRF 层，这里使用了 [AllenNLP](https://github.com/allenai/allennlp) 中的 CRF 实现，AllenNLP 中的 CRF 层实现了相当清晰高效，强烈建议使用。


## 配置模型参数

下面是模型的默认参数，大部分都是不用解释的，除 `condtraints` 之外。

在条件随机场中存在一个状态转移矩阵，在这里此状态转移矩阵就包含的是不同 tag 之间转移的概率。但并不是任何状态之间都能进行转移的，比如 `B-PER` 就不可能转移到 `I-LOC` 上。`condtraints` 就用来指明那些状态之间可以转移，这样将极大地减少可能性，在训练和解码过程中，能够大幅提升速度。请务必指定此参数，其创建方法见 `data.py`。

`name` 参数只是在保存模型时用以区别，并未更多含义。

```python
class Config:
    name = "hidden_256_embed_150" 
    hidden_size = 256  
    num_tags = len(TAG_MAP)
    embed_dim = 150
    embed_size = len(dct)
    dropout = 0.5
    device = device
    condtraints = condtraints
```


## 运行代码

如果要实际跑一跑代码，需要确保有 GPU 的支持，我在实验中使用全量数据在 Tasle V100 上需要跑 20 分钟左右。

详情请参见 `main.py` 中的代码，一切都注释的很清楚。下面是部分解释：

**训练模型**

```python
train(model, optimizer, train_dl, val_dl, 
      device=device, epochs=20, early_stop=True,save_every_n_epochs=3)
```

```
2019-06-01 20:25:27,658 - epoch 1 - loss: 6.30 acc: 0.72 - val_acc: 0.69
2019-06-01 20:28:05,706 - epoch 2 - loss: 2.04 acc: 0.82 - val_acc: 0.77
2019-06-01 20:30:53,383 - epoch 3 - loss: 1.30 acc: 0.88 - val_acc: 0.82
2019-06-01 20:33:30,144 - epoch 4 - loss: 0.95 acc: 0.91 - val_acc: 0.84
2019-06-01 20:36:18,832 - epoch 5 - loss: 0.74 acc: 0.92 - val_acc: 0.84
2019-06-01 20:38:55,712 - epoch 6 - loss: 0.60 acc: 0.94 - val_acc: 0.85
2019-06-01 20:41:42,535 - epoch 7 - loss: 0.50 acc: 0.95 - val_acc: 0.85
2019-06-01 20:44:16,465 - epoch 8 - loss: 0.42 acc: 0.96 - val_acc: 0.86
```

**评估模型：**

```python
metric = evaluate(model, test_dl, device)
print(metric.report())
```

测试集上的表现：

```
            PER         LOC         ORG         
precision   0.75        0.85        0.78        
recall      0.83        0.90        0.82        
f1          0.79        0.87        0.80        
------------------------------------------------
precision   0.80
recall      0.86
f1          0.83
```

经过我多次测试，模型最好能在测试集上达到 80% 的准确度。关于准确度的计算请参见 `metric.py` 中相关代码。

**加载已有模型**

在创建了模型后，可以加载先前保存的模型参数，以恢复模型。

```python
from trainer import load_model

load_model(model, 'model_hidden_256_embed_150_epoch_8_acc_0.89.tar')
```

**实际预测**

```python
from predict import predict_sentence_tags, get_entity

sentence = '在周恩来总理的领导下，由当时中共中央主管科学工作的陈毅、国务院副总理兼国家计委主任李富春具体领导，在北京召开了包括中央各部门、各有关高等学校和中国科学院的科学技术工作人员大会，动员制定十二年科学发展远景规划。来自全国23个单位的787名科技人员提出了发展远景规划的初步内容，体现出全国“重点发展，迎头赶上”的方针。在规划制定过程中，深切感到某些新技术是现代科学技术发展的关键。为了更快地发展这些新学科，使其在短时间内接近国际水平，把计算技术、自动化、电子学和半导体这四个学科的研究和发展列为“四大紧急措施”，经周恩来总理同意，确定由中国科学院负责采取紧急措施，尽快筹建相应的四个学科研究机构。'

tags = predict_sentence_tags(model, sentence, dct, device)

print(get_entity(sentence, tags))
```

```
{'PER': {'周恩来', '李富春', '陈毅'},
 'ORG': {'中共中央', '中国科学院', '国务院', '国家计委'},
 'LOC': {'北京'}}
```


## 参考

\[1\] [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/pdf/1508.01991v1.pdf)

\[2\] [Neural Architectures for Named Entity Recognition](http://aclweb.org/anthology/N16-1030)

\[3\] [https://github.com/Determined22/zh-NER-TF](https://github.com/Determined22/zh-NER-TF)  





