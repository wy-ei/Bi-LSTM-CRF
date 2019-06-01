import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from data import read_corpus, build_dict, TAG_MAP, NER_DataSet, condtraints
from bi_lstm_crf import BiLSTM_CRF
from trainer import train, evaluate, load_model


train_corpus_path = './datasets/train_data'
test_corpus_path = './datasets/test_data'

if __name__ == '__main__':
    
    # prepare data
    corpus = read_corpus(train_corpus_path)
    dct = build_dict(corpus)

    # build dataloader
    np.random.shuffle(corpus)
    train_ds = NER_DataSet(corpus[:-5000], dct)
    val_ds = NER_DataSet(corpus[-5000:], dct)
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=True, num_workers=0)


    # create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class Config:
        name = "hidden_512_embed_150"
        hidden_size = 512
        num_tags = len(TAG_MAP)
        embed_dim = 300
        embed_size = len(dct)
        dropout = 0.5
        device = device
        condtraints = condtraints

    model = BiLSTM_CRF(Config())
    model = model.to(device)
    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # train model
    train(model, optimizer, train_dl, val_dl, device=device, epochs=20, early_stop=True, save_every_n_epochs=3)


    # evaluate
    test_corpus = read_corpus(test_corpus_path)
    test_ds = NER_DataSet(test_corpus, dct)
    test_dl = DataLoader(test_ds, batch_size=64)
    
    metric = evaluate(model, test_dl, device)
    print(metric.report())