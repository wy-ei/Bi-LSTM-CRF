import os
from collections import Counter
import pickle
import re

import torch
from torch.utils.data import Dataset, DataLoader
from allennlp.modules.conditional_random_field import allowed_transitions

PAD_WORD = '<PAD>'
UNK_WORD = '<UNK>'
NUM_WORD = '<NUM>'
ENG_WORD = '<ENG>'


# 文档最大长度限制
SEQUENCE_MAX_LENGTH = 60

# tags, BIO
TAG_MAP = {
    "O": 0,
    "B-PER": 1, "I-PER": 2,
    "B-LOC": 3, "I-LOC": 4,
    "B-ORG": 5, "I-ORG": 6
}

BEGIN_TAGS = set([1,3,5])
OUT_TAG = TAG_MAP['O']

TAG_MAP_REVERSED = {
    v: k for k, v in TAG_MAP.items()
}

condtraints = allowed_transitions('BIO', TAG_MAP_REVERSED)

def get_entity_type(tag):
    return TAG_MAP_REVERSED[tag].split('-')[1]

def read_corpus(corpus_path):
    """
    read corpus and return the list of samples
    :param corpus_path:
    :return: samples
    """
    corpus = []

    with open(corpus_path, encoding='utf-8', mode='r') as fin:
        sequence, tags = [], []
        for line in fin:
            if line != '\n':
                [char, tag] = line.strip().split()
                sequence.append(char)
                tags.append(tag)
            else:
                corpus.append((sequence, tags))
                sequence, tags = [], []

    return corpus


def build_dict(corpus, num_words=5000):
    dct_file = './datasets/dct.pkl'
    if os.path.exists(dct_file):
        with open(dct_file, mode='rb') as fin:
            dct = pickle.load(fin)
            return dct

    counter = Counter()
    for sequence, _ in corpus:
        counter.update(sequence)
    
    words = [w for w, c in counter.most_common(num_words - 4)]
    words =  [PAD_WORD, UNK_WORD, NUM_WORD, ENG_WORD] + words
    
    dct = {word: i for i, word in enumerate(words)}

    with open(dct_file, mode='wb') as fout:
        pickle.dump(dct, fout)

    return dct

def sentence_to_tensor(sentence, dct):
    UNK = dct[UNK_WORD]
    idx = [dct.get(w, UNK) for w in sentence]
    idx = torch.tensor(idx, dtype=torch.long)
    return idx

class NER_DataSet(Dataset):
    def __init__(self, corpus, dictionary, sequence_max_length=SEQUENCE_MAX_LENGTH):
        self.sequence_max_length = sequence_max_length
        self.dct = dictionary
        self.samples = self.process_corpus(corpus)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        return self.samples[i]
    
    def process_item(self, sequence, tags):
        UNK = self.dct[UNK_WORD]
        PAD = self.dct[PAD_WORD]
        UNM = self.dct[NUM_WORD]
        ENG = self.dct[ENG_WORD]

        if len(sequence) > self.sequence_max_length:
            sequence = sequence[:self.sequence_max_length]
            tags = tags[:self.sequence_max_length]
        
        seq = sequence
        sequence = []
        for w in seq:
            if w.isdigit():
                sequence.append(UNM)
            elif ('\u0041' <= w <='\u005a') or ('\u0061' <= w <='\u007a'):
                sequence.append(ENG)
            else:
                sequence.append(self.dct.get(w, UNK))

        tags = [TAG_MAP[tag] for tag in tags]

        if len(sequence) < self.sequence_max_length:
            sequence += [PAD] * (self.sequence_max_length - len(sequence)) 
            tags += [0] * (self.sequence_max_length - len(tags))
        
        sequence = torch.tensor(sequence, dtype=torch.long)
        tags = torch.tensor(tags, dtype=torch.long)

        return sequence, tags

    def process_corpus(self, corpus):
        samples = [
            self.process_item(sequence, tags)
                for sequence, tags in corpus
        ]
        return samples