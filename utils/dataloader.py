import os
import glob
import pandas as pd
from tqdm import tqdm
import json
import re
import string
import pickle
import random

from utils.datautils import load_pickle, save_pickle

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

def readCVS(args, save=True):

    print("-------------read data-------------")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    _path = os.path.join(args.save_dir, "data.pkl")
    print(f"Save Path {_path}")
    if os.path.exists(args.save_dir + "/data.pkl") and not args.force_rewrite:
        _data_loaded = load_pickle(_path)
        print("-------------end read data-------------")
        return _data_loaded["data"], _data_loaded["labels"], _data_loaded["class_names"]

    
    data = pd.read_csv(args.datafile, header=0, low_memory=False)
    print("-------------end read data-------------")
    print("-------------process data-------------")
    _class_names = list(data.columns.values)[1:]
    _class_names = dict(zip(_class_names, range(1, len(_class_names)+1)))
    _class_names["none"] = 0
    _data = []
    _labels = []
    dataJson = json.loads(data.to_json(orient="index"))
    _chars = re.escape(string.punctuation)
    for idx, data in tqdm(dataJson.items()):
        text = data['Text']
        text = text.translate ({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})
        # text = re.sub(' +', ' ', text)
        words = text.split()
        text_list = [_class_names["none"]] * len(words)
        for _c_name, _words in data.items():
            if _c_name == 'Text' or _words == None:
                continue
            _words = _words.translate ({ord(c): " " for c in "!@#$%^&*()[]{};:,./<>?\|`~-=_+"})
            _words = re.sub(' +', ' ', _words)
            for _word in _words.split():
                text_list[words.index(_word)] = _class_names[_c_name]
        _data.append(text)
        _labels.append(text_list)
    print("-------------end process data-------------")
    
    if save:
        _data_to_save = {"data": _data, "labels": _labels, "class_names": _class_names}
        save_pickle(_data_to_save, _path)
    
    return _data, _labels, _class_names

def readjson(args, split="train", save=True):

    print("-------------read data-------------")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    _path = os.path.join(args.save_dir, f"data_{args.model_name}_{split}.pkl")
    print(f"Data Path {_path}")
    if os.path.exists(_path) and not args.force_rewrite:
        _data_loaded = load_pickle(_path)
        print("-------------end read data-------------")
        if True:
            _indexs = random.choices(range(len(_data_loaded["data"])-205), k=6)
            _indexs.extend(range(len(_data_loaded["data"])-205, len(_data_loaded["data"])))
            _data_loaded["data"] = [_data_loaded["data"][i] for i in _indexs]
            _data_loaded["labels"] = [_data_loaded["labels"][i] for i in _indexs]
            _data_loaded["intent"] = [_data_loaded["intent"][i] for i in _indexs]
            return _data_loaded["data"], _data_loaded["labels"], _data_loaded["intent"],  _data_loaded["class_names"], _data_loaded["intent_names"]
        return _data_loaded["data"], _data_loaded["labels"], _data_loaded["intent"],  _data_loaded["class_names"], _data_loaded["intent_names"]
    
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    if split == "train":
        with open(args.train_datafile, "r") as f:
            dataJSON = json.load(f)
    elif split == "test":
        with open(args.test_datafile, "r") as f:
            dataJSON = json.load(f)
    print("-------------end read data-------------")
    print("-------------process data-------------")
    _class_names = dataJSON["categories"]
    _class_names = dict(zip(_class_names, range(0, len(_class_names))))
    _intent_names = dataJSON["intents"]
    _intent_names = dict(zip(_intent_names, range(0, len(_intent_names))))
    _data = []
    _labels = []
    _intent = []
    for idx, data in tqdm(enumerate(dataJSON["data"]), total=len(dataJSON["data"])):
        text = data['Text'].strip()
        tokenizered_text = tokenizer(text, padding=True, truncation=True, return_tensors="pt").input_ids[0]
        text_list = [_class_names["None"]] * tokenizered_text.shape[0]
        intent_list = [0.0] * len(_intent_names)
        intent_list[_intent_names[data['Intent']]] = 1.0
        for _c_name, _phrases in data.items():
            if _c_name == 'Text' or _c_name == 'Intent':
                continue
            for _words in _phrases:
                _words = _words.strip()
                _words_withspace = f" {_words}"
                tokenizered_words_withoutspace = tokenizer(_words, padding=True, truncation=True, return_tensors="pt").input_ids[0][1:-1]
                tokenizered_words_withspace = tokenizer(_words_withspace, padding=True, truncation=True, return_tensors="pt").input_ids[0][1:-1]
                temp_text_withoutspace = tokenizered_text.unfold(0, tokenizered_words_withoutspace.shape[0], 1)
                temp_text_withspace = tokenizered_text.unfold(0, tokenizered_words_withspace.shape[0], 1)
                idx_1 = (temp_text_withoutspace[:, None] == tokenizered_words_withoutspace).all(-1).any(-1).nonzero().flatten()
                idx_2 = (temp_text_withspace[:, None] == tokenizered_words_withspace).all(-1).any(-1).nonzero().flatten()
                if idx_1.shape[0] == 1:
                    for i in range(idx_1[0], idx_1[0]+tokenizered_words_withoutspace.shape[0]):
                        text_list[i] = _class_names[_c_name]
                elif idx_2.shape[0] == 1:
                    for i in range(idx_2[0], idx_2[0]+tokenizered_words_withspace.shape[0]):
                        text_list[i] = _class_names[_c_name]
        _data.append(text)
        _labels.append(text_list)
        _intent.append(intent_list)

    print("-------------end process data-------------")
    if save:
        _data_to_save = {"data": _data, "labels": _labels, "intent": _intent, "class_names": _class_names, "intent_names": _intent_names}
        save_pickle(_data_to_save, _path)
    return _data, _labels, _intent, _class_names, _intent_names

class entityDataset(Dataset):
    def __init__(self, data, labels, intents, class_names, intent_names):
        print("-------------creating dataset-------------")
        self.data = data
        self.labels = labels
        self.intents = intents
        self.class_names = class_names
        self.intent_names = intent_names
        self.len = len(data)
        print("-------------end dataset-------------")
    def __getitem__(self, index):
        return self.data[index], self.labels[index], len(self.labels[index]), self.intents[index]
    def __len__(self):
        return self.len

def collate_fn(batch):
    _data, _labels, _tlen, _intent = zip(*batch)
    max_len = max(_tlen)
    sequence_padded = []
    for _label in _labels:
        _label_padded = _label + [-100] * (max_len - len(_label))
        sequence_padded.append(torch.tensor(_label_padded))
    return _data, torch.stack(sequence_padded), torch.tensor(_tlen), torch.tensor(list(_intent))

def get_dataloader(args, train_dataset, val_dataset=None, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) 
    return train_loader, val_loader, test_loader