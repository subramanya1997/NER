import os
import glob
import pandas as pd
from tqdm import tqdm
import json
import re
import string
import pickle

from utils.datautils import load_pickle, save_pickle

import torch
from torch.utils.data import Dataset, DataLoader

def readCVS(args, save=True):

    print("-------------read data-------------")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    _path = os.path.join(args.save_dir, "data.pkl")
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

class entityDataset(Dataset):
    def __init__(self, data, labels, class_names):
        print("-------------creating dataset-------------")
        self.data = data
        self.labels = labels
        self.class_names = class_names
        self.len = len(data)
        print("-------------end dataset-------------")
    def __getitem__(self, index):
        return self.data[index], self.labels[index], len(self.labels[index])
    def __len__(self):
        return self.len

def collate_fn(batch):
    _data, _labels, _tlen = zip(*batch)
    max_len = max(_tlen)
    sequence_padded = []
    for _label in _labels:
        _label_padded = _label + [0] * (max_len - len(_label))
        sequence_padded.append(torch.tensor(_label_padded))
    return _data, torch.stack(sequence_padded), torch.tensor(_tlen)

def get_dataloader(args, train_dataset, val_dataset=None, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn) 
    return train_loader, val_loader, test_loader