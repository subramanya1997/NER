import torch

import os
import yaml 
import argparse
from tqdm import tqdm

from utils.dataloader import readjson, entityDataset, get_dataloader
from torch.utils.data import random_split
from transformers import  RobertaTokenizer, RobertaForTokenClassification, RobertaConfig

def parse_arguments():
    print("-------------read args-------------")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config-file", help="Config File with all the parameters", default='config.yaml'
    )

    try:
        parsed_args = parser.parse_args()
    except (IOError) as msg:
        parser.error(str(msg))

    # Read config yamls file
    config_file = parsed_args.config_file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        if key not in parsed_args.__dict__ or parsed_args.__dict__[key] is None:
            value = value if value != "None" else None
            parsed_args.__dict__[key] = value

    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(parsed_args)
    print("-------------end read args-------------")
    return parsed_args

def createModel(args):
    print("-------------create model-------------")
    tokenizer, model = None, None
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    my_config = RobertaConfig.from_pretrained(f"{args.save_dir}/config.json")
    model = RobertaForTokenClassification(my_config)
    _path = os.path.join(args.save_result_dir, f"{args.model_name}_best.pth")
    model.load_state_dict(torch.load(_path, map_location=args.device))
    print("-------------end create model-------------")
    return tokenizer, model

def convert_length_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(
        lengths.size()[0], max_len
    ) < lengths.unsqueeze(1)
    mask = mask.long()
    return mask

def get_buckets(pred):
    final_buckets = []
    curr = torch.tensor(0)
    curr_i = 0
    for i in range(pred.shape[0]):
        if curr != pred[i]:
            final_buckets.append((curr_i, i, curr.item()))
            curr = pred[i]
            curr_i = i
    return final_buckets

def test(args, train_loader, tokenizer, model):
    print("-------------test-------------")
    model.eval()
    with torch.no_grad():
        for batch_idx, (text, targets, tlens) in enumerate(train_loader):
            data = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            _mask = convert_length_mask(tlens)
            # to device
            data.to(args.device)
            targets = targets.to(args.device)
            _mask = _mask.to(args.device)
            # forward
            logits = model(**data).logits

            for i in range(len(text)):
                print(" ")
                print(f"Text: {text[i]}")
                buckets = get_buckets(logits.argmax(dim=-1)[i])
                temp = ""
                for s, e, c in buckets:
                    if c != 0:
                        print(f"{args.idx_class[c]} : {tokenizer.decode(data['input_ids'][i][s:e])}")
                        temp += tokenizer.decode(data['input_ids'][i][s:e]) + " "
                print(f"Predicted: {temp.strip()}")
            if batch_idx > 10:
                return
    print("-------------end test-------------")



if __name__ == "__main__":
    args = parse_arguments()
    data, labels, class_names = readjson(args)
    args.class_names = class_names
    idx_class = {}
    for k, v in class_names.items():
        idx_class[v] = k
    args.idx_class = idx_class
    dataset = entityDataset(data, labels, class_names)
    train_size, val_size = int(args.train_spilt*len(dataset)), int(args.val_split*len(dataset))
    test_size = len(dataset)-train_size-val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])  
    train_loader, val_loader, test_loader = get_dataloader(args, train_dataset, val_dataset, test_dataset)
    tokenizer, model = createModel(args)
    test(args, train_loader, tokenizer, model)