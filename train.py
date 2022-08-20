import torch

import os
import yaml 
import argparse
from tqdm import tqdm

from utils.dataloader import readCVS, entityDataset, get_dataloader
from torch.utils.data import random_split
from torch import optim
from transformers import  RobertaTokenizer, RobertaForTokenClassification
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

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
    tokenizer, model, optimizer, lr_scheduler = None, None, None, None
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForTokenClassification.from_pretrained(args.model_name, num_labels=len(args.class_names)).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_step)
    loss_fn = CrossEntropyLoss()
    print("-------------end create model-------------")
    return tokenizer, model, optimizer, lr_scheduler, loss_fn

def convert_length_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(
        lengths.size()[0], max_len
    ) < lengths.unsqueeze(1)
    mask = mask.long()
    return mask

def train(args, train_loader, tokenizer, model, optimizer, lr_scheduler, loss_fn, epoch=1):
    print("-------------train-------------")
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_idx, (data, targets, tlens) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"train : epoch {epoch}"):
        data = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
        _mask = convert_length_mask(tlens)
        # to device
        data.to(args.device)
        targets = targets.to(args.device)
        _mask = _mask.to(args.device)
        # forward
        logits = model(**data).logits
        # loss
        bloss = 0
        for logit, target, mask in zip(logits, targets, _mask):
            _logit = logit[mask]
            _target = target[mask]
            bloss += loss_fn(_logit, _target)
            train_acc += (_logit.argmax(-1) == _target).sum().item()
        bloss.backward()
        train_loss += bloss.item()

        optimizer.zero_grad()
        optimizer.step()
        lr_scheduler.step()
        
    print("-------------end train-------------")
    return train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset)

def val(args, val_loader, tokenizer, model, loss_fn, epoch=1):
    print("-------------train-------------")
    model.eval()
    val_loss = 0
    val_acc = 0
    for batch_idx, (data, targets, tlens) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"val : epoch {epoch}"):
        data = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
        _mask = convert_length_mask(tlens)
        # to device
        data.to(args.device)
        targets = targets.to(args.device)
        _mask = _mask.to(args.device)
        # forward
        logits = model(**data).logits
        # loss
        bloss = 0
        for logit, target, mask in zip(logits, targets, _mask):
            _logit = logit[mask]
            _target = target[mask]
            bloss += loss_fn(_logit, _target)
            val_acc += (_logit.argmax(-1) == _target).sum().item()
        val_loss += bloss.item()
    print("-------------end val-------------")
    return val_loss / len(val_loader.dataset), val_acc / len(val_loader.dataset)

def train_and_val(args, train_loader, val_loader, tokenizer, model, optimizer, lr_scheduler, loss_fn):
    print("-------------train and val-------------")
    if not os.path.exists(args.save_result_dir):
        os.makedirs(args.save_result_dir)
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(args, train_loader, tokenizer, model, optimizer, lr_scheduler, loss_fn, epoch)
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        val_loss, val_acc = val(args, val_loader, tokenizer, model, loss_fn, epoch)
        print(f"Epoch: {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            _path = os.path.join(args.save_result_dir, f"{args.model_name}_{epoch}_{val_acc:.4f}.pth")
            torch.save(model.state_dict(), _path)

if __name__ == "__main__":
    args = parse_arguments()
    data, labels, class_names = readCVS(args)
    args.class_names = class_names
    dataset = entityDataset(data, labels, class_names)
    train_size, val_size = int(args.train_spilt*len(dataset)), int(args.val_split*len(dataset))
    test_size = len(dataset)-train_size-val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])  
    train_loader, val_loader, test_loader = get_dataloader(args, train_dataset, val_dataset, test_dataset)
    tokenizer, model, optimizer, lr_scheduler, loss_fn = createModel(args)
    train_loss, train_acc = train_and_val(args, train_loader, val_loader, tokenizer, model, optimizer, lr_scheduler, loss_fn)

