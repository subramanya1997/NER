import torch

import os
import yaml 
import argparse
from tqdm import tqdm

from utils.dataloader import readjson, entityDataset, get_dataloader
from torch.utils.data import random_split
from torch import optim
from model import ModelIntentNER
from transformers import  RobertaTokenizer, RobertaForTokenClassification
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

def parse_arguments():
    print("-------------read args-------------")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config-file", help="Config File with all the parameters", default='config.yaml'
    )
    parser.add_argument(
        "-lr", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument(
        "-wd", type=float, default=1e-8, help="weight decay"
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
    print("-------------end read args-------------")
    return parsed_args

def createModel(args):
    print("-------------create model-------------")
    model, optimizer, lr_scheduler = None, None, None
    model = ModelIntentNER(args).to(args.device)
    if args.train_intent and False:
        _path = os.path.join(args.save_result_dir, f"{args.model_name}-best.pth")
        print(f"load model from {_path}")
        model.load_state_dict(torch.load(_path, map_location=args.device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop_step)
    print("-------------end create model-------------")
    return model, optimizer, lr_scheduler

def convert_length_mask(lengths):
    max_len = lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device).expand(
        lengths.size()[0], max_len
    ) < lengths.unsqueeze(1)
    mask = mask.long()
    return mask

def train(args, train_loader, model, optimizer, lr_scheduler, epoch=1):
    print("-------------train-------------")
    model.train()
    train_loss = 0
    train_acc = 0
    train_intent_acc = 0
    for batch_idx, (data, targets, tlens, intents) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"train : epoch {epoch}"):
        _mask = convert_length_mask(tlens)
        # to device
        targets = targets.to(args.device)
        intents = intents.to(args.device)
        _mask = _mask.to(args.device)
        # forward
        nre_output, intent_output, nre_loss, intent_loss = model(data, targets, intents)
        # acc
        train_acc += ((nre_output.argmax(-1) == targets) * _mask).sum().item() / _mask.view(-1).sum().item()
        train_intent_acc += (intent_output.argmax(-1) == intents.argmax(-1)).sum().item() / len(intents)
        # loss
        if not args.train_both:
            if args.train_intent:
                loss = intent_loss
            else:
                loss = nre_loss
        else:
            loss = 0.8 * nre_loss + 0.2 * intent_loss
        train_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if batch_idx % 25 == 0:
            print(intent_output, intents.argmax(-1))
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
            print(f"nre_loss: {nre_loss.item():.4f}, intent_loss: {intent_loss.item():.4f}")
            print(f"Train acc: {train_acc / (batch_idx + 1):.4f}, Train intent acc: {train_intent_acc / (batch_idx + 1):.4f}")
    print(f"Epoch: {epoch}, Loss: {train_loss:.4f}")
    print("-------------end train-------------")
    return train_loss / len(train_loader), train_acc / len(train_loader)

def val(args, val_loader, model, epoch=1):
    print("-------------evaluate-------------")
    model.eval()
    val_loss = 0
    val_acc = 0
    val_intent_acc = 0
    for batch_idx, (data, targets, tlens, intents) in tqdm(enumerate(val_loader), total=len(val_loader), desc=f"val : epoch {epoch}"):
        _mask = convert_length_mask(tlens)
        # to device
        targets = targets.to(args.device)
        intents = intents.to(args.device)
        _mask = _mask.to(args.device)
        # forward
        nre_output, intent_output, nre_loss, intent_loss = model(data, targets, intents)
        # acc
        val_acc += ((nre_output.argmax(-1) == targets) * _mask).sum().item() / _mask.view(-1).sum().item()
        val_intent_acc += (intent_output.argmax(-1) == intents.argmax(-1)).sum().item() / len(intents)
        # loss
        if not args.train_both:
            if args.train_intent:
                loss = intent_loss
            else:
                loss = nre_loss
        else:
            loss = 0.8 * nre_loss + 0.2 * intent_loss

        if batch_idx % 25 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")
            print(f"nre_loss: {nre_loss.item():.4f}, intent_loss: {intent_loss.item():.4f}")
            print(f"Val acc: {val_acc / (batch_idx + 1):.4f}, Val intent acc: {val_intent_acc / (batch_idx + 1):.4f}")
    print(f"Epoch: {epoch}, Loss: {val_loss:.4f}")
    print("-------------end val-------------")
    return val_loss / len(val_loader), val_acc / len(val_loader)

def train_and_val(args, train_loader, val_loader, model, optimizer, lr_scheduler):
    print("-------------train and val-------------")
    if not os.path.exists(args.save_result_dir):
        os.makedirs(args.save_result_dir)
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(args, train_loader, model, optimizer, lr_scheduler, epoch)
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        val_loss, val_acc = val(args, val_loader, model, epoch)
        print(f"Epoch: {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            _path = os.path.join(args.save_result_dir, f"{args.model_name}_{epoch}_{val_acc:.4f}.pth")
            torch.save(model.state_dict(), _path)

    print("-------------end train and val-------------")
    return best_acc
    

if __name__ == "__main__":
    args = parse_arguments()
    data, labels, intent, class_names, intent_names = readjson(args, split="train")
    args.class_names = class_names
    args.intent_names = intent_names
    print(args)
    train_dataset = entityDataset(data, labels, intent, class_names, intent_names)
    data, labels, intent, class_names, intent_names = readjson(args, split="test")
    dataset = entityDataset(data, labels, intent, class_names, intent_names)
    val_size, test_size = int(args.val_split*len(dataset)), int(args.test_split*len(dataset))
    val_dataset, test_dataset = random_split(dataset, [val_size, test_size+1])  
    train_loader, val_loader, test_loader = get_dataloader(args, train_dataset, val_dataset, test_dataset)
    model, optimizer, lr_scheduler = createModel(args)
    best_acc = train_and_val(args, train_loader, val_loader, model, optimizer, lr_scheduler)
    print(f"Best acc: {best_acc:.4f}")

