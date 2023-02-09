import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

class classification_head(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout):
        super(classification_head, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        output = self.dropout(features)
        output = self.dense(output)
        output = self.LayerNorm(output)
        output = torch.tanh(output)
        output = self.dropout(output)
        return self.out_proj(output)
        

class ModelIntentNER(nn.Module):
    def __init__(self, args):
        super(ModelIntentNER, self).__init__()

        self.device_type = args.device
        self.tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
        self.model = RobertaModel.from_pretrained(args.model_name, add_pooling_layer=False)
        
        hidden_size = self.model.config.hidden_size
        dropout_percent = self.model.config.hidden_dropout_prob

        self.nre_classifier = classification_head(hidden_size, len(args.class_names), dropout_percent)
        self.intent_classifier = classification_head(hidden_size, 1, dropout_percent)
        
        if not args.train_both:
            if args.train_intent:
                for param in self.model.parameters():
                    param.requires_grad = True
                for param in self.nre_classifier.parameters():
                    param.requires_grad = False
                for param in self.intent_classifier.parameters():
                    param.requires_grad = True
            if not args.train_intent:
                for param in self.model.parameters():
                    param.requires_grad = True
                for param in self.nre_classifier.parameters():
                    param.requires_grad = True
                for param in self.intent_classifier.parameters():
                    param.requires_grad = False
        
        weight_nre = torch.ones(len(args.class_names)) * 3
        weight_nre[[args.class_names['None']]] = 1
        self.loss_fn_nre = CrossEntropyLoss(ignore_index=-100)
        weight_intent = torch.ones(len(args.intent_names)) * 1
        weight_intent[[args.intent_names['None'], args.intent_names['item_info_lookup'], args.intent_names['item_info']]] = 3
        weight_intent.to(self.device_type)
        # self.loss_fn_intent = BCEWithLogitsLoss(pos_weight=weight_intent)
        self.loss_fn_intent = MSELoss()
        # print(weight_intent)

    def forward(self, text, nre_targets=None, intent_targets=None):
        data = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device_type)
        outputs = self.model(**data).last_hidden_state
        outputs_CLS = outputs[:, 0, :]
        nre_output = self.nre_classifier(outputs)
        intent_output = self.intent_classifier(outputs_CLS)

        nre_loss, intent_loss = None, None
        if nre_targets is not None:
            nre_loss = self.loss_fn_nre(nre_output.view(-1, nre_output.shape[-1]), nre_targets.view(-1))
        if intent_targets is not None:
            itraget = intent_targets.argmax(-1)
            intent_loss = self.loss_fn_intent(intent_output.squeeze(), itraget.squeeze().float())
            # intent_loss = self.loss_fn_intent(intent_output.view(-1, intent_output.shape[-1]), intent_targets.view(-1))
            # intent_loss = self.loss_fn_intent(intent_output, intent_targets.float())
            # print(intent_loss, intent_output, intent_targets.argmax(-1))

        return nre_output, intent_output, nre_loss, intent_loss