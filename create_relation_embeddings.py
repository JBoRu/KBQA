import torch
from transformers import AutoTokenizer
from transformers import BertModel, RobertaModel
import numpy as np
import os
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


parser = argparse.ArgumentParser()
# datasets
parser.add_argument('--datasets', default=None, type=str, required=True)
parser.add_argument('--prefix', default=None, type=str, required=True)
parser.add_argument('--lm', default=None, type=str, required=True)
parser.add_argument('--device', default=None, type=str, required=True)
parser.add_argument('--method', default=None, type=str, required=True)
# filename = "./dataset/CWQ/relations.txt"
# filename = "/mnt/jiangjinhao/KBQA/datasets/CWQ/relations.txt"
# embeddings_dump_fn = "/mnt/jiangjinhao/KBQA/datasets/CWQ/relations_bert_embedding"
# filename = "/mnt/jiangjinhao/KBQA/datasets/CFQ/relations.txt"
# embeddings_dump_fn = "/mnt/jiangjinhao/KBQA/datasets/CFQ/relations_bert_embedding"
# filename = "/mnt/jiangjinhao/KBQA/datasets/webqsp/relations.txt"
# embeddings_dump_fn = "/mnt/jiangjinhao/KBQA/datasets/webqsp/relations_bert_embedding"

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

filename = os.path.join(args.prefix, args.datasets, "relations.txt")
postfix = "relations_"+args.lm+"_"+args.method+"_embedding"
embeddings_dump_fn = os.path.join(args.prefix, args.datasets, postfix)

rel_str_list = []
with open(filename, encoding='utf-8') as f_in:
    for line in f_in:
        rel_token_list = []
        rel = line.strip()
        rel_ori = line.strip()
        rel_tokens = rel.split(".")
        rel_tokens = [r.split("_") for r in rel_tokens]
        for t in rel_tokens:
            rel_token_list.extend(t)
        rel_str = " ".join(rel_token_list)
        rel_str_list.append(rel_str)
        if len(rel_str_list) < 3:
            print(rel_ori+"---->"+rel_str)
print("total %d relations"%(len(rel_str_list)))
device = "cuda:" + args.device

if args.lm == "bert":
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    pretrained_model = BertModel.from_pretrained('bert-base-uncased')
elif args.lm == "roberta":
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    pretrained_model = RobertaModel.from_pretrained('roberta-base')

print("load lm over")
encoded_input = tokenizer(text=rel_str_list, padding=True, return_tensors="pt")
sequence_ids = encoded_input["input_ids"]
# token_type_ids = encoded_input["token_type_ids"]
sequence_mask = encoded_input["attention_mask"]
outputs = pretrained_model(**encoded_input)
last_hidden_state = outputs.last_hidden_state # (bs, seq_len, hid_dim)

if args.method == "cls":
    rel_cls = last_hidden_state[:, 0, :]
elif args.method == "mean":
    sep_idx = torch.sum(sequence_mask, dim=-1).long() -1
    bs = len(sep_idx)
    bs_idx = torch.arange(0, bs).long()
    sequence_mask[bs_idx, sep_idx] = 0
    sequence_mask[:, 0] = 0 # (bs, seq_len)
    sequence_mask = sequence_mask.unsqueeze(dim=-1) # (bs, seq_len， 1)
    last_hidden_state = sequence_mask * last_hidden_state # (bs, seq_len, hid_dim)
    sep_idx = sep_idx.unsqueeze(dim=-1) # (bs, 1)
    rel_cls = torch.sum(last_hidden_state, dim=1) / (sep_idx-1) # (bs, hid_dim)

print(rel_cls.size())

rel_cls_np = rel_cls.data.cpu().numpy()
np.save(embeddings_dump_fn, rel_cls_np)
print("save!")




