import argparse
import torch
import numpy as np
# import gensim

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default=None, type=str, required=True)
# parser.add_argument('--prefix', default=None, type=str, required=True)
# parser.add_argument('--lm', default=None, type=str, required=True)
# parser.add_argument('--device', default=None, type=str, required=True)
# parser.add_argument('--method', default=None, type=str, required=True)

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

rel2id_f = "/mnt/nlp_data/Freebase/knowledge_graphs/relation2id.txt"
rel2vec_f = "/mnt/nlp_data/Freebase/embeddings/dimension_50/transe/relation2vec.bin"

rel2id = {}
with open(rel2id_f, "r") as f:
    l = 0
    for line in f.readlines():
        if len(line.split('\t')) == 1:
            continue
        rel, idx = line.strip().strip("\n").split("\t")
        rel = rel.strip().strip("\n")
        idx = int(idx.strip().strip("\n"))
        rel2id[rel] = idx
        l += 1
assert l == len(rel2id)

# rel2vec = gensim.models.KeyedVectors.load_word2vec_format(rel2vec_f,binary=True)
# rel2vec = np.load(rel2vec_f,allow_pickle=True)
# print(rel2vec.shape)
vec_size = 50
rel2vec = np.memmap(rel2vec_f ,dtype='float32',mode='r')
rel2vec = rel2vec.reshape(int(rel2vec.shape[0]/vec_size),vec_size)
print(rel2vec.shape)

filename = "/mnt/jiangjinhao/KBQA/datasets/" + args.datasets + "/relations.txt"
embeddings_dump_fn = "/mnt/jiangjinhao/KBQA/datasets/" + args.datasets + "/relations_transe_embedding"
rel_idx = []
kyes = rel2id.keys()
with open(filename,"r") as f:
    no_shot = 0
    for line in f.readlines():
        line = line.strip().strip("\n")
        try:
            idx = rel2id[line]
        except KeyError:
            print(line, " not shot")
            key = line.split(".")[-1]
            flag = False
            for k in kyes:
                if key in k:
                    print(line, " shot by last phrase")
                    idx = rel2id[k]
                    flag = True
                    break
            if not flag:
                no_shot += 1
print("Total %d no shot relation"%(no_shot))
