import json
import os.path
import random
import numpy as np

load_fn = "/mnt/jiangjinhao/KBQA/datasets/CFQ/"
dep = "train.dep"
simple = "train_simple.json"
fn_dep = load_fn+dep
fn_simple = load_fn+simple

dep_list = []
simple_list = []
l1 = 0
l2 = 0
with open(fn_dep, mode="r") as f:
    for line in f.readlines():
        dep_list.append(line)
        l1 += 1
print("load dep file: {}".format(l1))
with open(fn_simple, mode="r") as f:
    for line in f.readlines():
        simple_list.append(line)
        l2 += 1
print("load simple file: {}".format(l2))
assert l1==l2

# idx = np.arange(0,l1)
idx = [i for i in range(0,l1)]
np.random.shuffle(idx)
print("shuffle 1: ",idx[0:5])
np.random.shuffle(idx)
print("shuffle 2: ",idx[0:5])

for i in [2, 4, 6, 8]:
    train_len = int(l1*(i/10))
    train_idx = random.sample(idx, train_len)
    dep_list_train = []
    simple_list_train = []
    for j in train_idx:
        dep_list_train.append(dep_list[j])
        simple_list_train.append(simple_list[j])
    assert len(dep_list_train) == len(simple_list_train) == train_len

    print("Begin save file")
    save_fn = "/mnt/jiangjinhao/KBQA/datasets/CFQ"
    fn_dep = os.path.join(save_fn, "train.dep"+str(i))
    fn_simple = os.path.join(save_fn, "train_simple.json"+str(i))
    with open(fn_dep, mode="w") as f:
        for line in dep_list_train:
            f.write(line)
    print("save dep to {}".format(fn_dep))
    with open(fn_simple, mode="w") as f:
        for line in simple_list_train:
            f.write(line)
    print("save simple to {}".format(fn_simple))


