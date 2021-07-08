import json
import random
import numpy as np

load_fn = "/mnt/jiangjinhao/KBQA/datasets/CFQ/"
dep = "pretrain_filtered.dep"
simple = "pretrain_filtered_simple.json"
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
# pair_list = [i for i in zip(dep_list, simple_list)]
# np.random.shuffle(pair_list)
# dep_list_new = []
# simple_list_new = []
# for pair in pair_list:
#     dep_list_new.append(pair[0])
#     simple_list_new.append(pair[1])
idx = np.arange(0,l1)
np.random.shuffle(idx)
print("shuffle 1: ",idx[0:5])
np.random.shuffle(idx)
print("shuffle 2: ",idx[0:5])

test_len = int(l1*0.2)
dev_len = test_len
train_len = l1 - 2*test_len

train_idx = idx[0:train_len]
test_idx = idx[train_len:train_len+dev_len]
dev_idx = idx[train_len+dev_len:]
print("train: {}, test: {} dev: {}".format(len(train_idx), len(test_idx), len(dev_idx)))
assert (len(train_idx) + len(test_idx) + len(dev_idx)) == l1

dep_list_train = []
simple_list_train = []
for i in train_idx:
    dep_list_train.append(dep_list[i])
    simple_list_train.append(simple_list[i])

dep_list_test = []
simple_list_test = []
for i in test_idx:
    dep_list_test.append(dep_list[i])
    simple_list_test.append(simple_list[i])

dep_list_dev = []
simple_list_dev = []
for i in dev_idx:
    dep_list_dev.append(dep_list[i])
    simple_list_dev.append(simple_list[i])

print("Begin save file")
save_fn = "/mnt/jiangjinhao/KBQA/datasets/CFQ/"

for i in ["train", "test", "dev"]:
    fn_dep = save_fn + i + ".dep"
    fn_simple = save_fn + i + "_simple.json"

    if i == "train":
        dep = dep_list_train
        simple = simple_list_train
    elif i == "test":
        dep = dep_list_test
        simple = simple_list_test
    elif i == "dev":
        dep = dep_list_dev
        simple = simple_list_dev

    with open(fn_dep, mode="w") as f:
        for line in dep:
            f.write(line)
    print("save dep to {}".format(fn_dep))
    with open(fn_simple, mode="w") as f:
        for line in simple:
            f.write(line)
    print("save simple to {}".format(fn_simple))


