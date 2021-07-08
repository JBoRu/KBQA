import json
import numpy as np
from NSM.util.config import get_config
import time
from NSM.data.dataset_super import SingleDataLoader


def load_dict(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id


def load_data(config, logger):
    entity2id = load_dict(config['data_folder'] + config['entity2id'])
    word2id = load_dict(config['data_folder'] + config['word2id'])
    relation2id = load_dict(config['data_folder'] + config['relation2id'])
    # if config["is_eval"]:
    #     train_data = None
    #     valid_data = None
    # else:
    #     train_data = SingleDataLoader(config, word2id, relation2id, entity2id, logger, data_type="train")
    #     valid_data = SingleDataLoader(config, word2id, relation2id, entity2id, logger, data_type="dev")
    # test_data = SingleDataLoader(config, word2id, relation2id, entity2id, logger, data_type="test")
    test_data_type = None
    if config["is_eval"]:
        if config["eval_different_type"]:
            train_data = None
            valid_data = None
            test_data = SingleDataLoader(config, word2id, relation2id, entity2id, logger, data_type="test")
            num_kb_relation = test_data.num_kb_relation
            test_data_type ={}
            for type in ["composition", "conjunction", "comparative", "superlative"]:
                data_loader = SingleDataLoader(config, word2id, relation2id, entity2id, logger, data_type="test", question_type=type)
                test_data_type[type] = data_loader
        else:
            train_data = None
            valid_data = None
            test_data = SingleDataLoader(config, word2id, relation2id, entity2id, logger, data_type="test")
            num_kb_relation = test_data.num_kb_relation
    # elif config["pretrain"]:
    #     train_data = SingleDataLoader(config, word2id, relation2id, entity2id, logger, data_type="pretrain_filtered")
    #     valid_data = None
    #     test_data = None
    #     num_kb_relation = train_data.num_kb_relation
    elif config["pretrain"]:
        train_data = SingleDataLoader(config, word2id, relation2id, entity2id, logger, data_type="train")
        valid_data = SingleDataLoader(config, word2id, relation2id, entity2id, logger, data_type="dev")
        test_data = SingleDataLoader(config, word2id, relation2id, entity2id, logger, data_type="test")
        num_kb_relation = train_data.num_kb_relation
    else:
        train_data = SingleDataLoader(config, word2id, relation2id, entity2id, logger, data_type="train")
        valid_data = SingleDataLoader(config, word2id, relation2id, entity2id, logger, data_type="dev")
        test_data = SingleDataLoader(config, word2id, relation2id, entity2id, logger, data_type="test")
        num_kb_relation = train_data.num_kb_relation

    dataset = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data,
        "entity2id": entity2id,
        "relation2id": relation2id,
        "word2id": word2id,
        "num_kb_relation":num_kb_relation,
        "test_type_data": test_data_type
    }
    return dataset


if __name__ == "__main__":
    st = time.time()
    args = get_config()
    load_data(args)
