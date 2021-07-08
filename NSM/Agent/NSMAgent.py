import torch
import torch.nn as nn
from torch.autograd import Variable
from NSM.Model.nsm_model import GNNModel
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class NsmAgent(nn.Module):
    def __init__(self, args, logger, num_entity, num_relation, num_word):
        super(NsmAgent, self).__init__()
        self.reset_time = 0
        self.args = args
        self.logger = logger
        self.parse_args(args, num_entity, num_relation, num_word)
        if self.model_name.startswith('gnn'):
            self.model = GNNModel(args, num_entity, num_relation, num_word)
        else:
            raise NotImplementedError
    def parse_args(self, args, num_entity, num_relation, num_word):
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.num_word = num_word
        self.use_inverse_relation = args['use_inverse_relation']
        self.use_self_loop = args['use_self_loop']
        self.logger.info("Entity: {}, Relation: {}, Word: {}".format(num_entity, num_relation, num_word))
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.learning_rate = self.args['lr']
        self.q_type = args['q_type']
        self.num_step = args['num_step']
        self.model_name = args['model_name'].lower()
        self.label_f1 = args['label_f1']

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)


    def forward(self, batch, training=False):
        batch = self.deal_input(batch)
        return self.model(batch, training=training)

    def train_batch(self, batch, middle_dist, label_valid=None):
        batch = self.deal_input(batch)
        return self.model.train_batch(batch, middle_dist, label_valid)

    def deal_input(self, batch):
        return self.deal_input_seq(batch)

    def deal_input_seq(self, batch):
        local_entity, query_entities, kb_adj_mat, query_text, seed_dist, true_batch_id, answer_dist, query_string_list\
            = batch
        local_entity = torch.from_numpy(local_entity).type('torch.LongTensor').to(self.device)
        # local_entity_mask = (local_entity != self.num_entity).float()
        query_entities = torch.from_numpy(query_entities).type('torch.FloatTensor').to(self.device)
        answer_dist = torch.from_numpy(answer_dist).type('torch.FloatTensor').to(self.device)
        seed_dist = torch.from_numpy(seed_dist).type('torch.FloatTensor').to(self.device)
        current_dist = Variable(seed_dist, requires_grad=True)

        query_text = torch.from_numpy(query_text).type('torch.LongTensor').to(self.device)
        query_mask = (query_text != self.num_word).float()

        return current_dist, query_text, query_mask, kb_adj_mat, answer_dist, \
               local_entity, query_entities, true_batch_id, query_string_list
