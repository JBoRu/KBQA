import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time
from NSM.Modules.Instruction.base_instruction import BaseInstruction
from transformers import AutoTokenizer, RobertaTokenizer, BertTokenizer
from transformers import BertModel, RobertaModel
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class InstructionWithLSTMEncodeQuestion(BaseInstruction):

    def __init__(self, args, word_embedding, num_word):
        super(InstructionWithLSTMEncodeQuestion, self).__init__(args)
        self.word_embedding = word_embedding
        self.num_word = num_word
        self.encoder_def()
        entity_dim = self.entity_dim
        self.cq_linear = nn.Linear(in_features=2 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_step):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))

    def encoder_def(self):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        kge_dim = self.kge_dim
        entity_dim = self.entity_dim
        self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim,
                                    batch_first=True, bidirectional=False)

    def encode_question(self, query_text):
        batch_size = query_text.size(0)
        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        query_hidden_emb, (h_n, c_n) = self.node_encoder(self.lstm_drop(query_word_emb),
                                                         self.init_hidden(1, batch_size,
                                                                          self.entity_dim))  # 1, batch_size, entity_dim
        self.instruction_hidden = h_n
        self.instruction_mem = c_n
        self.query_node_emb = h_n.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
        self.query_hidden_emb = query_hidden_emb
        self.query_mask = (query_text != self.num_word).float()
        return query_hidden_emb, self.query_node_emb

    def init_reason(self, query_text):
        batch_size = query_text.size(0)
        if self.args["question_encoding_optim"]:
            self.encode_question_optim(query_text)
        else:
            self.encode_question(query_text)
        self.relational_ins = torch.zeros(batch_size, self.entity_dim).to(self.device)
        self.instructions = []
        self.attn_list = []

    def get_instruction(self, relational_ins, step=0, query_node_emb=None):
        query_hidden_emb = self.query_hidden_emb
        query_mask = self.query_mask
        if query_node_emb is None:
            query_node_emb = self.query_node_emb
        relational_ins = relational_ins.unsqueeze(1)
        question_linear = getattr(self, 'question_linear' + str(step))
        q_i = question_linear(self.linear_drop(query_node_emb))
        cq = self.cq_linear(self.linear_drop(torch.cat((relational_ins, q_i), dim=-1)))
        # batch_size, 1, entity_dim
        ca = self.ca_linear(self.linear_drop(cq * query_hidden_emb))
        # batch_size, max_local_entity, 1
        attn_weight = F.softmax(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER, dim=1)
        # batch_size, max_local_entity, 1
        relational_ins = torch.sum(attn_weight * query_hidden_emb, dim=1)
        return relational_ins, attn_weight

    def forward(self, query_text):
        self.init_reason(query_text)
        for i in range(self.num_step):
            relational_ins, attn_weight = self.get_instruction(self.relational_ins, step=i)
            self.instructions.append(relational_ins)
            self.attn_list.append(attn_weight)
            self.relational_ins = relational_ins
        return self.instructions, self.attn_list

    def encode_question_optim(self, query_text):
        batch_size = query_text.size(0)
        self.query_mask = (query_text != self.num_word).float()
        bs_len = self.query_mask.sum(dim=-1)-1 # (batch_size)
        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        query_hidden_emb, (h_n, c_n) = self.node_encoder(self.lstm_drop(query_word_emb),
                                                         self.init_hidden(1, batch_size,
                                                                          self.entity_dim))
        bs_len = bs_len.to(query_word_emb.device).long()
        bs_idx = torch.tensor([i for i in range(batch_size)]).to(query_word_emb.device).long()
        h_n = query_hidden_emb[bs_idx,bs_len,:] # (bs, entity_dim)
        self.instruction_hidden = h_n
        self.instruction_mem = c_n
        self.query_node_emb = h_n.unsqueeze(dim=1) # batch_size, 1, entity_dim
        self.query_hidden_emb = query_hidden_emb

        return query_hidden_emb, self.query_node_emb

class InstructionWithLMEncodeQuestion(BaseInstruction):

    def __init__(self, args, word_embedding, num_word):
        super(InstructionWithLMEncodeQuestion, self).__init__(args)
        self.word_embedding = word_embedding
        self.num_word = num_word
        self.pretrained_lm_def()
        entity_dim = self.entity_dim
        word_dim = self.word_dim
        self.word_embedding_linear = nn.Linear(in_features=word_dim, out_features=entity_dim)
        self.cq_linear = nn.Linear(in_features=2 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_step):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))

    def pretrained_lm_def(self):
        if self.args["lm_name"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.pretrained_model = BertModel.from_pretrained("bert-base-uncased")
        elif self.args["lm_name"] == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            self.pretrained_model = RobertaModel.from_pretrained("roberta-base")
        if self.args["update_last_lm_layer"]:
            for n, p in self.pretrained_model.named_parameters():
                if "layer.11" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            for n, p in self.pretrained_model.named_parameters():
                p.requires_grad = False

    # def pretrained_bert_def(self):
    #     self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    #     self.pretrained_bert = PretrainedBert.from_pretrained('bert-base-uncased')
    #     # fix the pretrained bert model
    #     if self.args["update_last_lm_layer"]:
    #         for n,p in self.pretrained_bert.named_parameters():
    #             if "layer.11" in n:
    #                 p.requires_grad = True
    #             else:
    #                 p.requires_grad = False
    #     else:
    #         for n,p in self.pretrained_bert.named_parameters():
    #             p.requires_grad = False

    # def pretrained_roberta_def(self):
    #     self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    #     self.pretrained_bert = PretrainedBert.from_pretrained('bert-base-uncased')
    #     # fix the pretrained bert model
    #     if self.args["update_last_lm_layer"]:
    #         for n,p in self.pretrained_bert.named_parameters():
    #             if "layer.11" in n:
    #                 p.requires_grad = True
    #             else:
    #                 p.requires_grad = False
    #     else:
    #         for n,p in self.pretrained_bert.named_parameters():
    #             p.requires_grad = False

    def encode_question(self, query_str_list):
        batch_size = len(query_str_list)
        self.query_mask, self.hidden_states = self.encode_question_with_lm(query_str_list)
        self.query_mask = self.query_mask[:, 1:]
        h_n = self.hidden_states[:,0,:] #(bs,entity_dim)
        query_hidden_emb = self.hidden_states[:, 1:, :] # (bs, seq_len, entity_dim)
        self.instruction_hidden = h_n
        self.instruction_mem = h_n
        self.query_node_emb = h_n.unsqueeze(dim=1)  # batch_size, 1, entity_dim
        self.query_hidden_emb = query_hidden_emb
        return query_hidden_emb, self.query_node_emb

    def encode_question_with_lm(self, query_str_list):
        inputs = self.tokenizer(text=query_str_list, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        outputs = self.pretrained_model(**inputs)
        sequence_mask = inputs["attention_mask"]

        sep_idx = sequence_mask.sum(dim=-1).long() - 1  # (bs)
        bs = len(sep_idx)
        bs_idx = torch.tensor([i for i in range(bs)]).long().to(sep_idx.device)
        sequence_mask[bs_idx, sep_idx] = 0

        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = self.word_embedding_linear(last_hidden_state)

        return sequence_mask, last_hidden_state

    def init_reason(self, query_str_list):
        batch_size = len(query_str_list)
        self.encode_question(query_str_list)
        self.relational_ins = torch.zeros(batch_size, self.entity_dim).to(self.device)
        self.instructions = []
        self.attn_list = []

    def get_instruction(self, relational_ins, step=0, query_node_emb=None):
        query_hidden_emb = self.query_hidden_emb # (bs, seq_len, hidden_dim)
        query_mask = self.query_mask # (bs, seq_len)
        if query_node_emb is None:
            query_node_emb = self.query_node_emb # (bs, 1, hidden_dim)
        relational_ins = relational_ins.unsqueeze(1) # (bs,1,hidden_dim)
        question_linear = getattr(self, 'question_linear' + str(step))
        q_i = question_linear(self.linear_drop(query_node_emb)) # (bs, 1, hidden_dim)
        cq = self.cq_linear(self.linear_drop(torch.cat((relational_ins, q_i), dim=-1))) # (bs, 1, hidden_dim)
        # batch_size, 1, entity_dim
        ca = self.ca_linear(self.linear_drop(cq * query_hidden_emb)) # (bs, seq_len, 1)
        # batch_size, max_local_entity, 1
        # cv = self.softmax_d1(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER)
        attn_weight = F.softmax(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER, dim=1) # (bs, seq_len, 1)
        # batch_size, max_local_entity, 1
        relational_ins = torch.sum(attn_weight * query_hidden_emb, dim=1) # (bs, hidden_dim)
        return relational_ins, attn_weight

    def forward(self, query_str_list):
        self.init_reason(query_str_list)
        for i in range(self.num_step):
            relational_ins, attn_weight = self.get_instruction(self.relational_ins, step=i)
            self.instructions.append(relational_ins)
            self.attn_list.append(attn_weight)
            self.relational_ins = relational_ins
        return self.instructions, self.attn_list