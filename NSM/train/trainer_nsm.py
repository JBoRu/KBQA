import torch
import time
import numpy as np
import os, math
from NSM.train.init import init_nsm
from NSM.train.evaluate_nsm import Evaluator_nsm
from NSM.data.load_data_super import load_data
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
from functools import reduce
from NSM.Agent.NSMAgent import NsmAgent
import torch.optim as optim
import torch.nn.init as init
tqdm.monitor_iterval = 0



class Trainer_KBQA(object):
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.best_dev_performance = 0.0
        self.best_h1 = 0.0
        self.reset_time = 0
        self.start_epoch = 0
        self.parse_args()
        self.load_data(args)
        self.init_model_and_load_ckpt(self.args)
        self.evaluator = Evaluator_nsm(args=args, student=self.student, entity2id=self.entity2id,
                                       relation2id=self.relation2id, device=self.device)

    def parse_args(self):
        self.scheduler_method = self.args["scheduler_method"]
        self.optim = self.args["optim"]
        self.init_method = self.args["initialize_method"]
        self.learning_rate = self.args['lr']
        self.weight_decay = self.args['weight_decay']
        self.test_batch_size = self.args['test_batch_size']
        self.pretrain_flag = self.args["pretrain"]
        self.device = torch.device('cuda' if self.args['use_cuda'] else 'cpu')
        self.decay_rate = self.args['decay_rate']
        self.mode = self.args["mode"]
        self.model_name = self.args['model_name']

    def init_model_and_load_ckpt(self, args):
        self.logger.info("Building {}.".format("Agent"))
        self.student = NsmAgent(self.args, self.logger, len(self.entity2id), self.num_kb_relation, len(self.word2id))
        self.logger.info("Architecture: {}".format(self.student))

        self.student.to(self.device)
        self.optim_def()
        self.scheduler_def()

        self.logger.info("Optimizer: %s LR: %.5f Scheduler %s InitMethod: %s" % (self.optim, self.learning_rate,
                                                                                 self.scheduler_method,
                                                                                 self.init_method))
        total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0) if w.requires_grad is True else 0
                            for w in self.student.parameters()])
        self.logger.info("Agent trainable params: {}".format(total_params))

    def optim_def(self):
        if self.optim == "AdamW":
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.student.named_parameters() if
                            p.requires_grad and not any(nd in n for nd in no_decay)],
                 'weight_decay': self.weight_decay},
                {'params': [p for n, p in self.student.named_parameters() if
                            p.requires_grad and any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            self.optim_student = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        elif self.optim == "Adam":
            trainable = filter(lambda p: p.requires_grad, self.student.parameters())
            self.optim_student = optim.Adam(trainable, lr=self.learning_rate)

    def scheduler_def(self):
        if self.scheduler_method == "ExponentialLR":
            assert self.decay_rate > 0, "Decay rate must > 0"
            self.scheduler = ExponentialLR(self.optim_student, self.decay_rate)
        # elif self.scheduler_method == "LambdaLR":
        elif self.scheduler_method == "ReduceLROnPlateau":
            assert self.args["linear_decay"] > 0, "Linear decay must > 0"
            self.scheduler = ReduceLROnPlateau(self.optim_student, mode="max", factor=self.args["linear_decay"],
                                               patience=self.args["patience"], min_lr=self.args["min_lr"],
                                               verbose=True)
        elif self.scheduler_method == "LinearWarmUp":
            assert self.args["warm_up_steps"] > 0, "WarmUp steps must > 0"
            warm_up_steps = self.args["warm_up_steps"]
            total_steps = self.args["num_epoch"] * math.ceil(self.train_data.num_data / self.args["batch_size"])
            self.scheduler = get_linear_schedule_with_warmup(self.optim_student, warm_up_steps, total_steps)
        elif self.scheduler_method == "CosineWarmUp":
            assert self.args["warm_up_steps"] > 0, "WarmUp steps must > 0"
            warm_up_steps = self.args["warm_up_steps"]
            total_steps = self.args["num_epoch"] * math.ceil(self.train_data.num_data / self.args["batch_size"])
            self.scheduler = get_cosine_schedule_with_warmup(self.optim_student, warm_up_steps, total_steps)
        else:
            self.scheduler = None

    def load_data(self, args):
        dataset = load_data(args,self.logger)
        self.train_data = dataset["train"]
        self.valid_data = dataset["valid"]
        self.test_data = dataset["test"]
        self.entity2id = dataset["entity2id"]
        self.relation2id = dataset["relation2id"]
        self.word2id = dataset["word2id"]
        self.num_kb_relation = self.test_data.num_kb_relation
        self.num_entity = len(self.entity2id)

    def load_pretrain(self):
        args = self.args
        if args['load_experiment'] is not None:
            ckpt_path = os.path.join(args['checkpoint_dir'], args['load_experiment'])
            print("Load ckpt from", ckpt_path)
            self.load_ckpt(ckpt_path)

    def evaluate(self, data, test_batch_size=20, mode="teacher", write_info=False):
        return self.evaluator.evaluate(data, test_batch_size, write_info)

    def train(self, start_epoch, end_epoch, tensorboard=None):
        eval_every = self.args['eval_every']
        f1, hits1 = self.evaluate(self.valid_data, self.test_batch_size, mode="teacher")
        self.logger.info("Before training: H1 : {:.4f}".format(hits1))
        self.logger.info("Strat Training------------------")

        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            loss, extras, h1_list_all, f1_list_all = self.train_epoch()
            if self.decay_rate > 0:
                self.scheduler.step()
            self.logger.info("Epoch: {}, loss : {:.4f}, time: {}".format(epoch + 1, loss, time.time() - st))
            self.logger.info("Training h1 : {:.4f}, f1 : {:.4f}".format(np.mean(h1_list_all), np.mean(f1_list_all)))
            if (epoch + 1) % eval_every == 0 and epoch + 1 > 0:
                eval_f1, eval_h1 = self.evaluate(self.valid_data, self.test_batch_size, mode="teacher")
                self.logger.info("EVAL F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
                if eval_h1 > self.best_h1:
                    self.best_h1 = eval_h1
                    self.save_ckpt("h1")
                    self.reset_time = 0
                else:
                    self.reset_time += 1
                    self.logger.info('No improvement after {} evaluation iter.'.format(str(self.reset_time)))

                if 0 < self.args["early_stop_patience"] < self.reset_time:
                    self.logger.info('No improvement after {} evaluation. Early Stopping.'.format(self.reset_time))
                    break
        self.save_ckpt("final")
        self.logger.info('Train Done! Evaluate on testset with saved model')
        if self.model_name != "back":
            self.evaluate_best(self.mode)

    def evaluate_best(self, mode):
        filename = os.path.join(self.args['checkpoint_dir'], "{}-h1.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size, mode="teacher", write_info=False)
        self.logger.info("Best h1 evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))

        filename = os.path.join(self.args['checkpoint_dir'], "{}-final.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size, mode="teacher", write_info=False)
        self.logger.info("Final evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))

    def evaluate_single(self, filename):
        if filename is not None:
            self.load_ckpt(filename)
        test_f1, test_hits = self.evaluate(self.test_data, self.test_batch_size, mode=self.mode, write_info=True)
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(test_f1, test_hits))

    def train_epoch(self):
        self.student.train()
        self.train_data.reset_batches(is_sequential=False)
        losses = []
        actor_losses = []
        ent_losses = []
        num_epoch = math.ceil(self.train_data.num_data / self.args['batch_size'])
        h1_list_all = []
        f1_list_all = []
        for iteration in tqdm(range(num_epoch)):
            batch = self.train_data.get_batch(iteration, self.args['batch_size'], self.args['fact_drop'])
            # label_dist, label_valid = self.train_data.get_label()
            # loss = self.train_step_student(batch, label_dist, label_valid)
            self.optim_student.zero_grad()
            loss, _, _, tp_list = self.student(batch, training=True)
            # if tp_list is not None:
            h1_list, f1_list = tp_list
            h1_list_all.extend(h1_list)
            f1_list_all.extend(f1_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for name, param in self.student.named_parameters()],
                                           self.args['gradient_clip'])
            self.optim_student.step()
            losses.append(loss.item())
        extras = [0, 0]
        return np.mean(losses), extras, h1_list_all, f1_list_all

    def save_ckpt(self, reason="h1"):
        model = self.student.model
        checkpoint = {
            'model_state_dict': model.state_dict()
        }
        model_name = os.path.join(self.args['checkpoint_dir'], "{}-{}.ckpt".format(self.args['experiment_name'],
                                                                                   reason))
        torch.save(checkpoint, model_name)
        self.logger.info("Checkpoint {}, save model as {}".format(reason, model_name))

    def load_ckpt(self, filename):
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint["model_state_dict"]
        model = self.student.model
        # model = self.student
        self.logger.info("Load param of {} from {}.".format(", ".join(list(model_state_dict.keys())), filename))
        model.load_state_dict(model_state_dict, strict=False)