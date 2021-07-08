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
        if args['finetune_whole']:
            self.load_whole()
        elif args['finetune_reasoning']:
            finetune_keys = self.load_reasoning()
        elif args["finetune_instruction"]:
            finetune_keys = self.load_instruction()
        elif args["finetune_gnn"]:
            finetune_keys = self.load_reasoning_weight()
        elif args["finetune_only_reasoning"]:
            finetune_keys = self.only_load_reasoning()
        elif args["finetune_matching"]:
            finetune_keys = self.load_matching_weight()
        elif args["finetune_scoring"]:
            finetune_keys = self.load_scoring_weight()
        # else:
        #     self.student = NsmAgent(self.args, self.logger, len(self.entity2id), self.num_kb_relation, len(self.word2id))
        # self.logger.info("Architecture: {}".format(self.student))
        if self.args["merge_lr"]:
            self.student.to(self.device)
            self.difference_optim_def(finetune_keys)
            self.difference_scheduler_def()
        else:
            self.student.to(self.device)
            self.optim_def()
            self.scheduler_def()

        if args["continue_training"]:
            self.load_ckpt_to_continue_training()

        self.logger.info("Optimizer: %s LR: %.5f Scheduler %s InitMethod: %s" % (self.optim, self.learning_rate,
                                                                                 self.scheduler_method,
                                                                                 self.init_method))
        total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0) if w.requires_grad is True else 0
                            for w in self.student.parameters()])

        model_info = "\n"
        for n, p in self.student.named_parameters():
            # line = str(n) + "---" + "True" if p.requires_grad else "False" + "\n"
            line = str(n) + "\t" + str(p.requires_grad) + "\n"
            model_info = model_info + line
        self.logger.info("Architecture: {}".format(model_info))
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
        self.num_kb_relation = dataset["num_kb_relation"]
        self.test_type_data = dataset["test_type_data"]
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
                eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size, mode="teacher")
                self.logger.info("Test F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
                eval_f1, eval_h1 = self.evaluate(self.valid_data, self.test_batch_size, mode="teacher")
                self.logger.info("EVAL F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
                if eval_h1 > self.best_h1:
                    self.best_h1 = eval_h1
                    self.save_ckpt(epoch, loss, eval_h1, "h1")
                    self.reset_time = 0
                else:
                    self.reset_time += eval_every
                    self.logger.info('No improvement after {} epoch iter.'.format(str(self.reset_time)))

                if 0 < self.args["early_stop_patience"] < self.reset_time:
                    self.logger.info('No improvement after {} evaluation. Early Stopping.'.format(self.reset_time))
                    break
        self.save_ckpt(epoch, loss, eval_h1, "final")
        self.logger.info('Train Done! Evaluate on testset with saved model')
        if self.model_name != "back":
            self.evaluate_best(self.mode)

    def pretrain(self, start_epoch, end_epoch, tensorboard):
        eval_every = self.args['eval_every']
        # avoid warning!
        self.optim_student.zero_grad()
        # self.optim_student.step()
        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            loss, extras, h1_list_all, f1_list_all = self.train_epoch()
            h1 = np.mean(h1_list_all)
            f1 = np.mean(f1_list_all)
            self.logger.info("Epoch: {}, loss : {:.4f}, time: {}".format(epoch + 1, loss, time.time() - st))
            self.logger.info("Training h1 : {:.4f}, f1 : {:.4f}".format(h1, f1))

            # if self.scheduler_method == "ExponentialLR":
            #     lr = self.optim_student.param_groups[0]['lr']
            #     self.logger.info("Epoch: {}, lr : {:.4f}".format(epoch + 1, lr))
            #     if lr > self.args['min_lr']:
            #         self.scheduler.step()
            #     if self.args["record"] and tensorboard is not None:
            #         self.global_lr_steps += 1
            #         tensorboard.add_scalar("Train/lr", lr, global_step=self.global_lr_steps)
            # elif self.scheduler_method == "ReduceLROnPlateau":
            #     self.scheduler.step(h1)
            #     if self.args["record"] and tensorboard is not None:
            #         lr = self.optim_student.param_groups[0]['lr']
            #         self.global_lr_steps += 1
            #         tensorboard.add_scalar("Train/lr", lr, global_step=self.global_lr_steps)

            if (epoch + 1) % eval_every == 0 and epoch + 1 > 0:
                eval_f1, eval_h1 = self.evaluate(self.valid_data, self.test_batch_size, mode="teacher")
                self.logger.info("EVAL F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
                if eval_h1 > self.best_h1:
                    self.best_h1 = eval_h1
                    self.save_ckpt(epoch, loss, eval_h1, "h1")
                    self.reset_time = 0
                else:
                    self.reset_time += eval_every
                    self.logger.info('No improvement after {} epoch iter.'.format(str(self.reset_time)))
            #
            # name = "pretrain_" + str(epoch)
            # self.save_ckpt(epoch=epoch, h1=h1, loss=loss)

            # if h1 > self.best_h1:
            #     self.best_h1 = h1
            #     self.save_ckpt(epoch, loss, h1, "h1")
            #     self.reset_time = 0
            # else:
            #     self.reset_time += 1
            #     self.logger.info('No improvement after %d evaluation iter.' % (self.reset_time))
        self.logger.info('Pertrain Done!')

    def fintunetrain(self, start_epoch, end_epoch, tensorboard=None):
        eval_every = self.args['eval_every']
        f1, hits1 = self.evaluate(self.valid_data, self.test_batch_size, mode="teacher")
        self.logger.info("Before fintune training: H1 : {:.4f}".format(hits1))
        self.logger.info("Strat fintune training------------------")

        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            loss, extras, h1_list_all, f1_list_all = self.train_epoch()
            self.update_lr(epoch)
            self.logger.info("Epoch: {}, loss : {:.4f}, time: {}".format(epoch, loss, time.time() - st))
            self.logger.info("Training h1 : {:.4f}, f1 : {:.4f}".format(np.mean(h1_list_all), np.mean(f1_list_all)))
            if (epoch + 1) % eval_every == 0:
                eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size, mode="teacher")
                self.logger.info("Test F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
                eval_f1, eval_h1 = self.evaluate(self.valid_data, self.test_batch_size, mode="teacher")
                self.logger.info("EVAL F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
                if eval_h1 > self.best_h1:
                    self.best_h1 = eval_h1
                    self.save_ckpt(epoch, loss, eval_h1, "h1")
                    self.reset_time = 0
                else:
                    self.reset_time += eval_every
                    self.logger.info('No improvement after {} epoch iter.'.format(str(self.reset_time)))

                if 0 < self.args["early_stop_patience"] < self.reset_time:
                    self.logger.info('No improvement after {} evaluation. Early Stopping.'.format(self.reset_time))
                    break
        self.save_ckpt(epoch, loss, eval_h1, "final")
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
        if self.args["eval_different_type"]:
            for type, data in self.test_type_data.items():
                test_f1, test_hits = self.evaluate(data, self.test_batch_size, mode=self.mode,
                                                   write_info=True)
                self.logger.info("Type: {} F1: {:.4f}, H1: {:.4f}".format(type, test_f1, test_hits))

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
            if self.args["merge_lr"]:
                self.optim_student_scratch.zero_grad()
                self.optim_student_finetune.zero_grad()
            else:
                self.optim_student.zero_grad()
            loss, _, _, tp_list = self.student(batch, training=True)
            # if tp_list is not None:
            h1_list, f1_list = tp_list
            h1_list_all.extend(h1_list)
            f1_list_all.extend(f1_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for name, param in self.student.named_parameters()],
                                           self.args['gradient_clip'])
            if self.args["merge_lr"]:
                self.optim_student_scratch.step()
                self.optim_student_finetune.step()
            else:
                self.optim_student.step()
            losses.append(loss.item())
        extras = [0, 0]
        return np.mean(losses), extras, h1_list_all, f1_list_all

    def save_ckpt(self, epoch, loss, h1, reason="h1"):
        model = self.student.model
        if self.args["merge_lr"]:
            checkpoint = {
                'epoch': epoch,
                'h1': h1,
                'loss': loss,
                'scratch_optimizer_state_dict': self.optim_scratch_scheduler.state_dict(),
                'finetune_optimizer_state_dict': self.optim_finetune_scheduler.state_dict(),
                'model_state_dict': model.state_dict(),
                'finetune_scheduler_state_dict': self.optim_finetune_scheduler.state_dict()
                if self.optim_finetune_scheduler is not None else None,
                'scratch_scheduler_state_dict': self.optim_scratch_scheduler.state_dict()
                if self.optim_scratch_scheduler is not None else None
            }
        else:
            checkpoint = {
                'epoch': epoch,
                'h1': h1,
                'loss': loss,
                'optimizer_state_dict': self.optim_student.state_dict(),
                'model_state_dict': model.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None
            }
        if self.args["pretrain"]:
            model_name = os.path.join(self.args['checkpoint_dir'], "{}-{}.ckpt".format(self.args['experiment_name'],
                                                                                       epoch))
        else:
            model_name = os.path.join(self.args['checkpoint_dir'], "{}-{}.ckpt".format(self.args['experiment_name'],
                                                                                   reason))
        torch.save(checkpoint, model_name)
        self.logger.info("Checkpoint {}, save model as {}".format(epoch, model_name))

    def load_ckpt(self, filename):
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint["model_state_dict"]
        model = self.student.model
        # model = self.student
        # self.logger.info("Load param of {} from {}.".format(", ".join(list(model_state_dict.keys())), filename))
        self.logger.info("Load param from {}.".format(filename))
        model.load_state_dict(model_state_dict, strict=False)

    def load_whole(self):
        embedding_params_fn = os.path.join(self.args['checkpoint_dir'], self.args['ckpt_4_embedding'])
        reasoning_param_fn = os.path.join(self.args['checkpoint_dir'], self.args['ckpt_4_pretrain'])
        cwq = torch.load(embedding_params_fn)
        cwq_state_dict = cwq['model_state_dict']
        cfq = torch.load(reasoning_param_fn)
        cfq_state_dict = cfq['model_state_dict']

        model_dict = self.student.model.state_dict()
        model_dict_keys = set(model_dict.keys())
        embedding = {k: v for k, v in cwq_state_dict.items() if "relation_embedding" in k}
        reasoning = {k: v for k, v in cfq_state_dict.items() if "relation_embedding" not in k and k in model_dict_keys}
        model_dict.update(embedding)
        model_dict.update(reasoning)
        self.student.model.load_state_dict(model_dict, strict=False)
        # if self.args["fix_reasoning"]:
        #     for k, v in self.student.model.named_parameters():
        #         if k in reasoning.keys():
        #             v.requires_grad = False
        if self.args["fix_relation_embedding"]:
            for k, v in self.student.model.named_parameters():
                if k in embedding.keys():
                    v.requires_grad = False

    def load_reasoning(self):
        embedding_params_fn = os.path.join(self.args['checkpoint_dir'], self.args['ckpt_4_embedding'])
        reasoning_param_fn = os.path.join(self.args['checkpoint_dir'], self.args['ckpt_4_pretrain'])
        cwq = torch.load(embedding_params_fn)
        cwq_state_dict = cwq['model_state_dict']
        cfq = torch.load(reasoning_param_fn)
        cfq_state_dict = cfq['model_state_dict']

        model_dict = self.student.model.state_dict()
        model_dict_keys = set(model_dict.keys())
        embedding = {k: v for k, v in cwq_state_dict.items() if "relation_embedding" in k}
        reasoning = {k: v for k, v in cfq_state_dict.items() if "reasoning" in k and k in model_dict_keys}
        model_dict.update(embedding)
        model_dict.update(reasoning)
        self.student.model.load_state_dict(model_dict, strict=False)
        if self.args["fix_relation_embedding"]:
            for k, v in self.student.model.named_parameters():
                if k in embedding.keys():
                    v.requires_grad = False
        return reasoning.keys()

    def load_instruction(self):
        embedding_params_fn = os.path.join(self.args['checkpoint_dir'], self.args['ckpt_4_embedding'])
        reasoning_param_fn = os.path.join(self.args['checkpoint_dir'], self.args['ckpt_4_pretrain'])
        cwq = torch.load(embedding_params_fn)
        cwq_state_dict = cwq['model_state_dict']
        cfq = torch.load(reasoning_param_fn)
        cfq_state_dict = cfq['model_state_dict']

        model_dict = self.student.model.state_dict()
        model_dict_keys = set(model_dict.keys())
        embedding = {k: v for k, v in cwq_state_dict.items() if "relation_embedding" in k}
        instruction = {k: v for k, v in cfq_state_dict.items() if ("instruction" in k and "pretrained" not in k)
                                                                    or ("encoder.layer.11" in k)}
        model_dict.update(embedding)
        model_dict.update(instruction)
        self.student.model.load_state_dict(model_dict, strict=False)
        if self.args["fix_relation_embedding"]:
            for k, v in self.student.model.named_parameters():
                if k in embedding.keys():
                    v.requires_grad = False
        return instruction.keys()

    def load_ckpt_to_continue_training(self):
        ckpt_path = os.path.join(self.args["checkpoint_dir"], self.args["load_experiment"])
        checkpoint = torch.load(ckpt_path)

        model_state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint["epoch"]
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
        scheduler_state_dict = checkpoint["scheduler_state_dict"]
        model = self.student.model
        model.load_state_dict(model_state_dict, strict=False)
        self.start_epoch = epoch+1
        self.optim_student.load_state_dict(optimizer_state_dict)
        if self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state_dict)

        self.logger.info("Loading checkpoint model from {} to continue training".format(ckpt_path))

    def difference_optim_def(self, finetune_keys):
        no_decay = ['bias', 'LayerNorm.weight']
        optim_group_scratch = []
        optim_group_finetune = []
        for k, v in self.student.model.named_parameters():
            pl = {}
            if v.requires_grad:
                if k in finetune_keys:
                    pl['params'] = v
                    pl['lr'] = self.args["fintune_param_lr"]
                    optim_group_finetune.append(pl)
                else:
                    pl['params'] = v
                    pl['lr'] = self.args["scratch_param_lr"]
                    optim_group_scratch.append(pl)

        self.optim_student_scratch = optim.Adam(optim_group_scratch)
        self.optim_student_finetune = optim.Adam(optim_group_finetune)

    def difference_scheduler_def(self):
        if self.scheduler_method == "ExponentialLR":
            assert self.decay_rate > 0, "Decay rate must > 0"
            self.optim_scratch_scheduler = ExponentialLR(self.optim_student_scratch, self.decay_rate)
            self.optim_finetune_scheduler = ExponentialLR(self.optim_student_finetune, self.decay_rate)
        else:
            self.scheduler = None
            self.logger.info("No scheduler!")

    def update_lr(self,epoch):
        if self.decay_rate > 0:
            if self.args["merge_lr"]:
                self.optim_scratch_scheduler.step()
                slr = self.optim_student_scratch.param_groups[0]["lr"]
                self.logger.info("Epoch: {}, Scratch lr : {:.6f}".format(epoch + 1, slr))

                self.optim_finetune_scheduler.step()
                flr = self.optim_student_finetune.param_groups[0]["lr"]
                self.logger.info("Epoch: {}, Fintune lr : {:.6f}".format(epoch + 1, flr))
                # if slr < self.args['min_lr']:
                #     for param_group in self.optim_student_scratch.param_groups:
                #         param_group['lr'] = self.args['min_lr']
                #     slr = self.optim_student_scratch.param_groups[0]["lr"]
                #     self.logger.info("After Epoch: {}, keep min lr : {:.4f}".format(epoch, slr))
                #     self.decay_rate = 0
            else:
                self.scheduler.step()

    def only_load_reasoning(self):
        reasoning_param_fn = os.path.join(self.args['checkpoint_dir'], self.args['ckpt_4_pretrain'])
        cfq = torch.load(reasoning_param_fn)
        cfq_state_dict = cfq['model_state_dict']

        model_dict = self.student.model.state_dict()
        model_dict_keys = set(model_dict.keys())
        reasoning = {k: v for k, v in cfq_state_dict.items() if "reasoning" in k and k in model_dict_keys}
        model_dict.update(reasoning)
        self.student.model.load_state_dict(model_dict, strict=False)
        return reasoning.keys()

    def load_reasoning_weight(self):
        embedding_params_fn = os.path.join(self.args['checkpoint_dir'], self.args['ckpt_4_embedding'])
        reasoning_param_fn = os.path.join(self.args['checkpoint_dir'], self.args['ckpt_4_pretrain'])
        cwq = torch.load(embedding_params_fn)
        cwq_state_dict = cwq['model_state_dict']
        cfq = torch.load(reasoning_param_fn)
        cfq_state_dict = cfq['model_state_dict']

        model_dict = self.student.model.state_dict()
        model_dict_keys = set(model_dict.keys())
        embedding = {k: v for k, v in cwq_state_dict.items() if "relation_embedding" in k}
        reasoning = {k: v for k, v in cfq_state_dict.items() if ("rel_linear" in k or "e2e_linear" in k)
                                                                and k in model_dict_keys}
        model_dict.update(embedding)
        model_dict.update(reasoning)
        self.student.model.load_state_dict(model_dict, strict=False)
        if self.args["fix_relation_embedding"]:
            for k, v in self.student.model.named_parameters():
                if k in embedding.keys():
                    v.requires_grad = False

        model_info = "\n"
        for k, v in reasoning.items():
            line = k + "\n"
            model_info = model_info + line
        self.logger.info("Load gnn weight: {} from {}".format(model_info, reasoning_param_fn))

        return reasoning.keys()

    def load_matching_weight(self):
        embedding_params_fn = os.path.join(self.args['checkpoint_dir'], self.args['ckpt_4_embedding'])
        reasoning_param_fn = os.path.join(self.args['checkpoint_dir'], self.args['ckpt_4_pretrain'])
        cwq = torch.load(embedding_params_fn)
        cwq_state_dict = cwq['model_state_dict']
        cfq = torch.load(reasoning_param_fn)
        cfq_state_dict = cfq['model_state_dict']

        model_dict = self.student.model.state_dict()
        model_dict_keys = set(model_dict.keys())
        embedding = {k: v for k, v in cwq_state_dict.items() if "relation_embedding" in k}
        reasoning = {k: v for k, v in cfq_state_dict.items() if "relation_linear" in k and k in model_dict_keys}
        model_dict.update(embedding)
        model_dict.update(reasoning)
        self.student.model.load_state_dict(model_dict, strict=False)
        if self.args["fix_relation_embedding"]:
            for k, v in self.student.model.named_parameters():
                if k in embedding.keys():
                    v.requires_grad = False

        model_info = "\n"
        for k, v in reasoning.items():
            line = k + "\n"
            model_info = model_info + line
        self.logger.info("Load gnn weight: {} from {}".format(model_info, reasoning_param_fn))

        return reasoning.keys()

    def load_scoring_weight(self):
        embedding_params_fn = os.path.join(self.args['checkpoint_dir'], self.args['ckpt_4_embedding'])
        reasoning_param_fn = os.path.join(self.args['checkpoint_dir'], self.args['ckpt_4_pretrain'])
        cwq = torch.load(embedding_params_fn)
        cwq_state_dict = cwq['model_state_dict']
        cfq = torch.load(reasoning_param_fn)
        cfq_state_dict = cfq['model_state_dict']

        model_dict = self.student.model.state_dict()
        model_dict_keys = set(model_dict.keys())
        embedding = {k: v for k, v in cwq_state_dict.items() if "relation_embedding" in k}
        reasoning = {k: v for k, v in cfq_state_dict.items() if "score_func" in k and k in model_dict_keys}
        model_dict.update(embedding)
        model_dict.update(reasoning)
        self.student.model.load_state_dict(model_dict, strict=False)
        if self.args["fix_relation_embedding"]:
            for k, v in self.student.model.named_parameters():
                if k in embedding.keys():
                    v.requires_grad = False

        model_info = "\n"
        for k, v in reasoning.items():
            line = k + "\n"
            model_info = model_info + line
        self.logger.info("Load gnn weight: {} from {}".format(model_info, reasoning_param_fn))

        return reasoning.keys()
