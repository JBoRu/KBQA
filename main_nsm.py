import argparse
import sys
from NSM.train.trainer_nsm import Trainer_KBQA
from NSM.util.utils import create_logger, init_seed, create_tensorboard
import time
import torch
import os

parser = argparse.ArgumentParser()
# datasets
parser.add_argument('--name', default='webqsp', type=str)
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--data_folder', default=None, type=str)

# embeddings
parser.add_argument('--word2id', default='vocab_new.txt', type=str)
parser.add_argument('--relation2id', default='relations.txt', type=str)
parser.add_argument('--entity2id', default='entities.txt', type=str)
parser.add_argument('--char2id', default='chars.txt', type=str)
parser.add_argument('--entity_emb_file', default=None, type=str)
parser.add_argument('--entity_kge_file', default=None, type=str)
parser.add_argument('--relation_emb_file', default=None, type=str)
parser.add_argument('--relation_kge_file', default=None, type=str)
parser.add_argument('--word_emb_file', default='word_emb_300d.npy', type=str)
parser.add_argument('--rel_word_ids', default='rel_word_idx.npy', type=str)

# GraftNet embeddings
parser.add_argument('--pretrained_entity_kge_file', default='entity_emb_100d.npy', type=str)

# dimensions, layers, dropout
parser.add_argument('--entity_dim', default=100, type=int)
parser.add_argument('--kge_dim', default=100, type=int)
parser.add_argument('--kg_dim', default=100, type=int)
parser.add_argument('--word_dim', default=300, type=int)
parser.add_argument('--lstm_dropout', default=0.3, type=float)
parser.add_argument('--linear_dropout', default=0.2, type=float)

# optimization
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--fact_scale', default=3, type=int)
parser.add_argument('--eval_every', default=5, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--gradient_clip', default=1.0, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--decay_rate', default=0.0, type=float)
parser.add_argument('--seed', default=19960626, type=int)
parser.add_argument('--lr_schedule', action='store_true')
parser.add_argument('--label_smooth', default=0.1, type=float)
parser.add_argument('--fact_drop', default=0, type=float)

# model options
parser.add_argument('--q_type', default='seq', type=str)
parser.add_argument('--share_encoder', action='store_true')
parser.add_argument('--use_inverse_relation', action='store_true')
parser.add_argument('--use_self_loop', action='store_true')
parser.add_argument('--train_KL', action='store_true')
parser.add_argument('--is_eval', action='store_true')
parser.add_argument('--checkpoint_dir', default='checkpoint/', type=str)
parser.add_argument('--log_level', type=str, default='info')
parser.add_argument('--experiment_name', default='debug', type=str)
parser.add_argument('--load_experiment', default=None, type=str)
parser.add_argument('--load_ckpt_file', default=None, type=str)
parser.add_argument('--eps', default=0.05, type=float) # threshold for f1


# RL options
parser.add_argument('--filter_sub', action='store_true')
parser.add_argument('--encode_type', action='store_true')
parser.add_argument('--reason_kb', action='store_true')
parser.add_argument('--num_layer', default=1, type=int)
parser.add_argument('--test_batch_size', default=20, type=int)
parser.add_argument('--num_step', default=1, type=int)
parser.add_argument('--mode', default='teacher', type=str)
parser.add_argument('--entropy_weight', default=0.0, type=float)

parser.add_argument('--use_label', action='store_true')
parser.add_argument('--tree_soft', action='store_true')
parser.add_argument('--filter_label', action='store_true')
parser.add_argument('--share_embedding', action='store_true')
parser.add_argument('--share_instruction', action='store_true')
parser.add_argument('--encoder_type', default='lstm', type=str)
parser.add_argument('--lambda_label', default=0.1, type=float)
parser.add_argument('--lambda_auto', default=0.01, type=float)
parser.add_argument('--label_f1', default=0.5, type=float)
parser.add_argument('--loss_type', default='kl', type=str)
parser.add_argument('--label_file', default=None, type=str)

# New add
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--scheduler_method', default=None, type=str,
                    help="the scheduler method used for optimizer which can be 'ExponentialLR', 'ReduceLROnPlateau'"
                         ", 'LinearWarmUp', 'CosineWarmUp'")
parser.add_argument('--optim', default='Adam', type=str)
parser.add_argument('--initialize_method', default="normal", type=str,
                    help="the initialize method which can be 'normal', 'xavier_uniform'")
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--tb_record', action='store_true',
                    help="Whether use tensorboard to record training process")
parser.add_argument('--question_encoder', default="lstm", type=str,
                    help='the method of encoding question')
parser.add_argument('--early_stop_patience', default=0, type=int)
parser.add_argument('--question_encoding_optim', action='store_true',
                    help='when using lstm encode question, skip the padding token')
parser.add_argument('--update_last_lm_layer', action='store_true',
                    help='update the last layer of pretrained lm')
parser.add_argument('--fix_relation_embedding', action='store_true',
                    help='fix the relation embedding load from pretrained file')
parser.add_argument('--pretrain', action='store_true',
                    help='whether pretain on large datasets')
parser.add_argument('--reason_with_same_param', action='store_true',
                    help='whether use one layer to reason')
parser.add_argument('--add_new_relation_embedding', action='store_true',
                    help='add a new learnable relation embedding to each datasets itself')
parser.add_argument('--finetune_whole', action='store_true',
                    help='finetune the whole params without bert encode relations')
parser.add_argument('--ckpt_4_embedding', type=str,
                    help='load embedding from ckpt')
parser.add_argument('--ckpt_4_pretrain', type=str,
                    help='finetune params from pretrained model')
parser.add_argument('--lm_name', default="bert", type=str,
                    help='the pretrained model name')
parser.add_argument('--finetune_reasoning', action='store_true',
                    help='finetune pretrained reasoning params and initialized other params')
parser.add_argument('--finetune_instruction', action='store_true',
                    help='finetune pretrained instruction params and initialized other params')
parser.add_argument('--finetune_gnn', action='store_true',
                    help='finetune the gnn parameters of instruction module and initialized other params')
parser.add_argument('--finetune_matching', action='store_true',
                    help='finetune the matching parameters of instruction module and initialized other params')
parser.add_argument('--finetune_scoring', action='store_true',
                    help='finetune the scoring parameters and initialized other params')
parser.add_argument('--continue_training', action='store_true',
                    help='load checkpoint model to continue train')
parser.add_argument('--data_split', default=0, type=int)
parser.add_argument('--merge_lr', action='store_true',
                    help='use different lr')
parser.add_argument('--fintune_param_lr', default=0, type=float)
parser.add_argument('--scratch_param_lr', default=0, type=float)
parser.add_argument('--min_lr', default=0, type=float)
parser.add_argument('--filter', action='store_true',
                    help='use filter data')
parser.add_argument('--finetune_only_reasoning', action='store_true',
                    help='only finetune reasoning module')
parser.add_argument('--eval_different_type', action='store_true',
                    help='only finetune reasoning module')


args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

if args.experiment_name == None:
    timestamp = str(int(time.time()))
    args.experiment_name = "{}-{}-{}".format(
        args.dataset,
        args.model_name,
        timestamp,
    )


def main():
    init_seed(args.seed)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    logger = create_logger(args)
    tensorboard = None
    if args.tb_record:
        tensorboard = create_tensorboard(args)
    trainer = Trainer_KBQA(args=vars(args), logger=logger)

    if args.pretrain:
        logger.info("Start pretrain!")
        trainer.pretrain(trainer.start_epoch, args.num_epoch - 1, tensorboard)
    elif args.finetune:
        logger.info("Start finetune!")
        trainer.fintunetrain(trainer.start_epoch, args.num_epoch - 1, tensorboard)
    elif not args.is_eval:
        logger.info("Start train!")
        trainer.train(trainer.start_epoch, args.num_epoch - 1, tensorboard)
    else:
        assert args.load_experiment is not None
        logger.info("Start test!")
        if args.load_experiment is not None:
            ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
            logger.info("Loading pre trained model from {}".format(ckpt_path))
        else:
            ckpt_path = None
        trainer.evaluate_single(ckpt_path)


if __name__ == '__main__':
    main()
