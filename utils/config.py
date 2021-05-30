import os
import argparse
import random
import numpy as np

import torch

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0

if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False
MAX_LENGTH = 10

parser = argparse.ArgumentParser(description='Seq_TO_Seq Dialogue')
parser.add_argument('-ds', '--dataset', help='dataset, kvr, woz or cam', default='kvr', required=False)
parser.add_argument('-t', '--task', help='navigate, weather or schedule', required=False, default="")
parser.add_argument('-dec', '--decoder', help='decoder model', required=False)
parser.add_argument('-hdd', '--hidden', help='Hidden size', required=False)
parser.add_argument('-bsz', '--batch', help='Batch_size', required=False)
parser.add_argument('-lr', '--learn', help='Learning Rate', required=False)
parser.add_argument('-dr', '--drop', help='Drop Out', required=False)
parser.add_argument('-um', '--unk_mask', help='mask out input token to UNK', type=int, required=False, default=1)
parser.add_argument('-l', '--layer', help='Layer Number', required=False)
parser.add_argument('-lm', '--limit', help='Word Limit', required=False, default=-10000)
parser.add_argument('-path', '--path', help='path of the file to load', required=False)
parser.add_argument('-clip', '--clip', help='gradient clipping', required=False, default=10)
parser.add_argument('-tfr', '--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False,
                    default=0.5)

parser.add_argument('-m', '--mode', choices=['train', 'prune', 'test'], default='train', help='Run mode')
parser.add_argument('-alp', '--alpha', type=float, default=32, help='help to control expansion')
parser.add_argument('-bet', '--beta', type=float, default=50, help='help to control expansion')
parser.add_argument('-osp', '--one_shot_prune_percentage', type=float, default=0.5,
                    help='% of neurons to prune per module')
parser.add_argument('-seed', '--seed', help='random seed', type=int, required=False, default=3)
parser.add_argument('-evalp', '--evalp', help='evaluation period', required=False, default=1)
parser.add_argument('-gs', '--genSample', help='Generate Sample', required=False, default=0)
parser.add_argument('-rec', '--record', help='use record function during inference', type=int, required=False,
                    default=0)

args = vars(parser.parse_args())
print(str(args))
print("USE_CUDA: " + str(USE_CUDA))


def set_seeds(seed):
    # set all possible seeds
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if args["task"] in ['schedule', 'navigate', 'weather']:
    args["dataset"] = 'kvr'
elif args["task"] in ['restaurant', 'hotel', 'attraction']:
    args["dataset"] = 'woz'
else:
    args["dataset"] = 'cam'

set_seeds(args["seed"])
LIMIT = int(args["limit"])
if args["dataset"] == 'kvr':
    MEM_TOKEN_SIZE = 6
elif args["dataset"] == 'cam':
    MEM_TOKEN_SIZE = 4
else:
    MEM_TOKEN_SIZE = 12
