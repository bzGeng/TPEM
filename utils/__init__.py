import os
import pandas as pd
import logging
import torch.nn as nn
import torch
from torch.nn import Parameter
from models.layers import *
from torch import optim


class Optimizers(object):
    def __init__(self):
        self.optimizers = []
        self.lrs = []

    def add(self, optimizer, lr):
        self.optimizers.append(optimizer)
        self.lrs.append(lr)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def __getitem__(self, index):
        return self.optimizers[index]

    def __setitem__(self, index, value):
        self.optimizers[index] = value


class Schedulers(object):
    def __init__(self):
        self.schedulers = []

    def add(self, scheduler):
        self.schedulers.append(scheduler)

    def step(self, res):
        for scheduler in self.schedulers:
            scheduler.step(res)

    def __getitem__(self, index):
        return self.schedulers[index]

    def __setitem__(self, index, value):
        self.schedulers[index] = value


class CsvLogger(object):
    def __init__(self, file_name='logger', resume=False, path='./csvdata/', data_format='csv'):

        self.data_name = os.path.join(path, file_name)
        self.data_path = '{}.csv'.format(self.data_name)
        self.log = None
        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
            else:
                os.remove(self.data_path)
                self.log = pd.DataFrame()
        else:
            self.log = pd.DataFrame()

        self.data_format = data_format

    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.log = self.log.append(df, ignore_index=True)

    def save(self):
        return self.log.to_csv(self.data_path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.data_path
        if os.path.isfile(path):
            self.log = pd.read_csv(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))


def check_resume_folder(path):
    for try_epoch in range(50, 0, -1):
        if os.path.exists('{}/EPOCH{}'.format(path, try_epoch)):
            resume_from_epoch = try_epoch
            break
    path_with_epoch = str(path) + '/EPOCH' + str(resume_from_epoch)
    assert len(os.listdir(path_with_epoch)) == 1
    full_path = path_with_epoch + '/' + str(os.listdir(path_with_epoch)[0])
    return full_path


def load_or_build_masks(masks, model, growing_embeddings_masks):
    module_list = ['encoder', 'extKnow']
    if not masks:
        masks = {
            'encoder': {},
            'extKnow': {}
        }
        for superior_module in module_list:
            named_modules = getattr(model, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableLinear) or isinstance(module, SharableEmbedding):
                    mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                    if 'cuda' in module.weight.data.type():
                        mask = mask.cuda()
                    masks[superior_module][name] = mask
                if isinstance(module, SharableGRU):
                    masks[superior_module][name] = {}
                    for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                        if 'weight' in gru_weight_name:
                            mask = torch.ByteTensor(getattr(module, gru_weight_name).data.size()).fill_(0)
                            if 'cuda' in getattr(module, gru_weight_name).data.type():
                                mask = mask.cuda()
                            masks[superior_module][name][gru_weight_name] = mask
    elif model.mode == 'test':
        for superior_module in module_list:
            named_modules = getattr(model, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableLinear):
                    dim1, dim2 = module.weight.size()
                    mask = masks[superior_module][name][:dim1, :dim2]
                    if 'cuda' in module.weight.data.type():
                        mask = mask.cuda()
                    masks[superior_module][name] = mask
                if isinstance(module, SharableGRU):
                    for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                        if 'weight' in gru_weight_name:
                            dim1, dim2 = getattr(module, gru_weight_name).size()
                            mask = masks[superior_module][name][gru_weight_name][:dim1, :dim2]
                            if 'cuda' in getattr(module, gru_weight_name).data.type():
                                mask = mask.cuda()
                            masks[superior_module][name][gru_weight_name] = mask
                if isinstance(module, SharableEmbedding):
                    mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                    for key, value in model.lang.word2index.items():
                        if key in growing_embeddings_masks[superior_module][name]:
                            mask[value] = growing_embeddings_masks[superior_module][name][key][:model.hidden_size]
                    if 'cuda' in module.weight.data.type():
                        mask = mask.cuda()
                    masks[superior_module][name] = mask
    else:
        for superior_module in module_list:
            named_modules = getattr(model, superior_module).named_modules()
            for name, module in named_modules:
                if isinstance(module, SharableEmbedding):
                    mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                    if 'cuda' in module.weight.data.type():
                        mask = mask.cuda()
                    for key, value in model.lang.word2index.items():
                        if key in growing_embeddings_masks[superior_module][name]:
                            pre_emb_max_size = growing_embeddings_masks[superior_module][name][key].size(0)
                            mask[value][:pre_emb_max_size] = growing_embeddings_masks[superior_module][name][key]
                    masks[superior_module][name] = mask

    return masks


def load_or_build_piggymasks(path, model, train_len):
    continual_learning_info = torch.load(path + '/continual_learning_info.th')
    all_piggymasks = continual_learning_info['piggymasks']
    if model.tasks.index(model.task) > 0:
        if model.task not in all_piggymasks:
            piggymasks = {
                'encoder': {},
                'extKnow': {}
            }
            module_list = ['encoder', 'extKnow']
            params_to_optimize_via_Adam = []
            for superior_module in module_list:
                named_modules = getattr(model, superior_module).named_modules()
                for name, module in named_modules:
                    if isinstance(module, SharableLinear) or isinstance(module, SharableEmbedding):
                        piggymasks[superior_module][name] = torch.zeros_like(model.masks[superior_module][name], dtype=torch.float32)
                        piggymasks[superior_module][name].fill_(0.01)
                        if 'cuda' in module.weight.data.type():
                            piggymasks[superior_module][name] = piggymasks[superior_module][name].cuda()
                        piggymasks[superior_module][name] = Parameter(piggymasks[superior_module][name])
                        module.piggymask = piggymasks[superior_module][name]
                        params_to_optimize_via_Adam.append(module.piggymask)
                    if isinstance(module, SharableGRU):
                        piggymasks[superior_module][name] = {}
                        for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                            if 'weight' in gru_weight_name:
                                piggymasks[superior_module][name][gru_weight_name] = torch.zeros_like(
                                    model.masks[superior_module][name][gru_weight_name], dtype=torch.float32)
                                piggymasks[superior_module][name][gru_weight_name].fill_(0.01)
                                if 'cuda' in getattr(module, gru_weight_name).data.type():
                                    piggymasks[superior_module][name][gru_weight_name] = piggymasks[superior_module][name][gru_weight_name].cuda()
                                piggymasks[superior_module][name][gru_weight_name] = Parameter(piggymasks[superior_module][name][gru_weight_name])
                                module.piggymasks_weights[gru_weight_name+'_piggymask'] = piggymasks[superior_module][name][gru_weight_name]
                                params_to_optimize_via_Adam.append(module.piggymasks_weights[gru_weight_name+'_piggymask'])
            all_piggymasks[model.task] = piggymasks
            model.piggymask_optimizer = optim.Adam(params_to_optimize_via_Adam, lr=5e-5)
            model.piggymask_scheduler = get_cosine_schedule_with_stop(model.piggymask_optimizer, train_len * 1,
                                                                      train_len * 15, train_len * 30)
            return all_piggymasks
        else:
            piggymasks = all_piggymasks[model.task]
            module_list = ['encoder', 'extKnow']
            params_to_optimize_via_Adam = []
            for superior_module in module_list:
                named_modules = getattr(model, superior_module).named_modules()
                for name, module in named_modules:
                    if isinstance(module, SharableLinear) or isinstance(module, SharableEmbedding):
                        piggymask = piggymasks[superior_module][name]
                        if 'cuda' in module.weight.data.type():
                            piggymask = piggymask.cuda()
                        module.piggymask = piggymask
                        params_to_optimize_via_Adam.append(module.piggymask)
                    if isinstance(module, SharableGRU):
                        for gru_weight_name in [p for layerparams in module._all_weights for p in layerparams]:
                            if 'weight' in gru_weight_name:
                                piggymask = piggymasks[superior_module][name][gru_weight_name]
                                if 'cuda' in getattr(module, gru_weight_name).data.type():
                                    piggymask = piggymask.cuda()
                                module.piggymasks_weights[gru_weight_name + '_piggymask'] = piggymask
                                params_to_optimize_via_Adam.append(
                                    module.piggymasks_weights[gru_weight_name + '_piggymask'])
            all_piggymasks[model.task] = piggymasks
            return all_piggymasks
    else:
        return {}


def info_reload(path):
    if path:
        continual_learning_info = torch.load(path + '/continual_learning_info.th')
        task_history = continual_learning_info['task_history']
        hidden_sizes = continual_learning_info['hidden_sizes']
        masks = continual_learning_info['masks']
        task_specific_bias = continual_learning_info['tasks_specific_bias']
        tasks_specific_decoders = continual_learning_info['tasks_specific_decoders']
        langs = continual_learning_info['langs']
        growing_embeddings = continual_learning_info['growing_embeddings']
        growing_embeddings_masks = continual_learning_info['growing_embeddings_masks']
        free_ratio = continual_learning_info['free_ratio']
    else:
        task_history = []
        hidden_sizes = []
        masks = {}
        task_specific_bias = {}
        tasks_specific_decoders = {}
        langs = {}
        growing_embeddings = {}
        growing_embeddings_masks = {}
        free_ratio = 1.0

    return task_history, hidden_sizes, masks, task_specific_bias, tasks_specific_decoders, langs, growing_embeddings, growing_embeddings_masks, free_ratio


def get_cosine_schedule_with_stop(
        optimizer: torch.optim.Optimizer, num_warmup_steps: int, stop_steps: int, num_training_steps: int, num_cycles: float = 0.5,
        last_epoch: int = -1
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > stop_steps:
            return 0.0
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_optimize_parameters(parameters):
    params_to_optimize = []
    for name, param in dict(parameters).items():
        if 'piggy' in name:
            continue
        else:
            params_to_optimize.append(param)

    return params_to_optimize


def set_logger(filepath):
    global logger
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    _format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return
