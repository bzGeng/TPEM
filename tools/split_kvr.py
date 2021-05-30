from utils.utils_general import *
from utils.utils_Ent_kvr import read_langs
import json
from utils.config import *
import ast
import sys


class SplitDataset:
    def __init__(self):
        self.data_format = 'pair_{dataset}_{domain}'
        self.save_format = '../data/split/{domain}.{dataset}.file'


def split_dataset():
    file_train = '../data/KVR/train.txt'
    file_dev = '../data/KVR/dev.txt'
    file_test = '../data/KVR/test.txt'

    pair_train, train_max_len = read_langs(file_train, max_line=None, split=True)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None, split=True)
    pair_test, test_max_len = read_langs(file_test, max_line=None, split=True)
    sd = SplitDataset()
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1
    sd.pair_train_navigate, sd.pair_train_schedule, sd.pair_train_weather = split_by_domain(pair_train)
    sd.pair_dev_navigate, sd.pair_dev_schedule, sd.pair_dev_weather = split_by_domain(pair_dev)
    sd.pair_test_navigate, sd.pair_test_schedule, sd.pair_test_weather = split_by_domain(pair_test)

    if not os.path.exists('../data/split'):
        os.makedirs('../data/split')
    for domain in ['navigate', 'schedule', 'weather']:
        for dataset in ['train', 'dev', 'test']:
            torch.save(getattr(sd, sd.data_format.format(dataset=dataset, domain=domain)),
                       sd.save_format.format(domain=domain, dataset=dataset))

    batch_size = 128
    for domain in ['navigate', 'schedule', 'weather']:
        lang = Lang()
        train = get_seq(getattr(sd, sd.data_format.format(dataset='train', domain=domain)), lang, batch_size, True)
        dev = get_seq(getattr(sd, sd.data_format.format(dataset='train', domain=domain)), lang, batch_size, False)
        test = get_seq(getattr(sd, sd.data_format.format(dataset='train', domain=domain)), lang, batch_size, False)
        torch.save(lang, sd.save_format.format(dataset='lang', domain=domain))
        print("Read {} sentence pairs train in domain {}" .format(len(getattr(sd, sd.data_format.format(dataset='train', domain=domain))), domain))
        print("Read {} sentence pairs dev in domain {}".format(len(getattr(sd, sd.data_format.format(dataset='dev', domain=domain))), domain))
        print("Read {} sentence pairs test in domain {}".format(len(getattr(sd, sd.data_format.format(dataset='test', domain=domain))), domain))
        print("Vocab_size for domain {}: {} ".format(domain, lang.n_words))


def split_by_domain(data_all):
    data_navigate = []
    data_schedule = []
    data_weather = []
    for data in data_all:
        if data['domain'] == 'navigate':
            data_navigate.append(data)
        elif data['domain'] == 'schedule':
            data_schedule.append(data)
        elif data['domain'] == 'weather':
            data_weather.append(data)
        else:
            print("Data lost for not belong to any given domains !")
    return data_navigate, data_schedule, data_weather


if __name__ == '__main__':
    split_dataset()