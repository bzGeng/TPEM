from utils.utils_general import *
from utils.utils_Ent_woz import read_langs
from utils.config import *


class SplitDataset:
    def __init__(self):
        self.data_format = 'pair_{dataset}_{domain}'
        self.save_format = '../data/split/{domain}.{dataset}.file'


def split_dataset():
    file_train = '../data/MULTIWOZ2.1/train.txt'
    file_dev = '../data/MULTIWOZ2.1/dev.txt'
    file_test = '../data/MULTIWOZ2.1/test.txt'

    pair_train, train_max_len = read_langs(file_train, max_line=None, split=True)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None, split=True)
    pair_test, test_max_len = read_langs(file_test, max_line=None, split=True)
    sd = SplitDataset()
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1
    sd.pair_train_restaurant, sd.pair_train_attraction, sd.pair_train_hotel = split_by_domain(pair_train)
    sd.pair_dev_restaurant, sd.pair_dev_attraction, sd.pair_dev_hotel = split_by_domain(pair_dev)
    sd.pair_test_restaurant, sd.pair_test_attraction, sd.pair_test_hotel = split_by_domain(pair_test)

    if not os.path.exists('../data/split'):
        os.makedirs('../data/split')
    for domain in ['restaurant', 'attraction', 'hotel']:
        for dataset in ['train', 'dev', 'test']:
            torch.save(getattr(sd, sd.data_format.format(dataset=dataset, domain=domain)),
                       sd.save_format.format(domain=domain, dataset=dataset))

    batch_size = 128
    for domain in ['restaurant', 'attraction', 'hotel']:
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
    data_restaurant = []
    data_attraction = []
    data_hotel = []
    for data in data_all:
        if data['domain'] == 'restaurant':
            data_restaurant.append(data)
        elif data['domain'] == 'attraction':
            data_attraction.append(data)
        elif data['domain'] == 'hotel':
            data_hotel.append(data)
        else:
            print("Data lost for not belong to any given domains !")
    return data_restaurant, data_attraction, data_hotel


if __name__ == '__main__':
    split_dataset()