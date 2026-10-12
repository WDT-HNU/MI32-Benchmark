import argparse
import random
import datetime
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from Datasets import seedv_dataset
from datasets import (faced_dataset, seedv_dataset, shu_dataset, isruc1_dataset, isruc3_dataset, \
                      chb_dataset, speech_dataset, stress_dataset, tuev_dataset, tuab_dataset)
from Downstream.finetune_trainer import Trainer
from models import (model_for_faced, model_for_seedv, model_for_shu, model_for_isruc, model_for_chb, \
    model_for_speech, model_for_stress, model_for_tuev, model_for_tuab)

def main():
    parser = argparse.ArgumentParser(description='CodeBrain Downstream')
    parser.add_argument('--seed', type=int, default = 42, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default = 4, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default = 50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default = 64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default = 1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default = 5e-4, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default = 5, help='clip_value')
    parser.add_argument('--dropout', type=float, default = 0.1, help='dropout')
    parser.add_argument('--n_layer', type = int, default = 8, help='n_layer')

    parser.add_argument('--downstream_dataset', type=str, default='SEED-V',
                        help='[FACED, SEED-V, SHU-MI, ISRUC_S1, ISRUC_S3'
                             'CHB-MIT, BCIC2020-3, MentalArithmetic, TUEV, TUAB]')
    parser.add_argument('--datasets_dir', type=str,
                        default='',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default = 5, help='number of classes')
    parser.add_argument('--model_dir', type=str,
                        default='',
                        help='model_dir')
    parser.add_argument('--log_dir', type=str,
                        default='',
                        help='log_dir')

    parser.add_argument('--num_workers', type=int, default = 16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--frozen', type = bool,
                        default = False, help='frozen')
    parser.add_argument('--use_pretrained_weights', type = bool,
                        default = True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        default='',
                        help='foundation_dir')

    parser.add_argument('--codebook_size_t', default = 4096, type = int,
                        help = 'number of temporal codebook (default: 4096)')
    parser.add_argument('--codebook_size_f', default = 4096, type = int,
                        help = 'number of frequency codebook (default: 4096)')
    parser.add_argument('--codebook_dim', default = 32, type = int,
                        help = 'dimention of codebook (default: 32)')

    params = parser.parse_args()
    params.model_dir = params.model_dir + params.downstream_dataset + '/'
    params.log_dir = params.log_dir + params.downstream_dataset + '/'
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    params.file_name = str(params.log_dir) + str(current_time) + "_" + str(params.cuda) + ".txt"
    print(params)
    with open(params.file_name, "a") as file:
        file.write(str(params) + "\n")

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    with open(params.file_name, "a") as file:
        file.write('The downstream dataset is {}'.format(params.downstream_dataset))
        file.write("\n")
    if params.downstream_dataset == 'SEED-V':
        params.num_of_classes = 5
        load_dataset = seedv_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_seedv.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'FACED':
        params.num_of_classes = 9
        load_dataset = faced_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_faced.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'SHU-MI':
        params.num_of_classes = 2
        load_dataset = shu_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_shu.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'ISRUC_S1':
        params.num_of_classes = 5
        load_dataset = isruc1_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_isruc.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'ISRUC_S3':
        params.num_of_classes = 5
        load_dataset = isruc3_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_isruc.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'CHB-MIT':
        params.num_of_classes = 2
        load_dataset = chb_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_chb.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'BCIC2020-3':
        params.num_of_classes = 5
        load_dataset = speech_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_speech.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'MentalArithmetic':
        params.num_of_classes = 2
        load_dataset = stress_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_stress.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()
    elif params.downstream_dataset == 'TUEV':
        params.num_of_classes = 6
        load_dataset = tuev_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_tuev.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_multiclass()
    elif params.downstream_dataset == 'TUAB':
        params.num_of_classes = 2
        load_dataset = tuab_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()
        model = model_for_tuab.Model(params)
        t = Trainer(params, data_loader, model)
        t.train_for_binaryclass()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
