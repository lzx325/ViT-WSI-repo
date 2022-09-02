from __future__ import print_function

import argparse
import os
import math
from os.path import join,basename,splitext,dirname
from datetime import datetime as dt
from pprint import pprint
from copy import copy
import ipdb
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
import h5py
# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train,eval
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, Generic_SlideGraph_Dataset
from utils.utils import parse_dict_args,str2bool
from utils.config_utils import parse_args

def seed_all(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_dataset(args):
    print('\nLoad Dataset')

    if args.model_type=="graph_vit_aggr":
        DatasetClass=Generic_SlideGraph_Dataset
    else:
        DatasetClass=Generic_MIL_Dataset

    print("dataset class is:", DatasetClass)

    if args.task == 'task_1_tumor_vs_normal':
        args.n_classes=2
        dataset = DatasetClass(csv_path = args.info_csv_path,
                                data_dir = args.features_dir,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict= args.label_dict,
                                patient_strat=False,
                                ignore=[])

    elif args.task == 'task_2_tumor_subtyping':
        args.n_classes=len(args.label_dict)

        dataset = DatasetClass(csv_path = args.info_csv_path,
                                data_dir = args.features_dir,
                                shuffle = False, 
                                seed = args.seed, 
                                print_info = True,
                                label_dict = args.label_dict,
                                patient_strat= False,
                                ignore=[])

        if args.model_type in ['clam_sb', 'clam_mb']:
            assert args.subtyping 
            
    else:
        raise NotImplementedError
    
    return dataset

def check_and_create_dirs(args):
    # check dirs
    print(args.split_dir)
    assert os.path.isdir(args.split_dir)
    assert os.path.isdir(args.features_dir)

    if args.mode=="train":
        # prepare dirs
        date_str=dt.now().strftime('%Y-%m-%d-%H:%M:%S')
        assert args.fold is not None
        args.train_dir = os.path.join(args.train_root_dir, str(args.exp_code), f"fold-{args.fold}" , date_str)
        args.ckpt_dir=join(args.train_dir,"ckpt")
        args.tensorboard_dir=join(args.train_dir,"tensorboard")
        args.history_dir=join(args.train_dir,"history")
        args.vis_dir=join(args.history_dir,"vis")
        args.tb_data_dir=join(args.history_dir,"tb_data")
        args.config_dir=join(args.train_dir,"config")
        args.results_dir=join(args.train_dir,"results")

        dir_keys=["train_dir","ckpt_dir","tensorboard_dir","history_dir","vis_dir","tb_data_dir","config_dir","results_dir"]

        for dk in dir_keys:
            dp=getattr(args,dk)
            os.makedirs(dp,exist_ok=True)

    elif args.mode=="evaluate":
        dir_keys=["eval_dir"]
        for dk in dir_keys:
            dp=getattr(args,dk)
            os.makedirs(dp,exist_ok=True)

def print_and_save_parameters(args,print_parameters=True,save_parameters=True):
    if print_parameters:
        pprint(args.__dict__)

    if save_parameters:
        d=dict()
        for k,v in args.__dict__.items():
            if isinstance(v,argparse.Namespace):
                d[k]=v.__dict__
            else:
                d[k]=v
        if args.mode=="evaluate":
            with open(join(args.eval_dir,"eval_config.yaml"), 'w') as f:
                yaml.dump(d, stream=f, default_flow_style=False, sort_keys=False)
        elif args.mode=="train":
            with open(join(args.config_dir,"config.yaml"), 'w') as f:
                yaml.dump(d, stream=f, default_flow_style=False, sort_keys=False)


def main_train(args):
    seed_all(args.seed)

    # potentially add new entries to args
    check_and_create_dirs(args)

    dataset=create_dataset(args)

    print_and_save_parameters(args)


    assert hasattr(args,"fold")
    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, args.fold))

    datasets = (train_dataset, val_dataset, test_dataset)

    results, test_auc, val_auc, test_acc, val_acc  = train(datasets, args.fold, args)

def main_evaluate(args):
    seed_all(args.seed)
    check_and_create_dirs(args)
    dataset=create_dataset(args)
    print_and_save_parameters(args)
    assert os.path.isfile(args.model_ckpt_fp)
    detailed_out_fp=os.path.join(args.eval_dir, 'EVAL--detailed.csv')
    # assert not os.path.isfile(detailed_out_fp)
    datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}
    if datasets_id[args.eval_split] < 0:
        split_dataset = dataset
    else:
        csv_path = '{}/config/splits.csv'.format(args.associated_train_dir, args.fold)
        datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
        split_dataset = datasets[datasets_id[args.eval_split]]

    model, patient_results, metric_dict, df  = eval(split_dataset, args, args.model_ckpt_fp)
    df.to_csv(detailed_out_fp, index=False)
    yaml_metric_dict=copy(metric_dict)
    del yaml_metric_dict["eval_scores"]["confusion_matrix"]
    with open(os.path.join(args.eval_dir, 'EVAL--summary.yaml'.format(args.fold)),'w') as f:
        yaml.dump(yaml_metric_dict,f)
    with open(os.path.join(args.eval_dir, 'EVAL--summary.pkl'.format(args.fold)),'wb') as f:
        pickle.dump(metric_dict,f)

def get_parser():
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("mode",type=str)
    parser.add_argument("--config",type=str,required=True)
    parser.add_argument("--default",type=str,required=True)
    parser.add_argument("--fold",type=int,required=True)
    return parser

def preprocess_args(args):
    args.features_dir=join(args.data_files_dir,args.features_dir)
    args.info_csv_path=join(args.data_files_dir,args.info_csv_path)
    args.split_dir=join(args.data_files_dir,args.split_dir)
    if hasattr(args,"patches_dir"):
        args.patches_dir=join(args.data_files_dir,args.patches_dir)
        h5_file_fp=next(Path(join(args.patches_dir,"patches")).glob("*.h5"))
        with h5py.File(h5_file_fp,'r') as f:
            attr_dict=dict(f["coords"].attrs)
            for k in attr_dict.keys():
                if isinstance(attr_dict[k],np.ndarray):
                    attr_dict[k]=attr_dict[k].tolist()
                if isinstance(attr_dict[k],np.number):
                    attr_dict[k]=attr_dict[k].item()

            args.patches_attrs=attr_dict
    args.split_dir=join(args.split_dir,f"{args.task}_{int(args.label_frac*100)}")
    if args.mode=="evaluate":
        args.testing=True
        if not args.associated_train_dir and args.model_ckpt_fp:
            args.associated_train_dir=join(dirname(args.model_ckpt_fp),"..")
        elif not args.associated_train_dir and not args.model_ckpt_fp and args.associated_train_root_dir:
            fold_dir=join(args.associated_train_root_dir,args.exp_code,f"fold-{args.fold}")
            time_dirs=os.listdir(fold_dir)
            latest_time_dir=list(sorted(time_dirs))[-1]
            args.associated_train_dir=join(fold_dir,latest_time_dir)
            
            print(f"finding latest train_dir in {fold_dir}, which is {latest_time_dir}")
        else:
            assert False
        args.eval_dir=join(args.associated_train_dir,"results/EVAL")
        args.model_ckpt_fp=join(args.associated_train_dir,"ckpt","best_model.pt")
    else:
        args.testing=False
        
def add_default_args(args):
    json_fp=args.default
    with open(json_fp) as f:
        d=yaml.safe_load(f)
    for k,v in d.items():
        if not hasattr(args,k):
            setattr(args,k,v)

if __name__=="__main__":
    args=parse_args(get_parser())
    add_default_args(args)
    preprocess_args(args)
    if args.mode=="train":
        main_train(args)
        
    elif args.mode=="evaluate":
        main_evaluate(args)
    else:
        pass
