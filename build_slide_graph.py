import os
import sys
from os.path import join,basename,dirname,splitext
from pathlib import Path
from glob import glob
import numpy as np
import scipy
from scipy import io
import scipy.stats as stats
import scipy.sparse as sp
import pickle as pkl
from pprint import pprint
import argparse

from slide_graph_builder.graph_builder import SlideGraph
import utils.utils as utils


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_files_dir",type=str,required=True)
    parser.add_argument("--dataset_name",type=str,required=True)
    parser.add_argument("--patches_dir_name",type=str,required=True)
    parser.add_argument("--features_dir_name",type=str,required=True)
    parser.add_argument('--n_tasks',type=int,default=-1)
    parser.add_argument('--task_id',type=int,default=-1)
    parser.add_argument('--knn',type=int,default=4)
    args=parser.parse_args()
    return args
if __name__=="__main__":
    args=parse_args()
    data_files_dir=args.data_files_dir
    dataset_name=args.dataset_name
    

    patches_dir_name=args.patches_dir_name
    patches_dir_dp=os.path.join(data_files_dir,"patches",patches_dir_name,"patches")
    assert os.path.isdir(patches_dir_dp)
    features_dir_name=args.features_dir_name
    features_dir_dp = os.path.join(data_files_dir,"features",features_dir_name,"h5_files")
    assert os.path.isdir(features_dir_dp)

    slide_ids=[os.path.splitext(fn)[0] for fn in os.listdir(features_dir_dp)]
    total=len(slide_ids)

    slide_graph_dir=os.path.join(data_files_dir,"slide_graph")
    build_graph_config_name=f"{args.patches_dir_name}--{args.features_dir_name}--feature_nn_{args.knn}-coord_nn_{args.knn}"
    build_graph_out_dir=join(slide_graph_dir,build_graph_config_name)
    os.makedirs(build_graph_out_dir,exist_ok=True)

    if args.n_tasks>0 and 0<=args.task_id<args.n_tasks:
        rg=utils.divide_work(total,args.n_tasks,args.task_id)
        print(f"running from [{rg.start},{rg.stop})")
    else:
        print(f"running total from ({0},{total})")
        rg=range(0,total)
    
    for i in rg:
        save_fp=join(build_graph_out_dir,"{}.pkl".format(slide_ids[i]))
        if not os.path.isfile(save_fp):
            print(f"processing: {save_fp}")
            slide_graph=SlideGraph.build_slide_graph(patches_dir_dp,features_dir_dp,slide_ids[i],patches_dir_name,features_dir_name,k=args.knn)
            slide_graph.compute()
            slide_graph.to_pkl(save_fp)
        else:
            print(f"skipping: {save_fp}")