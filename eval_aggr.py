# common imports for data science
import os
import sys
from os.path import join,basename,dirname,splitext
from pathlib import Path
import numpy as np
import scipy
from scipy import io
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from pprint import pprint
import argparse
from collections import defaultdict,OrderedDict,deque
import gzip
import yaml
import pickle as pkl
# common imports for sklearn
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

class CLAMTrainDir(object):
    def __init__(self,train_dir,eval_fns=None):
        self.train_dir=train_dir
        self.__setup_config_dict()
        if eval_fns is None:
            self.eval_fns=["eval_config.yaml","EVAL--detailed.csv","EVAL--summary.pkl"]
        else:
            self.eval_fns=eval_fns
    def __setup_config_dict(self):
        self.config_dict=dict()
        # set up subdir_dict
        subdir_dict=dict()
        subdir_dict["ckpt_dp"]="ckpt"
        subdir_dict["best_model_fp"]=join(subdir_dict["ckpt_dp"],"best_model.pt")
        ckpt_fps=list(
            sorted(
                Path(self.train_dir,subdir_dict["ckpt_dp"]).glob("epoch-*.pt"),
                key=lambda x: int(x.stem.split('-')[1])
            )
        )
        if len(ckpt_fps)>0:
            subdir_dict["latest_model_fp"]=str(ckpt_fps[-1].relative_to(self.train_dir))
        else:
            subdir_dict["latest_model_fp"]=None
        subdir_dict["config_fp"]=join("config","config.yaml")
        subdir_dict["splits_fp"]=join("config","splits.csv")
        subdir_dict["tb_data_dp"]=join("history","tb_data")
        subdir_dict["results_dp"]="results"
        subdir_dict["EVAL_dp"]=join("results","EVAL")
        self.config_dict["subdir_dict"]=subdir_dict

    def get_subdir(self,key):
        subdir_dict=self.config_dict["subdir_dict"]
        dir_fp=join(self.train_dir,subdir_dict[key])
        os.makedirs(dir_fp,exist_ok=True)
        return dir_fp

    def read_eval_scores(self):
        ret_dict=dict()
        EVAL_dp=self.get_subdir("EVAL_dp")

        for eval_fn in self.eval_fns:
            noext,ext=splitext(eval_fn)
            if ext==".yaml":
                with open(join(EVAL_dp,eval_fn)) as f:
                    ret_dict[noext]=yaml.load(f)
            elif ext==".csv":
                table=pd.read_csv(join(EVAL_dp,eval_fn))
                ret_dict[noext]=table
            elif ext==".pkl":
                with open(join(EVAL_dp,eval_fn),'rb') as f:
                    ret_dict[noext]=pkl.load(f)
        return ret_dict

class CLAMTrainFoldsDir(object):
    def __init__(self,train_folds_dir,force_time_dir=dict()):
        self.train_folds_dir=train_folds_dir
        self.force_time_dir=force_time_dir
        self.__setup_config_dict(self.force_time_dir)
        self.__setup_train_dir_objs()
    def __setup_config_dict(self,force_time_dir):
        self.config_dict=dict()

        # find folds in dir
        self.config_dict["subdir_dict"]=dict()
        fold_train_dir_dict=dict()
        for fold_dp in Path(self.train_folds_dir).glob("fold-*"):
            if os.path.isdir(fold_dp):
                print(fold_dp.parts[-1],":")
                time_dirs=os.listdir(fold_dp)
                print("list of time_dirs:")
                pprint(time_dirs)
                if len(time_dirs)>0:
                    if fold_dp.parts[-1] in force_time_dir:
                        time_dn=force_time_dir[fold_dp.parts[-1]]
                        print("using forced time_dir:")
                        print(time_dn)
                        fold_train_dir_dict[fold_dp.parts[-1]]=join(
                            self.train_folds_dir,
                            fold_dp.parts[-1],
                            time_dn
                        )
                    else:
                        latest_time_dir=list(sorted(time_dirs))[-1]
                        print("latest time_dir:")
                        print(latest_time_dir)
                        fold_train_dir_dict[fold_dp.parts[-1]]=join(
                            self.train_folds_dir,
                            fold_dp.parts[-1],
                            latest_time_dir
                        )
                else:
                    print("no time_dir found")
                print()
        self.config_dict["subdir_dict"]["fold_train_dir_dict"]=fold_train_dir_dict

        aggr_dir=join(self.train_folds_dir,"fold_aggr")
        if os.path.isdir(aggr_dir):
            fold_aggr_detailed_fp=join(aggr_dir,"EVAL--fold_aggr--detailed.csv")
            fold_aggr_summary_fp=join(aggr_dir,"EVAL--fold_aggr--summary.csv")
            self.config_dict["subdir_dict"]["fold_aggr_dp"]=aggr_dir
            self.config_dict["subdir_dict"]["EVAL--fold_aggr--detailed_fp"]=fold_aggr_detailed_fp
            self.config_dict["subdir_dict"]["EVAL--fold_aggr--summary_fp"]=fold_aggr_summary_fp

    def __setup_train_dir_objs(self):
        fold_train_dir_dict=self.config_dict["subdir_dict"]["fold_train_dir_dict"]
        self.train_dir_objs=dict()
        for fold in fold_train_dir_dict:
            self.train_dir_objs[fold]=CLAMTrainDir(fold_train_dir_dict[fold])


    def get_subdir(self,key):
        subdir_dict=self.config_dict["subdir_dict"]
        fold_train_dir_dict=self.config_dict["subdir_dict"]["fold_train_dir_dict"]
        subdir_fp=subdir_dict[key] if key in subdir_dict else fold_train_dir_dict[key]
        dir_fp=join(self.train_folds_dir,subdir_fp)
        os.makedirs(dir_fp,exist_ok=True)
        return dir_fp

    def compute_aggr_eval_metrics(self):
        eval_detailed_df_list=list()
        eval_summary_dict_list=list()
        folds=list(sorted([int(s.split('-')[1]) for s in self.train_dir_objs.keys() if '-' in s]))
        for fold in folds:
            train_dir_obj=self.get_train_dir_objs(f'fold-{fold}')
            eval_scores_dict=train_dir_obj.read_eval_scores()
            eval_detailed_df=eval_scores_dict["EVAL--detailed"]
            eval_detailed_df["fold"]=fold
            eval_detailed_df_list.append(eval_detailed_df)
            eval_summary_dict=eval_scores_dict["EVAL--summary"]
            eval_summary_dict_list.append(eval_summary_dict)


        df=pd.concat(eval_detailed_df_list,axis=0)
        micro_acc=np.mean(df["Y"]==df["Y_hat"])
        macro_acc=np.mean([summary_dict["eval_scores"]["accuracy"] for summary_dict in eval_summary_dict_list])

        if eval_summary_dict_list[0]["result_type"]=="multiclass_classification":
            assert all([summary_dict["result_type"]=="multiclass_classification" for summary_dict in eval_summary_dict_list])
            auc_class_micro_fold_macro=np.mean([summary_dict["eval_scores"]["auc_roc_class_micro"] for summary_dict in eval_summary_dict_list])
            auc_class_macro_fold_macro=np.mean([summary_dict["eval_scores"]["auc_roc_class_macro"] for summary_dict in eval_summary_dict_list])

            aggr_dir=join(self.train_folds_dir,"fold_aggr")
            fold_aggr_detailed_fp=join(aggr_dir,"EVAL--fold_aggr--detailed.csv")
            fold_aggr_summary_fp=join(aggr_dir,"EVAL--fold_aggr--summary.yaml")
            os.makedirs(aggr_dir,exist_ok=True)

            df.to_csv(join(aggr_dir,"EVAL--fold_aggr--detailed.csv"),index=None)
            with open(join(aggr_dir,"EVAL--fold_aggr--summary.yaml"),'w') as f:
                aggr_dict={
                    'micro_acc':float(micro_acc),
                    "macro_acc":float(macro_acc),
                    "auc_class_micro_fold_macro":float(auc_class_micro_fold_macro),
                    "auc_class_macro_fold_macro":float(auc_class_macro_fold_macro),
                    'summary_dict_list':eval_summary_dict_list
                }
                yaml.dump(aggr_dict,f)


        elif eval_summary_dict_list[0]["result_type"]=="binary_classification":
            assert all([summary_dict["result_type"]=="binary_classification" for summary_dict in eval_summary_dict_list])
            auc_fold_macro=np.mean([summary_dict["eval_scores"]["auc_roc"] for summary_dict in eval_summary_dict_list])

            aggr_dir=join(self.train_folds_dir,"fold_aggr")
            fold_aggr_detailed_fp=join(aggr_dir,"EVAL--fold_aggr--detailed.csv")
            fold_aggr_summary_fp=join(aggr_dir,"EVAL--fold_aggr--summary.yaml")
            os.makedirs(aggr_dir,exist_ok=True)


            df.to_csv(fold_aggr_detailed_fp,index=None)
            with open(fold_aggr_summary_fp,'w') as f:
                aggr_dict={
                    'micro_acc':float(micro_acc),
                    "macro_acc":float(macro_acc),
                    "auc_fold_macro":float(auc_fold_macro),
                    'summary_dict_list':eval_summary_dict_list,
                }
                yaml.dump(aggr_dict,f)

        self.__setup_config_dict(self.force_time_dir)


    def get_train_dir_objs(self,key):
        return self.train_dir_objs[key]
if __name__=="__main__":
    train_folds_dir=sys.argv[1]
    trainfoldsdir=CLAMTrainFoldsDir(train_folds_dir)
    trainfoldsdir.compute_aggr_eval_metrics()
    
