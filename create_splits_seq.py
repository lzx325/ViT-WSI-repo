import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np
from utils.utils import parse_dict_args,str2bool
parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)') # lizx: after preparing val and test, the percentage of remaining data that goes to train
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'])
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--label_dict', type=str, required=True)
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument("--test_same_as_val",type=str2bool,default=False)
args = parser.parse_args()
label_dict=parse_dict_args(args,"label_dict")
print(label_dict)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    # label_dict = {
    #     'normal_tissue':0, 
    #     'tumor_tissue':1,
    #     'negative':0,
    #     'itc':0,
    #     'micro':1,
    #     'macro':1,
    # }
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = label_dict,
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    # csv_path="dataset_files/HMU1st_3major_150slides.features/HMU1st_3major_150slides.csv"
    # label_dict = {'胶质瘤':0, '脑膜瘤':1, '垂体腺瘤':2}
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.csv_path,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = label_dict,
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = os.path.join(args.save_dir,str(args.task) + '_{}'.format(int(lf * 100)))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf, test_same_as_val=args.test_same_as_val) # lizx: sets the dataset.split_gen attribute. Each time the generator will generate the train/val/test split for one fold
        
        for i in range(args.k):
            dataset.set_splits() # lizx: set the attributes: self.train_ids, self.val_ids, self.test_ids
            descriptor_df = dataset.test_split_gen(return_descriptor=True,allow_val_test_overlap=args.test_same_as_val) # print and return the train/val/test splits
            splits = dataset.return_splits(from_id=True)
            # splits is a wrapper class that has attribute: `slide_data` (dataframe) `data_dir` `num_class`

            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



