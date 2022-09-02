import os
from os.path import join
from pathlib import Path
from timeit import timeit
import time
from PIL import Image
import openslide

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader,Subset, TensorDataset
import torchvision.transforms as transforms

from utils.utils import temp_seed
import torch.utils.data._utils as _utils

class PrefetchedPathologyPairedPatchesDataset(Dataset):
    def __init__(self,pairs_h5_fp, data_files_dir,patch_size,transform=None, use_tiny_frac=False):
        self.df=pd.read_hdf(pairs_h5_fp)
        with temp_seed(1):
            if use_tiny_frac:
                tiny_frac_size=10000
                print(f"sample a tiny fraction ({tiny_frac_size}/{len(self.df)})")
                self.df=self.df.sample(tiny_frac_size)
                # 20:50 failed
                self.df=self.df.reset_index(drop=True)
                print("using all")
        self.data_files_dir=data_files_dir
        self.patch_size=patch_size
        self.transform=transform
        self.prefetched_data=dict()
        if self.transform==None:
            self.transform=transforms.ToTensor()
    def prefetch(self):
        print("Opening file handles")
        slide_ids_to_load=np.union1d(self.df["slide_id_1"].unique(),self.df["slide_id_2"].unique())
        slide_ids_to_load.sort()
        for i,slide_id in enumerate(slide_ids_to_load,1):
            print(f"{i},{slide_id}")
            slide_fp=join(self.data_files_dir,"WSI",slide_id+".svs")
            slide_fh = openslide.OpenSlide(slide_fp)
            self.prefetched_data[slide_id]=slide_fh
    def __getitem__(self,index):

        df_row=self.df.loc[index]
        if len(self.prefetched_data)==0:
            self.prefetch()
        data1_fh=self.prefetched_data[df_row["slide_id_1"]]
        data2_fh=self.prefetched_data[df_row["slide_id_2"]]

        # os.makedirs(inspection_dir,exist_ok=True)
        X1=data1_fh.read_region((df_row["X_1"],df_row["Y_1"]),0,(self.patch_size,self.patch_size)).convert('RGB')
        X2=data2_fh.read_region((df_row["X_2"],df_row["Y_2"]),0,(self.patch_size,self.patch_size)).convert("RGB")
        if self.transform:
            X1_transformed=self.transform(X1)
            X2_transformed=self.transform(X2)

        Y=np.array([self.df.loc[index,"label"]],dtype=np.float32)
            
        return X1_transformed,X2_transformed,Y
    def __len__(self):
        return len(self.df)

class BatchPrioritizedDataLoader(object):
    def __init__(self,dataset,batch_size=1,num_workers=1,shuffle=False,collate_fn=None):
        self.dataset=dataset
        self.batch_size=batch_size
        self.num_workers=num_workers
        if collate_fn is None:
            self.collate_fn=_utils.collate.default_collate
        else:
            self.collate_fn=collate_fn
        self.shuffle=shuffle
    def __iter__(self):
        self.current_idx=0
        if self.shuffle:
            self.idx_order=np.random.permutation(len(self.dataset))
        else:
            self.idx_order=np.arange(len(self.dataset))
        return self
    def __next__(self):
        if self.current_idx>=len(self.dataset):
            raise StopIteration
        else:
            start_idx=self.current_idx
            end_idx=min(self.current_idx+self.batch_size,len(self.dataset))
            subset=Subset(self.dataset,self.idx_order[start_idx:end_idx])
            torch_data_loader=DataLoader(subset, batch_size=None, num_workers=self.num_workers)
            self.current_idx+=self.batch_size
            return self.collate_fn([example for example in torch_data_loader])
    def __len__(self):
        return (len(self.dataset)+self.batch_size-1)//(self.batch_size)
        
class BatchPrioritizedDataLoaderV2(object):
    def __init__(self,dataset,batch_size=1,num_workers=1,shuffle=False,collate_fn=None, prefetch_factor=None, verbose=False, pin_memory=False):
        self.dataset=dataset
        self.batch_size=batch_size
        self.num_workers=num_workers
        if collate_fn is None:
            self.collate_fn=_utils.collate.default_collate
        else:
            self.collate_fn=collate_fn
        self.prefetch_factor=prefetch_factor
        self.pin_memory=pin_memory
        self.shuffle=shuffle
        self.verbose=verbose
    def __iter__(self):
        self.current_idx=0
        if self.prefetch_factor:
            self.torch_data_loader=DataLoader(self.dataset, batch_size=None, 
            num_workers=self.num_workers, shuffle=False, pin_memory=self.pin_memory, prefetch_factor=self.prefetch_factor)
        else:
            self.torch_data_loader=DataLoader(self.dataset, batch_size=None, pin_memory=self.pin_memory, num_workers=self.num_workers, shuffle=False)
        self.torch_data_loader_iter=iter(self.torch_data_loader)
        if self.shuffle:
            self.idx_order=np.random.permutation(len(self.dataset))
        else:
            self.idx_order=np.arange(len(self.dataset))
        return self
    def __next__(self):
        if self.current_idx>=len(self.dataset):
            raise StopIteration
        else:
            start_idx=self.current_idx
            end_idx=min(self.current_idx+self.batch_size,len(self.dataset))
            fetched_data=list()
            tic=time.time()
            for idx in range(0,end_idx-start_idx):
                fetched_data.append(next(self.torch_data_loader_iter))
            toc=time.time()
            if self.verbose:
                print("Size of worker result queue: ",self.torch_data_loader_iter._worker_result_queue.qsize())
                print("Size of index queue: ")
                for i in range(len(self.torch_data_loader_iter._index_queues)):
                    print(f"\tqueue[{i}]: {self.torch_data_loader_iter._index_queues[i].qsize()}")
                print("Time to get data", toc-tic)
            tic=time.time()
            collated=self.collate_fn(fetched_data)
            toc=time.time()
            if self.verbose:
                print("Time to collate data", toc-tic)
            self.current_idx+=self.batch_size
            return collated
    def __len__(self):
        return (len(self.dataset)+self.batch_size-1)//(self.batch_size)
        
if __name__=="__main__":
    dataset=TensorDataset(torch.arange(100))
    myloader=BatchPrioritizedDataLoaderV2(dataset,batch_size=10,num_workers=10)
    for batch in myloader:
        print(batch)
