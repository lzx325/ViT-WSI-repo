import pickle
from networkx.algorithms import dag
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pdb
import argparse
import os

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import pandas as pd
import collections
import contextlib
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]

def collate_features(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	coords = np.vstack([item[1] for item in batch])
	return [img, coords]

def collate_slide_graph(batch):
	return batch

def get_simple_loader(dataset, batch_size=1, num_workers=1, collate_fn=collate_MIL):
	kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_fn, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False, batch_size=1, collate_fn=collate_MIL):
	"""
		return either the validation loader or training loader 
	"""
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_fn, **kwargs)	
			else:
				loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate_fn, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_fn, **kwargs)
	
	else:
		ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
		loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SubsetSequentialSampler(ids), collate_fn = collate_fn, **kwargs )

	return loader

def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None, test_same_as_val=False):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		for c in range(len(val_num)): # lizx: for each class
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)
			if test_same_as_val:
				all_test_ids.extend(val_ids)
			else:
				if custom_test_ids is None: # sample test split
					test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
					remaining_ids = np.setdiff1d(remaining_ids, test_ids)
					all_test_ids.extend(test_ids)
				else: # pre-built test split, do not need to sample
					all_test_ids.extend(custom_test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)
			
def parse_dict_args(args,key):
    dict_args=args.__getattribute__(key)
    d=dict()
    for arg in dict_args.split(","):
        arg=arg.strip()
        k,v=arg.split(':')
        d[k]=int(v)
    return d

def normalize8(I,cmap=None,mn=None,mx=None):
    if mn is None:
        mn = I.min()
    if mx is None:
        mx = I.max()
    mx -= mn
    if mx>0:
        I = ((I - mn)/mx) 
    else:
        I=np.zeros_like(I,dtype=np.float32)
    if cmap is not None:
        cm = plt.get_cmap('jet')
        I=cm(I)
        assert np.allclose(I[...,3],1)
        I=I[...,:3]
    I=(I*255).astype(np.uint8)
    return I

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    store.get_storer('mydata').attrs.metadata = kwargs
    store.close()

def h5load(filename):
	with pd.HDFStore(filename) as store:
		data = store['mydata']
		metadata = store.get_storer('mydata').attrs.metadata
		return data, metadata

class TensorNamespace(argparse.Namespace):
	def __init__(self):
		super(TensorNamespace,self).__init__()
	def to(self,device,non_blocking=False):
		for nm in dir(self):
			obj = getattr(self,nm)
			if type(obj)==torch.Tensor:
				setattr(self,nm,obj.to(device,non_blocking=non_blocking))
		return self
	def size(self,dim=None):
		if dim is not None:
			return self.x.size(dim)
		else:
			return self.x.size()

class GenericTensorNamespace(argparse.Namespace):
	def __init__(self):
		super(GenericTensorNamespace,self).__init__()
	def to(self,device,non_blocking=False) -> "GenericTensorNamespace":
		for nm in dir(self):
			obj = getattr(self,nm)
			if type(obj)==torch.Tensor:
				setattr(self,nm,obj.to(device,non_blocking=non_blocking))
			
		return self
	def numpy(self) -> "GenericTensorNamespace":
		new_ns=GenericTensorNamespace()
		for nm in dir(self):
			obj = getattr(self,nm)
			if type(obj)==torch.Tensor:
				setattr(new_ns,nm,obj.cpu().numpy())
		return new_ns
	

def divide_work(n_total,ntasks,taskid):
	# assert "SLURM_NTASKS" in os.environ
	# assert "SLURM_PROCID" in os.environ
	# slurm_ntasks=int(os.environ["SLURM_NTASKS"])
	# slurm_procid=int(os.environ["SLURM_PROCID"])
	assert 0<=taskid<ntasks
	work_per_task=(n_total+ntasks-1)//ntasks
	return range(
		taskid*work_per_task,
		min((taskid+1)*work_per_task,n_total),
	)

def iprint(s="",*args,ident=0,**kwargs):
	print('\t'*ident+s,*args,**kwargs)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_latest_time_dir(fold_dir,return_full_path=True):
	time_dirs=os.listdir(fold_dir)
	latest_time_dir=list(sorted(time_dirs))[-1]
	if return_full_path:
		return os.path.join(fold_dir,latest_time_dir)
	else:
		return latest_time_dir
	
def convert_numpy_values(d):
	for k in d.keys():
		if isinstance(d[k],np.ndarray):
			d[k]=d[k].flatten().tolist()
		if isinstance(d[k],np.number):
			d[k]=d[k].item()
	return d