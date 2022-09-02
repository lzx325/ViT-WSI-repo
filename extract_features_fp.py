import torch
import torch.nn as nn
import torchvision

from math import floor
import os
import random
import numpy as np
import pdb
import time
import tempfile

from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader

import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from pprint import pprint
from PIL import Image
import h5py
import openslide
import shutil


import utils.model_utils
import utils.utils as utils
from datasets.optimized_dataset import BatchPrioritizedDataLoaderV2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1, target_feature_dim=1024,custom_transforms=None):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""

	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size,custom_transforms=custom_transforms, unsqueeze=True)
	if len(dataset)==0:
		return None
	x, y = dataset[0]
	kwargs = {'num_workers': 16, 'pin_memory': True} if device.type == "cuda" else {}
	# loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)
	loader=BatchPrioritizedDataLoaderV2(dataset=dataset, batch_size=batch_size, **kwargs,collate_fn=collate_features)
	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	tic=time.time()
	for count, (batch, coords) in enumerate(loader):
		toc=time.time()
		print("batch {}/{}, Time to load data: {}".format(count, len(loader),toc-tic))
		with torch.no_grad():	
			tic=time.time()
			# lizx added
			model.eval()
			if count % print_every == 0:
				
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			mini_bs = coords.shape[0]
			
			features = model(batch)
			features = features.cpu().numpy()
			if features.shape[1] < target_feature_dim:
				if count==0:
					print(f"Original feature dim: {features.shape[1]}, Padded {target_feature_dim-features.shape[1]}")
				features=np.concatenate([features,np.zeros((features.shape[0],target_feature_dim-features.shape[1]))],axis=1)
				
			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
			toc=time.time()
			print("batch {}/{}, computation time {}".format(count, len(loader),toc-tic))
			tic=time.time()
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--model_type', type=str, required=True)
parser.add_argument('--custom_ckpt_fp',type=str,required=True)
parser.add_argument('--begin_idx',type=int,default=-1)
parser.add_argument('--end_idx',type=int,default=-1)
parser.add_argument('--n_tasks',type=int,default=-1)
parser.add_argument('--task_id',type=int,default=-1)

args = parser.parse_args()


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')
	if args.model_type=="resnet50_baseline":
		model = resnet50_baseline(pretrained=True)
		model = model.to(device)
		custom_transforms=None
	elif args.model_type=="resnet18_pretrained_on_patch":
		model = torchvision.models.resnet18(num_classes=1)
		model = utils.model_utils.ResNetFeatureExtractor(model)
		custom_ckpt_fp=args.custom_ckpt_fp
		ckpt=torch.load(custom_ckpt_fp)
		inp=torch.rand(1,3,256,256)
		res=model(inp)
		model = model.to(device)
		custom_transforms=None
	elif args.model_type=="vit_large_patch16_384":
		from models.vit_feature_extractor import vit_large_patch16_384
		model,custom_transforms=vit_large_patch16_384()
		model=model.to(device)

	print(f"using model {args.model_type}")
	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)
	tmp_dir=tempfile.mkdtemp()
	tmp_slide_dir=os.path.join(tmp_dir,"slides")
	os.makedirs(tmp_slide_dir,exist_ok=True)
	if args.begin_idx>=0 and args.end_idx>0:
		args.end_idx=min(args.end_idx,total)
		print(f"running from [{args.begin_idx},{args.end_idx})")
		rg=range(args.begin_idx,args.end_idx)
	elif args.n_tasks>0 and 0<=args.task_id<args.n_tasks:
		rg=utils.divide_work(total,args.n_tasks,args.task_id)
		print(f"running from [{rg.start},{rg.stop})")
	else:
		print(f"running total from ({0},{total})")
		rg=range(0,total)

	for bag_candidate_idx in rg:
		slide_id = bags_dataset[bag_candidate_idx]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		if not os.path.isfile(h5_file_path):
			print(f"{h5_file_path} not found, continue")
			continue
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)


		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 
		if not bags_dataset.df.set_index("slide_id").loc[slide_id,"process"].item()==0:
			print("skipped in process list: {}".format(slide_id))
			continue
		
		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		try:
			
			output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
			model = model, batch_size = args.batch_size, verbose = 1, print_every = 1, 
			custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size,
			custom_transforms=custom_transforms)
		finally:
			wsi.close()
		if output_file_path is not None:
			time_elapsed = time.time() - time_start
			print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
			file = h5py.File(output_file_path, "r")

			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)
			features = torch.from_numpy(features)
			bag_base, _ = os.path.splitext(bag_name)
			torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))
		else:
			print("current slide is skipped for feature computation because no patches are extracted from it")



