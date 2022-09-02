import numpy as np
import torch
from torch.functional import Tensor
from utils.utils import *
import os
import h5py
from PIL import Image
from scipy import io
from datasets.dataset_generic import save_splits
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import auc as calc_auc
from datetime import datetime as dt
from wsi_core.WholeSlideImage import WholeSlideImage
import utils.utils
import tensorboardX

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStoppingModelSaver:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(
        self, epoch, 
        val_loss, model, 
        routine_ckpt_dir,
        best_ckpt_name="best_model.pt"
    ):
        self.save_routine_ckpt(
            epoch,model,routine_ckpt_dir
        )
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_best_checkpoint(val_loss, epoch, model,routine_ckpt_dir,  best_ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_best_checkpoint(val_loss, epoch, model, routine_ckpt_dir, best_ckpt_name)
            self.counter = 0

    def save_best_checkpoint(self, val_loss, epoch, model, routine_ckpt_dir, best_ckpt_name):
        '''Saves model when validation metric decrease.'''
        os.makedirs(routine_ckpt_dir,exist_ok=True)
        if self.verbose:
            print(f'Validation metric decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Linking current best model ...')
        best_ckpt_fp=os.path.join(routine_ckpt_dir,best_ckpt_name)
        if os.path.isfile(best_ckpt_fp):
            os.unlink(best_ckpt_fp)
        
        best_ckpt_tgt_fn=f"epoch-{epoch}.pt"
        os.symlink(best_ckpt_tgt_fn,best_ckpt_fp)
        self.val_loss_min = val_loss
    
    def save_routine_ckpt(self, epoch, model, routine_ckpt_dir):
        os.makedirs(routine_ckpt_dir,exist_ok=True)
        ckpt_name=f"epoch-{epoch}.pt"
        if self.verbose:
            print(f'Routinely saving epoch {epoch} to {ckpt_name}')
        torch.save(model.state_dict(),os.path.join(routine_ckpt_dir,ckpt_name))


def get_heatmap(
    wsi: WholeSlideImage,A,coords,
    **kwargs,
):
    default_heatmap_params={
                'vis_level':-1,
                'cmap_1': 'jet',
                "value_type":"one_part",
                'alpha': 0.4, 
                'use_holes': True, 
                'binarize': False, 
                'blank_canvas': False, 
                'thresh': -1, 
                'patch_size': (256, 256), 
                'normalization_method': None,
                "segment":False,
                "return_overlay":True,
                "return_PIL":False,
    }
    default_heatmap_params.update(kwargs)
    overlay=wsi.visHeatmap(
        A,
        coords,
        **default_heatmap_params
    )
    return overlay


def visualize_examples(
    epoch,
    model,
    model_type,
    features_dir,
    wsi_dir,
    slide_ids,
    save_dir,
    ext,
):
    if slide_ids is None:
        return
    device=next(model.parameters()).device
    for slide_id in slide_ids:
        slide_save_dir=os.path.join(save_dir,slide_id)
        pt_fp=os.path.join(features_dir,"pt_files","%s.pt"%(slide_id))
        h5_fp=os.path.join(features_dir,"h5_files","%s.h5"%(slide_id))
        slide_fp=os.path.join(wsi_dir,f"{slide_id}.{ext}")
        with h5py.File(h5_fp,'r') as f:
            coords=f["coords"][()]
        data=torch.load(pt_fp)
        with torch.no_grad():
            data=data.to(device)
            wsi=WholeSlideImage(slide_fp)

            if model_type in ['clam_sb', 'clam_mb']:
                _, _, _, A_all, _=model(data)
                A_all=A_all.cpu().numpy()
            elif model_type in ["mil"]:
                ret_val=model(data)
                _,_,_,A_all,_=model(data)

                A_all=A_all.cpu().numpy()
                A_all=A_all.transpose([1,0])
            else:
                assert False
            for cl in range(len(A_all)):
                png_percnorm_dir=os.path.join(slide_save_dir,"class%02d"%(cl),"png_percnorm")
                png_percnorm_fp=os.path.join(png_percnorm_dir,"%s_class%02d_percnorm_epoch%03d.png"%(slide_id,cl,epoch))
                png_nopercnorm_dir=os.path.join(slide_save_dir,"class%02d"%(cl),"png_nopercnorm")
                png_nopercnorm_fp=os.path.join(png_nopercnorm_dir,"%s_class%02d_nopercnorm_epoch%03d.png"%(slide_id,cl,epoch))
                overlay_dir=os.path.join(slide_save_dir,"class%02d"%(cl),"png_overlay")
                overlay_fp=os.path.join(overlay_dir,"%s_class%02d_overlay_epoch%03d.png"%(slide_id,cl,epoch))
                mat_percnorm_dir=os.path.join(slide_save_dir,"class%02d"%(cl),"mat_percnorm")
                mat_percnorm_fp=os.path.join(mat_percnorm_dir,"%s_class%02d_percnorm_epoch%03d.mat"%(slide_id,cl,epoch))
                mat_nopercnorm_dir=os.path.join(slide_save_dir,"class%02d"%(cl),"mat_nopercnorm")
                mat_nopercnorm_fp=os.path.join(mat_nopercnorm_dir,"%s_class%02d_nopercnorm_epoch%03d.mat"%(slide_id,cl,epoch))

                os.makedirs(png_percnorm_dir,exist_ok=True)
                os.makedirs(png_nopercnorm_dir,exist_ok=True)
                os.makedirs(overlay_dir,exist_ok=True)
                os.makedirs(mat_percnorm_dir,exist_ok=True)
                os.makedirs(mat_nopercnorm_dir,exist_ok=True)

                im_arr=get_heatmap(wsi,A_all[cl],coords,return_overlay=True,convert_to_percentiles=True,return_PIL=False)
                im_cmap=normalize8(im_arr,"jet")
                im=Image.fromarray(im_cmap)
                im.save(png_percnorm_fp)
                io.savemat(mat_percnorm_fp,{"overlay":im_arr})

                im_arr=get_heatmap(wsi,A_all[cl],coords,return_overlay=True,convert_to_percentiles=False,return_PIL=False)
                im_cmap=normalize8(im_arr,"jet")
                im=Image.fromarray(im_cmap)
                im.save(png_nopercnorm_fp)
                io.savemat(mat_nopercnorm_fp,{"overlay":im_arr})

                im=get_heatmap(wsi,A_all[cl],coords,return_overlay=False,convert_to_percentiles=True,return_PIL=True)
                im.save(overlay_fp)

class SummaryWriter(tensorboardX.SummaryWriter):
    def __init__(self,writer_dir,txt_dir,flush_secs=15):
        super(SummaryWriter,self).__init__(writer_dir,flush_secs)
        self.txt_dir=txt_dir
        os.makedirs(self.txt_dir,exist_ok=True)
        self.txt_fp=os.path.join(self.txt_dir,"tb_scalars.txt")
        self.txt_file=open(self.txt_fp,"a")
        print("scalar_name,Value,Step",file=self.txt_file)
    def add_scalar(self,tag,scalar_value,global_step):
        super(SummaryWriter,self).add_scalar(tag,scalar_value,global_step)
        print(f"{tag},{float(scalar_value)},{int(global_step)}",file=self.txt_file)
        self.txt_file.flush()
    def close(self):
        super(SummaryWriter,self).close()
        self.txt_file.close()

def initiate_model(args, ckpt_path=None):
    print('Init Model') 
            
    if args.model_type in ["vit_aggr"]:
        from models.vit import ViTAggregation
        vit_model_dict={
            "heads":args.heads,
            "dim_head":args.dim_head,
            "mlp_dim":args.mlp_dim,
            "dim":args.dim,
            "depth":args.depth,
            "aggr":args.aggr,
            "n_classes":args.n_classes,

        }
        model=ViTAggregation(**vit_model_dict)

    elif args.model_type in ["graph_vit_aggr"]:
        from models.vit import GraphViTAggregation
        graph_vit_model_dict={
            "heads":args.heads,
            "dim_head":args.dim_head,
            "mlp_dim":args.mlp_dim,
            "n_classes":args.n_classes
        }

        model=GraphViTAggregation(**graph_vit_model_dict)

    else:
        raise NotImplementedError
    
    print_network(model)

    # load ckpt
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        ckpt_clean = {}
        for key in ckpt.keys():
            # if 'instance_loss_fn' in key:
            #     continue
            ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
        def to_qkv_patch(ckpt):
            if args.model_type in ["vit_aggr"]:
                if 'transformer.layers.0.0.fn.to_qkv.weight' in ckpt:
                    print("patching transformer.layers.0.0.fn.to_qkv.weight")
                    to_qkv_weight=ckpt["transformer.layers.0.0.fn.to_qkv.weight"]
                    assert to_qkv_weight.shape[0] %3 == 0
                    per_slot_size=to_qkv_weight.shape[0]//3
                    to_q_weight=to_qkv_weight[:per_slot_size,:]
                    to_k_weight=to_qkv_weight[per_slot_size:2*per_slot_size,:]
                    to_v_weight=to_qkv_weight[2*per_slot_size:3*per_slot_size,:]
                    ckpt["transformer.layers.0.0.fn.to_q.weight"]=to_q_weight
                    ckpt["transformer.layers.0.0.fn.to_k.weight"]=to_k_weight
                    ckpt["transformer.layers.0.0.fn.to_v.weight"]=to_v_weight
                    del ckpt['transformer.layers.0.0.fn.to_qkv.weight']
            return ckpt

        ckpt_clean=to_qkv_patch(ckpt_clean)
        model.load_state_dict(ckpt_clean, strict=True)

    # relocate
    model.relocate()

    # set model train/eval
    if args.testing:
        model.eval()
    else:
        model.train()
    print("Done")
    return model

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))


    writer_dir=args.tensorboard_dir
    vis_dir=args.vis_dir
    ckpt_dir=args.ckpt_dir
    tb_data_dir=args.tb_data_dir

    if args.log_data:
        writer = SummaryWriter(writer_dir,tb_data_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.config_dir, 'splits.csv'))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    model=initiate_model(args)


    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    if args.model_type in ["graph_vit_aggr"]:
        from utils.utils import collate_slide_graph
        train_loader = get_split_loader(train_split, training=True, testing = args.testing, 
        weighted = args.weighted_sample,batch_size=None,collate_fn=collate_slide_graph)
        train2_loader = get_split_loader(train_split,  testing = args.testing,batch_size=None,collate_fn=collate_slide_graph)
        val_loader = get_split_loader(val_split,  testing = args.testing,batch_size=None,collate_fn=collate_slide_graph)
        test_loader = get_split_loader(test_split, testing = args.testing,batch_size=None,collate_fn=collate_slide_graph)
    elif args.model_type in ["vit_aggr"]:
        train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
        train2_loader = get_split_loader(train_split,  testing = args.testing)
        val_loader = get_split_loader(val_split,  testing = args.testing)
        test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        # lizx: 
        early_stopping = EarlyStoppingModelSaver(patience = float("inf"), stop_epoch=150, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    
    validate(cur,0,model,train2_loader,args.n_classes,"train2",
        early_stopping=None,writer=writer,loss_fn=loss_fn, train_dir=args.train_dir, data_threshold=args.eval_data_threshold)

    
    for epoch in range(1,args.max_epochs+1):
        train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, data_threshold=args.train_data_threshold)
        validate(cur,epoch,model,train2_loader, args.n_classes,"train2",
            early_stopping=None,writer=writer,loss_fn=loss_fn, train_dir=args.train_dir, data_threshold=args.eval_data_threshold)
        stop = validate(cur, epoch, model, val_loader, args.n_classes, "val",
            early_stopping, writer, loss_fn, args.train_dir, data_threshold=args.eval_data_threshold)

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, "best_model.pt")))
    else:
        os.makedirs(ckpt_dir,exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_dir,"best_model.pt"))

    results_dict, metric_dict, df, acc_logger= summary(model, val_loader, args.n_classes)
    val_error=1-metric_dict["eval_scores"]["accuracy"]
    val_auc=metric_dict["eval_scores"]["auc_roc_class_macro"] if args.n_classes>2 else metric_dict["eval_scores"]["auc_roc"]
    print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

    results_dict, metric_dict, df, acc_logger = summary(model, test_loader, args.n_classes)
    test_error=1-metric_dict["eval_scores"]["accuracy"]
    test_auc=metric_dict["eval_scores"]["auc_roc_class_macro"] if args.n_classes>2 else metric_dict["eval_scores"]["auc_roc"]
    print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))

    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))

        if writer and acc is not None:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
    
    writer.close()
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 

def data_subset(data,threshold):
    if isinstance(data,torch.Tensor):
        n=len(data)
        if n>threshold:
            data=data[:threshold]
    elif isinstance(data,TensorNamespace):
        n=data.n_nodes
        if n>threshold:
            data.degree=data.degree[:threshold]
            data.M=data.M[:threshold,:threshold]
            data.x=data.x[:threshold,:]
            data.n_nodes=threshold
    if n>threshold:
        print(f"data exceeded threhold ({n}), subsetting to ({threshold})")
    return data

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, data_threshold = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        if data_threshold:
            data=data_subset(data,threshold=data_threshold)
        ret_val = model(data)
        if type(ret_val)==tuple:
            logits, Y_prob, Y_hat, _, _ = ret_val
        elif type(ret_val) == dict:
            logits,Y_prob,Y_hat=ret_val["logits"],ret_val["Y_prob"],ret_val["Y_hat"]
        
        acc_logger.log(Y_hat, label)
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), data.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # backward pass
        loss.backward()

        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)
    train_error /= len(loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, set_name, early_stopping = None, writer = None, loss_fn = None, train_dir=None, data_threshold=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    ckpt_dir=os.path.join(train_dir,f"ckpt")
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            if data_threshold:
                data=data_subset(data,threshold=data_threshold)
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)


            ret_val = model(data)
            if type(ret_val)==tuple:
                logits, Y_prob, Y_hat, _, _ = ret_val
            elif type(ret_val) == dict:
                logits,Y_prob,Y_hat=ret_val["logits"],ret_val["Y_prob"],ret_val["Y_hat"]

            acc_logger.log(Y_hat, label)
            
            loss = loss_fn(logits, label)

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')

    cf_matrix=confusion_matrix(labels,prob.argmax(1))

    
    
    if writer:
        writer.add_scalar('%s/loss'%(set_name), val_loss, epoch)
        writer.add_scalar('%s/auc'%(set_name), auc, epoch)
        writer.add_scalar('%s/error'%(set_name), val_error, epoch)

    print('\n{}, loss: {:.4f}, error: {:.4f}, auc: {:.4f}'.format(set_name,val_loss, val_error, auc))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     
    print("confusion matrix:")
    print(cf_matrix)
    if early_stopping:
        assert train_dir
        early_stopping(epoch, val_error, model,routine_ckpt_dir=ckpt_dir)

        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    if args.model_type in ["graph_vit_aggr"]:
        from utils.utils import collate_slide_graph
        loader = get_simple_loader(dataset,batch_size=None,collate_fn=collate_slide_graph)
    else:
        loader = get_simple_loader(dataset)

    patient_results, metric_dict, df, _ = summary(model, loader, args.n_classes, data_threshold=args.eval_data_threshold)
    if args.n_classes==2:
        print('test_accuracy: ', metric_dict["eval_scores"]["accuracy"])
        print('auc: ', metric_dict["eval_scores"]["auc_roc"])
    else:
        print('test_accuracy: ', metric_dict["eval_scores"]["accuracy"])
        print('auc (class_micro): ', metric_dict["eval_scores"]["auc_roc_class_micro"])
        print('auc: (class_macro)', metric_dict["eval_scores"]["auc_roc_class_macro"])

    return model, patient_results, metric_dict, df

def summary(model, loader, n_classes,data_threshold=None,return_intermediates=list()):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    intermediates_tensor_lists=list()
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        if data_threshold:
            data=data_subset(data,data_threshold)
        slide_id = slide_ids.iloc[batch_idx]
        print(f"Evaluating slide {batch_idx+1}, {slide_id}")

        with torch.no_grad():
            ret_val = model(data,return_intermediates=return_intermediates)
            if type(ret_val)==tuple:
                logits, Y_prob, Y_hat, _, intermediates_tensor = ret_val
            elif type(ret_val) == dict:
                logits,Y_prob,Y_hat, intermediates_tensor=ret_val["logits"],ret_val["Y_prob"],ret_val["Y_hat"],ret_val["intermediates_tensor"]
        if return_intermediates:
            intermediates_tensor_lists.append(intermediates_tensor.numpy())
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error
    
    intermediates_numpy_dict=dict()
    for ts_nm in return_intermediates:
        intermediates_numpy_dict[ts_nm]=np.concatenate([getattr(ns,ts_nm) for ns in intermediates_tensor_lists])

    del data
    test_error /= len(loader)

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)

    metric_dict=compute_multiclass_metrics(all_labels,all_probs,n_classes)
    
    if return_intermediates:
        return patient_results, metric_dict, df, acc_logger, intermediates_numpy_dict
    else:
        return patient_results, metric_dict, df, acc_logger

def compute_multiclass_metrics(all_labels,all_probs,n_classes,return_extended=False):

    if len(np.unique(all_labels)) == 1:
        auc_score = -1
        ret_dict={
            "result_type":None,
        }
    else:
        all_pred=all_probs.argmax(axis=1)
        accuracy=np.mean(all_pred==all_labels)
        cf_matrix=confusion_matrix(all_labels,all_pred,labels=[i for i in range(n_classes)])
        if n_classes == 2:
            fpr,tpr,_=roc_curve(all_labels,all_probs[:, 1])
            auc_score=calc_auc(fpr,tpr)
            ret_dict={
                "result_type":"binary_classification",
                "eval_scores": {
                    "accuracy":float(accuracy),
                    "auc_roc":float(auc_score),
                    "confusion_matrix":cf_matrix
                }
            }

            if return_extended:
                ret_dict["extended"]={
                    "roc_fpr":fpr,
                    "roc_tpr":tpr
                }
        else:
            aucs=list()
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            if return_extended:
                roc_macro_fpr_per_class=list()
                roc_macro_tpr_per_class=list()

            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                    if return_extended:
                        roc_macro_fpr_per_class.append(fpr)
                        roc_macro_tpr_per_class.append(tpr)
                else:
                    aucs.append(float('nan'))
                    if return_extended:
                        roc_macro_fpr_per_class.append(np.array([]))
                        roc_macro_tpr_per_class.append(np.array([]))
            
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
            micro_class_auc_score = calc_auc(fpr, tpr)
            macro_class_auc_score = np.nanmean(np.array(aucs))

            ret_dict={
                "result_type":"multiclass_classification",
                "eval_scores":{
                    "auc_roc_by_class":{c:float(auc) for c,auc in zip(range(n_classes),aucs)},
                    "auc_roc_class_micro":float(micro_class_auc_score),
                    "auc_roc_class_macro":float(macro_class_auc_score),
                    "accuracy":float(accuracy),
                    "confusion_matrix":cf_matrix,
                }
            }

            if return_extended:
                ret_dict["extended"]={
                    "roc_micro_fpr":fpr,
                    "roc_micro_tpr":tpr,
                    "roc_macro_fpr_per_class":roc_macro_fpr_per_class,
                    "roc_macro_tpr_per_class":roc_macro_tpr_per_class,
                }
                
    return ret_dict
