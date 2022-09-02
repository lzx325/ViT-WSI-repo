import numpy as np
import networkx as nx
from os.path import join,basename,dirname,splitext
# from scipy.spatial import distance_matrix
import scipy.sparse as sp
from sklearn.metrics.pairwise import pairwise_distances
import h5py
import pickle as pkl

from . import algos

class SlideGraph(object):
    def __init__(
        self,
        slide_features,
        slide_coords,
        nx_graph,
        patches_attrs
    ):
        self.slide_features,self.slide_coords,self.nx_graph,self.patches_attrs=slide_features,slide_coords,nx_graph,patches_attrs
    
    def compute(self):
        adj_matrix=nx.to_numpy_array(self.nx_graph)
        fw=algos.FloydWarshallPred(adj_matrix.astype("float32"))
        fw.floyd_warshall_parallel()
        self.M=np.asarray(fw.M)
        self.Pred=np.asarray(fw.Pred)
        
    def to_pkl(self,pkl_fp):
        assert hasattr(self,"M") and hasattr(self,"Pred")
        with open(pkl_fp,'wb') as f:
            pkl.dump(
                {
                    "slide_features":self.slide_features,
                    "slide_coords":self.slide_coords,
                    "nx_graph":self.nx_graph,
                    "patches_attrs":self.patches_attrs,
                    "floyd_warshall":{
                        "M":self.M,
                        "Pred":self.Pred
                    }
                    
                },file=f
            )
    @staticmethod
    def build_slide_graph(
        patch_h5_dir,feature_h5_dir,slide_id,
        patch_dir_name,feature_dir_name,k=8
    ):
        h5_fp=join(patch_h5_dir,slide_id+".h5")
        with h5py.File(h5_fp,'r') as f:
            patches_attrs=dict(f["coords"].attrs)
        h5_fp=join(feature_h5_dir,slide_id+".h5")
        with h5py.File(h5_fp,'r') as f:
            features=f['features'][()]
            coords=f['coords'][()]
        features_dist_matrix=pairwise_distances(features,metric="cosine")
        coords_dist_matrix=pairwise_distances(coords,metric="euclidean")

        features_nn_ind=np.argsort(features_dist_matrix,axis=1)[:,1:(k+1)]
        coords_nn_ind=np.argsort(coords_dist_matrix,axis=1)[:,1:(k+1)]
        node_list1=list()
        node_list2=list()
        mat_value_list=list()
        for i in range(features_nn_ind.shape[0]):
            nn_coords=np.union1d(coords_nn_ind[i],features_nn_ind[i])
            nn_coords=nn_coords[nn_coords!=i]
            node_list1.append(nn_coords)
            node_list2.append(np.full(nn_coords.shape,i,dtype=np.int64))
            mat_value_list.append(1-features_dist_matrix[i,nn_coords])
        row_values=np.concatenate(node_list1)
        col_values=np.concatenate(node_list2)
        mat_values=np.concatenate(mat_value_list)
        coo_mat=sp.coo_matrix((mat_values,(row_values,col_values)),shape=(len(features),len(features)))
        nx_graph=nx.from_scipy_sparse_matrix(coo_mat)
        return SlideGraph(
            features,
            coords,
            nx_graph,
            patches_attrs
        )

        
        


