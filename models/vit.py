import math
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_normal_
import utils.utils
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):

        self.norm(x)
        return self.fn(x = self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MultiheadAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., project_out=False):
        super().__init__()
        inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim,inner_dim, bias = False)
        self.to_k = nn.Linear(dim,inner_dim, bias = False)
        self.to_v = nn.Linear(dim,inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, k_inp, v_inp, attn_bias=None): # x is q_inp

        b, n, _, h = *x.shape, self.heads
        q, k, v = self.to_q(x),self.to_k(k_inp), self.to_v(v_inp)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), [q,k,v])
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if attn_bias is not None:
            assert attn_bias.shape==dots.shape, "wrong shape: attn_bias: {} recieved, {} expected".format(tuple(attn_bias.shape),tuple(dots.shape))
            dots=dots+attn_bias
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out),attn

class SelfAttention(MultiheadAttention):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    def forward(self, x, attn_bias):
        return super().forward(x,x,x,attn_bias)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        # print("dim:", dim, "heads:",heads, "dim_head:",dim_head, 'mlp_dim:',mlp_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, project_out=True)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, attn_bias=None, return_attn=False):
        attn_list=list()
        for self_attn, ff in self.layers:
            out,attn=self_attn(x,attn_bias=attn_bias)
            x = out + x
            x = ff(x) + x
            attn_list.append(attn)

        if return_attn:
            return x, attn_list
        else:
            return x

class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, project_out=True)),
                PreNorm(dim, MultiheadAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, project_out=True)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, enc_src, attn_bias=None):
        for self_attn, mh_attn, ff in self.layers:
            out,attn = self_attn(x=x,attn_bias=attn_bias)
            x = out + x
            out,mh_attn = mh_attn(x=x, k_inp=enc_src, v_inp=enc_src)

            x = out + x
            x = ff(x) + x
        # lizx: return the last layer attention
        return x,mh_attn

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class ViTAggregation(nn.Module):
    def __init__(self,n_classes,heads=4,dim_head=32,mlp_dim=128,dim=1024,depth=1,aggr="cls_token"):
        super(ViTAggregation,self).__init__()
        self.transformer=TransformerEncoder(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=0.
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes)
        )
        self.aggr=aggr
        if self.aggr=="cls_token":
            self.virtual_token_feature=nn.Parameter(torch.randn(1,1,dim,dtype=torch.float32))

    def forward(self,inp,return_intermediates=list()):

        if len(inp.shape)==2:
            inp=inp[None,:,:]
        
        intermediates_tensor=utils.utils.GenericTensorNamespace()
        if self.aggr=="cls_token":
            inp=self.__add_virtual_token_feature(inp)
        inp_feature,attn_list=self.transformer(inp,return_attn=True)

        if self.aggr=="gap":
            inp_feature=inp_feature.mean(dim=1)
        elif self.aggr=="cls_token":
            inp_feature=inp_feature[:,0,:]
        if "WSI_repr" in return_intermediates:
            intermediates_tensor.WSI_repr=inp_feature
        val=self.mlp_head(inp_feature)
        Y_hat=torch.topk(val,1,dim=1)[1]
        Y_prob=F.softmax(val,dim=1)

        return {"logits":val,"Y_prob":Y_prob,"Y_hat":Y_hat,"attn_list":attn_list,"intermediates_tensor":intermediates_tensor}

    def __add_virtual_token_feature(self,features):
        return torch.cat([self.virtual_token_feature.repeat(len(features),1,1),features],dim=1)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = self.transformer.to(device)
        self.mlp_head = self.mlp_head.to(device)

        if self.aggr=="cls_token":
            self.virtual_token_feature = nn.Parameter(self.virtual_token_feature.to(device))

class ViTPairAggregation(nn.Module):
    def __init__(self,heads=4,dim_head=32,mlp_dim=128,aggr="gap"):
        super(ViTPairAggregation,self).__init__()
        dim=1024
        self.transformer_encoder=TransformerEncoder(
            dim=dim,
            depth=1,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=0.
        )
        self.transformer_decoder=TransformerDecoder(
            dim=dim,
            depth=1,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=0.
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

        self.aggr=aggr

        if self.aggr=="cls_token":
            self.virtual_token_feature=nn.Parameter(torch.randn(1,1,dim,dtype=torch.float32))

    def __add_virtual_token_feature(self,features):
        return torch.cat([self.virtual_token_feature.repeat(len(features),1,1),features],dim=1)

    def forward(self,query_inp, subject_inp):
        if len(query_inp.shape)==2:
            query_inp=query_inp[None,:,:]
        if len(subject_inp.shape)==2:
            subject_inp=subject_inp[None,:,:]
        
        subject_emb=self.transformer_encoder(subject_inp)
        if self.aggr=="cls_token":
            query_inp=self.__add_virtual_token_feature(query_inp)
        query_emb, mh_attn=self.transformer_decoder(x=query_inp,enc_src=subject_emb)
        if self.aggr=="gap":
            aggr_feature=query_emb.mean(dim=1)
        elif self.aggr=="cls_token":
            aggr_feature=query_emb[:,0,:]
        val=self.mlp_head(aggr_feature)
        return_dict={
            'val':val,
            "mh_attn":mh_attn,
            "WSI_repr":aggr_feature,
        }
        return return_dict

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer_encoder = self.transformer_encoder.to(device)
        self.transformer_decoder = self.transformer_decoder.to(device)
        self.mlp_head = self.mlp_head.to(device)

class GraphViTAggregation(nn.Module):
    def __init__(self,
        n_classes,
        hidden_dim=1024,
        heads=4,
        dim_head=32,mlp_dim=128,
        degree_centrality_vocab_size=512,
        shortest_path_length_vocab_size=512
    ):
        super(GraphViTAggregation,self).__init__()
        self.degree_centrality_vocab_size=degree_centrality_vocab_size
        self.shortest_path_length_vocab_size=shortest_path_length_vocab_size
        self.heads=heads
        self.degree_centrality_embedder=nn.Embedding(degree_centrality_vocab_size,hidden_dim)
        self.shortest_path_length_embedder=nn.Embedding(shortest_path_length_vocab_size,self.heads)

        # for virtual node
        self.virtual_node_feature=nn.Parameter(torch.randn(1,hidden_dim,dtype=torch.float32))
        self.virtual_node_attn_bias=nn.Parameter(torch.randn(self.heads,dtype=torch.float32))
        self.transformer=TransformerEncoder(
            dim=hidden_dim,
            depth=1,
            heads=self.heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=0.
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def __node_features_add_virtual_node(self,node_features):
        return torch.cat([self.virtual_node_feature,node_features],dim=0)

    def __graph_attn_bias_add_virtual_node(self,graph_attn_bias):
        h,n,_=graph_attn_bias.shape
        new_graph_attn_bias=torch.zeros((h,n+1,n+1),dtype=torch.float32,device=graph_attn_bias.device)
        new_graph_attn_bias[:,1:,1:]=graph_attn_bias
        val=self.virtual_node_attn_bias.view(self.heads,1)
        new_graph_attn_bias[:,0,:]=val
        new_graph_attn_bias[:,1:,0]=val
        return new_graph_attn_bias

    def forward(self,data, return_intermediates=None):
        degree=torch.clamp(data.degree,0,self.degree_centrality_vocab_size)
        degree_embedding=self.degree_centrality_embedder(degree)
        node_features=degree_embedding+data.x
        node_features=self.__node_features_add_virtual_node(node_features)

        M=torch.clamp(data.M,0,self.shortest_path_length_vocab_size)
        shortest_path_length_embedding=self.shortest_path_length_embedder(M)
        graph_attn_bias=rearrange(shortest_path_length_embedding,"i j h -> h i j")
        graph_attn_bias=self.__graph_attn_bias_add_virtual_node(graph_attn_bias)
        return self.forward_assembled(node_features,graph_attn_bias)

    def forward_assembled(self,inp, attn_bias):
        if len(inp.shape)==2:
            inp=inp[None,:,:]
        if len(attn_bias.shape)==3:
            attn_bias=attn_bias[None,:,:,:]
        val=self.transformer(inp,attn_bias)
        val=val[:,0,:]
        val=self.mlp_head(val)
        Y_hat=torch.topk(val,1,dim=1)[1]
        Y_prob=F.softmax(val,dim=1)
        return val,Y_prob,Y_hat,None,None

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = self.transformer.to(device)
        self.mlp_head = self.mlp_head.to(device)
        self.degree_centrality_embedder = self.degree_centrality_embedder.to(device)
        self.shortest_path_length_embedder = self.shortest_path_length_embedder.to(device)
        self.virtual_node_feature=nn.Parameter(self.virtual_node_feature.to(device))
        self.virtual_node_attn_bias=nn.Parameter(self.virtual_node_attn_bias.to(device))

if __name__=="__main__":
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    inp=torch.rand(20000,1024)
    vita=ViTAggregation(n_classes=3)
    vita=vita.to(device)
    inp=inp.to(device)
    res=vita(inp)
    out=res[0].sum()

    out.backward()


    
    