import torch
from torch import nn
from torch.nn import functional as F

class PatchEmbed(nn.Module):
    """
    split image with Linear projection

    Args:
        img_size (tuple<int,int>) - size of images
        patch_size (tuple<int,int>) - size of patch
        in_channels (int)
        embed_dim (int)  - embedding dimension of each patch

    Attributes:
        n_patches (int) - number of patches
        proj (nn.Module) - projection network
    """
    def __init__(self,img_size,patch_size,in_channels=3,embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0]//patch_size[0])*(img_size[1]//patch_size[1])
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.flatten = nn.Flatten(2)

    def forward(self,X):
        X = self.proj(X)
        X = self.flatten(X)
        X = X.transpose(1,2)
        return X

class Attention(nn.Module):
    """
    Attention module

    Args:
        dim (int) - embedding dimension of each patch
        n_heads (int) - number of heads
        qkv_bias (bool)
        attn_p (float) - dropout rate of attention layer
        proj_p (float) - dropout rate of linear layer

    Attributes:
        dim (int)  - embedding dimension of each patch
        n_heads (int) - number of heads
        head_dim  (int)  - embedding dimension for each head
        scale (float) - scale factor for qkv
        qkv (nn.Module) - QKV matrices
        proj (nn.Module) - projection layer
        attn_drop (nn.Module) - dropout of attention layer
        proj_drop (nn.Module) - dropout of linear layer
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0.0, proj_p=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim//n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias) # 3 for q, k, v
        self.attn_drop = nn.Dropout(attn_p)
        self.proj= nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self,X):
        n_sample,n_token,dim = X.shape

        if dim != self.dim: raise ValueError()

        qkv = self.qkv(X)
        qkv = qkv.reshape(n_sample,n_token,3,self.n_heads,self.head_dim)
        qkv = qkv.permute(2,0,3,1,4) # (3,n_sample,n_heads,n_patch,head_dim)
        q,k,v = qkv[0],qkv[1],qkv[2] # (n_sample,n_heads,n_patch,head_dim)
        k_t = k.transpose(-2,-1)

        dp = (q @ k_t)*self.scale # (n_sample,n_heads,n_patch,n_patch)
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)
        weighted_avg = attn @ v # (n_sample,n_heads,n_patch,head_dim)
        weighted_avg = weighted_avg.transpose(1,2) # (n_sample,n_patch,n_heads,head_dim)
        weighted_avg = weighted_avg.flatten(2) # (n_sample,n_patch,n_dim)
        X = self.proj(weighted_avg)
        X = self.proj_drop(X)
        return X

class MLP(nn.Module):
    """
    Multi-Layer Perceptron

    Args:
        in_features - number of input features
        hidden_features  - number of hidden features
        out_features  - number of output features
        p (float) - dropout rate of output features

    Attributes:
        fc1 (nn.Module) - 1st layer
        act (nn.Module) - activtion of 1st layer
        fc2 (nn.Module) - 2nd layer
        drop (nn.Module) - dropout of 2nd layer
    """
    def __init__(self,in_features,hidden_features,out_features,p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop = nn.Dropout(p)

    def forward(self,X):
        X = self.fc1(X)
        X = self.act(X)
        X = self.fc2(X)
        X = self.drop(X)
        return X

class Block(nn.Module):
    """
    Block

    Args:
        dim (int) - embedding dimension of each patch
        n_heads (int) - number of heads
        mlp_ratio (int) - ratio of hidden neuron in mlp to embedding dimension
        qkv_bias (bool)
        attn_p (float) - dropout rate of attention layer
        p (float) - dropout rate of linear layer

    Attributes:
        norm1 (nn.Module) - 1st Layer Normalization
        attn (nn.Module) - Attention Module
        norm2 (nn.Module) - 2nd Layer Normalization
        mlp (nn.Module) - Multi-Layer Perceptron module
    """
    def __init__(self,dim,n_heads,mlp_ratio=4.0,qkv_bias=False,p=0.0,attn_p=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim,eps=1e-6)
        self.attn = Attention(
            dim=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim,eps=1e-6)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(mlp_ratio*dim),
            out_features=dim
        )

    def forward(self,X):
        X = X + self.attn(self.norm1(X))
        X = X + self.mlp(self.norm2(X))
        return X

class VisionTransformer(nn.Module):
    """
    Pytorch implementation of Vision Transformer (Dosovitskiy 2020)
    return size: 512

    Args:
        img_size (tuple<int,int>) - size of images
        patch_size (tuple<int,int>) - size of patch
        in_channels (int)
        n_classes (int) - number of classes
        embed_dim (int) - embedding dimension of each patch
        depth (int) -
        n_heads (int) - number of heads
        mlp_ratio (int) - ratio of hidden neuron in mlp to embedding dimension
        qkv_bias (bool)
        attn_p (float) - dropout rate of attention layer
        p (float) - dropout rate of linear layer

    Attributes:
        patch_embed (nn.Module)
        cls_token (nn.Parameter)
        pos_embed (nn.Parameter)
        pos_drop (nn.Module)
        blocks (nn.ModuleList)
        norm (nn.Module)
        head (nn.Module)
    """
    def __init__(self,img_size,patch_size,in_channels,n_classes,
                 embed_dim=512,depth=4,n_heads=16,qkv_bias=False,attn_p=0.0,p=0.0,mlp_ratio=4.0):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size,patch_size,in_channels,embed_dim)
        self.cls_token = nn.Parameter(
            torch.zeros(1,1,embed_dim)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1,1+self.patch_embed.n_patches,embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(
                dim=embed_dim,
                n_heads=n_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,p=p,attn_p=attn_p
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim,eps=1e-6)
        # self.head = nn.Linear(embed_dim,n_classes)

    def forward(self,X):
        n_samples = X.shape[0]
        X = self.patch_embed(X)
        cls_token = self.cls_token.expand(n_samples,-1,-1)
        X = torch.cat((cls_token,X),dim=1)
        X = X + self.pos_embed
        X = self.pos_drop(X)

        for block in self.blocks:
            X = block(X)

        X = self.norm(X)
        X = X[:,0]
        # X = self.head(X)

        return X




class AttentionST(nn.Module):
    """
    Decoder Block 2nd Attention module (accepting source and target)

    Args:
        dim (int) - embedding dimension of each patch
        n_heads (int) - number of heads
        qkv_bias (bool)
        attn_p (float) - dropout rate of attention layer
        proj_p (float) - dropout rate of linear layer

    Attributes:
        dim (int)  - embedding dimension of each patch
        n_heads (int) - number of heads
        head_dim  (int)  - embedding dimension for each head
        scale (float) - scale factor for qkv
        qkv (nn.Module) - QKV matrices
        proj (nn.Module) - projection layer
        attn_drop (nn.Module) - dropout of attention layer
        proj_drop (nn.Module) - dropout of linear layer
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0.0, proj_p=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim//n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_t = nn.Linear(dim,dim*3,bias=qkv_bias) # 3 for q, k, v
        self.qkv_s = nn.Linear(dim,dim*3,bias=qkv_bias) # 3 for q, k, v
        self.attn_drop = nn.Dropout(attn_p)
        self.proj= nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self,T,S):
        assert T.shape == S.shape
        n_sample,n_token,dim = T.shape

        if dim != self.dim: raise ValueError()

        qkv_t = self.qkv_t(T).reshape(n_sample,n_token,3,self.n_heads,self.head_dim)
        qkv_s = self.qkv_s(S).reshape(n_sample,n_token,3,self.n_heads,self.head_dim)
        qkv_t = qkv_t.permute(2,0,3,1,4) # (3,n_sample,n_heads,n_patch,head_dim)
        qkv_s = qkv_s.permute(2,0,3,1,4) # (3,n_sample,n_heads,n_patch,head_dim)
        _,_,v = qkv_t[0],qkv_t[1],qkv_t[2] # (n_sample,n_heads,n_patch,head_dim)
        q,k,_ = qkv_s[0],qkv_s[1],qkv_s[2] # (n_sample,n_heads,n_patch,head_dim)
        k_t = k.transpose(-2,-1)

        dp = (q @ k_t)*self.scale # (n_sample,n_heads,n_patch,n_patch)
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)
        weighted_avg = attn @ v # (n_sample,n_heads,n_patch,head_dim)
        weighted_avg = weighted_avg.transpose(1,2) # (n_sample,n_patch,n_heads,head_dim)
        weighted_avg = weighted_avg.flatten(2) # (n_sample,n_patch,n_dim)
        X = self.proj(weighted_avg)
        X = self.proj_drop(X)
        return X

class DecoderBlock(nn.Module):
    """
    Decoder Block

    Args:
        dim (int) - embedding dimension of each patch
        n_heads (int) - number of heads
        mlp_ratio (int) - ratio of hidden neuron in mlp to embedding dimension
        qkv_bias (bool)
        attn_p (float) - dropout rate of attention layer
        p (float) - dropout rate of linear layer

    Attributes:
        norm1 (nn.Module) - 1st Layer Normalization
        attn1 (nn.Module) - Attention Module
        norm2 (nn.Module) - 2nd Layer Normalization
        norms (nn.Module) - Source Normalization
        attn2 (nn.Module) - Attention Module (decoder)
        norm2 (nn.Module) - 3nd Layer Normalization
        mlp (nn.Module) - Multi-Layer Perceptron module
    """
    def __init__(self,dim,n_heads,mlp_ratio=4.0,qkv_bias=False,p=0.0,attn_p=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim,eps=1e-6)
        self.attn1 = Attention(
            dim=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim,eps=1e-6)
        self.attn2 = AttentionST(
            dim=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attn_p=attn_p,
            proj_p=p
        )
        self.norm3 = nn.LayerNorm(dim,eps=1e-6)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(mlp_ratio*dim),
            out_features=dim
        )

    def forward(self,X,S):
        X = X + self.attn1(self.norm1(X))
        X = X + self.attn2(self.norm2(X),S)
        X = X + self.mlp(self.norm3(X))


        return X

class TransformerEncoder(nn.Module):
    """
    Transformer (Encoder)

    Args:
        seq_len (int) - length of sequence
        embed_dim (int) - embedding dimension of each patch
        depth (int) -
        n_heads (int) - number of heads
        mlp_ratio (int) - ratio of hidden neuron in mlp to embedding dimension
        qkv_bias (bool)
        attn_p (float) - dropout rate of attention layer
        p (float) - dropout rate of linear layer

    Attributes:
        patch_embed (nn.Module)
        cls_token (nn.Parameter)
        pos_embed (nn.Parameter)
        pos_drop (nn.Module)
        blocks (nn.ModuleList)
        norm (nn.Module)
        head (nn.Module)
    """
    def __init__(self,seq_len,embed_dim=512,depth=4,n_heads=16,qkv_bias=False,attn_p=0.0,p=0.0,mlp_ratio=4.0,end=True):
        super().__init__()

        self.cls_token = nn.Parameter(
            torch.zeros(1,1,embed_dim)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1,1+seq_len,embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(
                dim=embed_dim,
                n_heads=n_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,p=p,attn_p=attn_p
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim,eps=1e-6)
        self.end = end

    def forward(self,X):
        n_samples = X.shape[0]
        cls_token = self.cls_token.expand(n_samples,-1,-1)
        X = torch.cat((cls_token,X),dim=1)
        X = X + self.pos_embed
        X = self.pos_drop(X)

        for block in self.blocks:
            X = block(X)

        X = self.norm(X)
        if self.end:
            X = X[:,0]

        return X

class TransformerDecoder(nn.Module):
    """
    Transformer (Decoder)

    Args:
        seq_len (int) - length of sequence
        embed_dim (int) - embedding dimension of each patch
        depth (int) -
        n_heads (int) - number of heads
        mlp_ratio (int) - ratio of hidden neuron in mlp to embedding dimension
        qkv_bias (bool)
        attn_p (float) - dropout rate of attention layer
        p (float) - dropout rate of linear layer

    Attributes:
        patch_embed (nn.Module)
        cls_token (nn.Parameter)
        pos_embed (nn.Parameter)
        pos_drop (nn.Module)
        blocks (nn.ModuleList)
        norm (nn.Module)
        head (nn.Module)
    """
    def __init__(self,seq_len,embed_dim=512,depth=4,n_heads=16,qkv_bias=False,attn_p=0.0,p=0.0,mlp_ratio=4.0,end=True):
        super().__init__()

        self.cls_token = nn.Parameter(
            torch.zeros(1,1,embed_dim)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1,1+seq_len,embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                DBlock(
                dim=embed_dim,
                n_heads=n_heads,mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,p=p,attn_p=attn_p
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim,eps=1e-6)
        self.end = end

    def forward(self,X,S):
        n_samples = X.shape[0]
        cls_token = self.cls_token.expand(n_samples,-1,-1)
        X = torch.cat((cls_token,X),dim=1)
        X = X + self.pos_embed
        X = self.pos_drop(X)

        for block in self.blocks:
            X = block(X,S)

        X = self.norm(X)

        if self.end:
            X = X[:,0]

        return X

class EncoderDecoderTransformer(nn.Module):

    def __init__(self,
        seq_len,embed_dim=512,depth=4,n_heads=16,qkv_bias=False,attn_p=0.0,p=0.0,mlp_ratio=4.0
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            seq_len,
            embed_dim,
            depth,
            n_heads,
            qkv_bias,
            attn_p,
            p,
            mlp_ratio,
            end = False
        )
        self.decoder = TransformerDecoder(
            seq_len,
            embed_dim,
            depth,
            n_heads,
            qkv_bias,
            attn_p,
            p,
            mlp_ratio,
            end = False
        )

    def forward(self,S,T):
        S = self.decoder(T, self.encoder(S))
        return self.decoder(T, self.encoder(S))


class TransformerWrapper(nn.Module):

    def __init__(self,embed,transformer):
        super().__init__()
        self.embed = embed
        self.transformer = transformer

    def forward(self,X):
        return self.transformer(self.embed(X))
