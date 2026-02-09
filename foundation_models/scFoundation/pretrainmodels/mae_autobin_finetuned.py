import torch
from torch import nn
from .transformer import pytorchTransformerModule

def gatherData(data, labels, pad_token_id):
    value_nums = labels.sum(1)
    max_num = max(value_nums)


    fake_data = torch.full((data.shape[0], max_num), pad_token_id,
                           device=data.device)
    data = torch.hstack([data, fake_data])

    fake_label = torch.full((labels.shape[0], max_num), 1,
                            device=labels.device)
    none_labels = ~labels
    labels = labels.float()
    labels[none_labels] = torch.tensor(-float('Inf'), device=labels.device)

    tmp_data = torch.tensor([(i + 1) * 20000 for i in range(labels.shape[1], 0, -1)], device=labels.device)
    labels += tmp_data

    labels = torch.hstack([labels, fake_label])

    fake_label_gene_idx = labels.topk(max_num).indices

    new_data = torch.gather(data, 1, fake_label_gene_idx)

    padding_labels = (new_data == pad_token_id)

    return new_data, padding_labels


def exists(val):
    return val is not None

class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(self, dim, max_seq_len, bin_num, bin_alpha, mask_token_id = None, pad_token_id = None):
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.bin_num = bin_num
        self.bin_alpha = bin_alpha
        
        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)
        
        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)
        
        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        # print('self.bin_num_idx',self.bin_num_idx, self.bin_num_idx.shape)

        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=0):
        x_mask_idx = (x==self.mask_token_id).nonzero()
        x_pad_idx = (x==self.pad_token_id).nonzero()
        # print("x_mask",x_mask_idx.shape,x_mask_idx)
        
        x = self.mlp(x) # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x) # [B,N,H]
        x_crosslayer = self.mlp2(x) # [B,N,H]
        x = self.bin_alpha * x + x_crosslayer # [B,N,H]
        weight = self.Softmax(x) # [B, N, H]
        # print('weight', weight.shape, weight, torch.sum(weight, 2))
        
        bin_num_idx = self.bin_num_idx.to(x.device) # [H,]
        # print('bin_num_idx', bin_num_idx.shape)
        
        token_emb = self.emb(bin_num_idx) # [H, D]
        # print('token_emb', token_emb.shape)
        x = torch.matmul(weight, token_emb) #[B, N, D]
    
        # print("x_emb",x.shape,x)
        
        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)

        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)
        # print(mask_token_emb.dtype)
        # print("x", x.dtype)
        x[x_mask_idx[:,0],x_mask_idx[:,1],:] = mask_token_emb.repeat(x_mask_idx.shape[0],1)
        # print("x_emb",x.shape,x)

        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        x[x_pad_idx[:,0],x_pad_idx[:,1],:] = pad_token_emb.repeat(x_pad_idx.shape[0],1)
    
        if output_weight:
            return x,weight
        return x

class RandomPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


class MaeAutobin_finetuned(nn.Module):
    def __init__(
            self,
            *,
            max_seq_len=19266,  # max length of sequence
            embed_dim=768,  # encoder dim of tokens
            bin_alpha = 1.0,
            bin_num = 100,
            pad_token_id = 103,
            mask_token_id = 102,
    ):
        super(MaeAutobin_finetuned, self).__init__()

        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

        # encoder
        self.token_emb = AutoDiscretizationEmbedding2(embed_dim, max_seq_len, bin_num=bin_num, bin_alpha=bin_alpha, pad_token_id=self.pad_token_id, mask_token_id=self.mask_token_id)
        self.pos_emb = nn.Embedding(max_seq_len+1, embed_dim) 

        self.encoder = pytorchTransformerModule(
            max_seq_len=max_seq_len,
            dim=embed_dim,
            depth=12,
            heads=12
        )
        
        # mlp
        self.fc1 = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # ['n_class']
        ) 
        self.norm = torch.nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6)
        
    def emb(self, X):
        x = X
        value_labels = x > 0
        x, x_padding = gatherData(x, value_labels, self.pad_token_id)
        data_gene_ids = torch.arange(19264, device=x.device).repeat(x.shape[0], 1)
        position_gene_ids, _ = gatherData(data_gene_ids, value_labels,
                                        self.pad_token_id)
        
        embedding = self.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0)
        position_emb = self.pos_emb(position_gene_ids)
        embedding += position_emb

        return embedding, x_padding
    
    def forward(self, X):
        embedding, x_padding = self.emb(X)
        logits = self.encoder(embedding, x_padding)

        # mlp
        logits_pooled = logits.mean(dim=1)
        output = self.fc1(logits_pooled)

        return output