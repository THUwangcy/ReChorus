# -*- coding: UTF-8 -*-
# @Author  : Hanyu Li
# @Email   : hanyu-li23@mails.tsinghua.edu.cn

""" SetRank
Reference:
    "SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval"
    Pang et al., SIGIR'2020.
"""
import torch
import torch.nn as nn
from models.BaseRerankerModel import RerankModel
from models.BaseRerankerModel import RerankSeqModel
from models.general import *
from models.sequential import *
from models.developing import *
from torch.nn import functional as F

class LearnedPositionEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0,max_len = 50):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0),:]
        return self.dropout(x)

class MAB(nn.Module):#copy from torch source code
    def __init__(self, d_model, nhead, d_feedforward, dropout):
        super().__init__()
        self.attn=nn.MultiheadAttention(embed_dim=d_model,num_heads=nhead,dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, Q, K, V, src_mask = None,src_key_padding_mask = None):
        x = self.norm1(Q + self.attn_block(Q, K, V, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x


    # self-attention block
    def attn_block(self, Q, K, V, attn_mask, key_padding_mask):
        x = self.attn(Q, K, V,attn_mask=attn_mask,key_padding_mask=key_padding_mask,need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.dropout2(x)

class MSAB(nn.Module):
    def __init__(self, d_model, nhead, d_feedforward, dropout):
        super().__init__()
        self.MAB1=MAB(d_model, nhead, d_feedforward, dropout)

    def forward(self, x, src_mask = None,src_key_padding_mask = None):
        x = self.MAB1(x, x, x, src_mask, src_key_padding_mask)
        return x

class IMSAB(nn.Module):
    def __init__(self, d_model, nhead, d_feedforward, dropout, m_clusters=20):
        super().__init__()
        self.MAB1=MAB(d_model, nhead, d_feedforward, dropout)
        self.MAB2=MAB(d_model, nhead, d_feedforward, dropout)
        self.I=nn.Parameter(torch.Tensor(m_clusters,d_model))
        nn.init.normal_(self.I, mean=0.0, std=0.01)#I is not linear weight nor embedding, so it needs to be initialized specifically

    def forward(self, x, src_mask = None,src_key_padding_mask = None):
        #I_r=torch.repeat_interleave(self.I.unsqueeze(1),x.shape[1],dim=1)
        I_r=self.I.unsqueeze(1).repeat(1,x.shape[1],1)#same above
        h = self.MAB1(I_r, x, x, src_mask, src_key_padding_mask)
        x = self.MAB2(x, h, h, src_mask, None)
        return x

class SetRankBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of item embedding vectors.')
        parser.add_argument('--n_blocks', type=int, default=4,
                            help='num of blocks of MSAB/IMSAB')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--num_hidden_unit', type=int, default=64,
                            help='Number of hidden units in Transformer layer.')
        parser.add_argument('--setrank_type', type=str, default='IMSAB',
                            help='msab or imsab')
        return parser

    def _base_init(self, args, corpus):
        self.args = args
        self.emb_size = args.emb_size
        self.n_blocks = args.n_blocks
        self.num_heads = args.num_heads
        self.num_hidden_unit = args.num_hidden_unit
        self.setrank_type = args.setrank_type
        self.positionafter = 1

        self.dropout = args.dropout

        self.corpus=corpus

        self._base_define_params()
        self.apply(self.init_weights)

    def _base_define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        if self.positionafter == 0:
            self.ordinal_position_embedding = nn.Embedding(self.train_max_neg_item + self.train_max_pos_item, self.emb_size + self.ranker_emb_size * 2) # reranker and ranker embedding
        else:
            self.ordinal_position_embedding=nn.Embedding(self.train_max_neg_item+self.train_max_pos_item,self.num_hidden_unit)
        self.rFF0 = nn.Linear(self.emb_size + self.ranker_emb_size * 2, self.num_hidden_unit, bias=True)
        if self.setrank_type=='MSAB':
            self.encoder = nn.ModuleList([MSAB(d_model=self.num_hidden_unit, nhead=self.num_heads, d_feedforward=128, dropout=self.dropout) for _ in range(self.n_blocks)])
        elif self.setrank_type=='IMSAB':
            self.encoder = nn.ModuleList([IMSAB(d_model=self.num_hidden_unit, nhead=self.num_heads, d_feedforward=128, dropout=self.dropout, m_clusters=20) for _ in range(self.n_blocks)])
        self.rFF1 = nn.Linear(self.num_hidden_unit, 1, bias=True)

    def forward(self, feed_dict):
        batch_size = feed_dict['item_id'].shape[0]
        #history_max = feed_dict['history_items'].shape[1]
        
        i_ids = feed_dict['item_id']  # [batch_size, sample_num]
        #u_ids = feed_dict['user_id']  # [batch_size, sample_num] Here this should be switched to PV vector, which is from the pretrained model and represents the relationship between u and i

        i_vectors = self.i_embeddings(i_ids)  # [batch_size, sample_num, emb_size]
        u_vectors = torch.cat([feed_dict['u_v'],feed_dict['i_v']],dim=2)#pv, consist of sequence vector and candidate embedding of the base ranker model, attaching the sequence embedding to every candidate emb [batch_size, sequence_emb+item_emb]
        #score = feed_dict['scores']

        di = torch.cat((i_vectors,u_vectors),dim=2)
        pi = self.ordinal_position_embedding(feed_dict['position'])  #learnable position encoding

        if self.positionafter==0:
            xi = di+pi
            xi = self.rFF0(xi)  #reshape dimension to num_hidden_unit
        else:
            xi = self.rFF0(di)  #reshape dimension to num_hidden_unit
            xi = xi+pi
        
        padding_mask = feed_dict['padding_mask']

        xi = torch.transpose(xi, 0, 1)
        for block in self.encoder:
            xi = block(xi, None, padding_mask)

        prediction = self.rFF1(xi)
        prediction = torch.transpose(prediction, 0, 1)
        
        return {'prediction': prediction.view(batch_size, -1)}

class SetRankGeneral(RerankModel, SetRankBase):
    reader = 'ImpressionReader'
    runner = 'ImpressionRunner'

    @staticmethod
    def parse_model_args(parser):
        parser = SetRankBase.parse_model_args(parser)
        return RerankModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        RerankModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        return SetRankBase.forward(self, feed_dict)

class SetRankSequential(RerankSeqModel, SetRankBase):
    reader = 'ImpressionSeqReader'
    runner = 'ImpressionRunner'

    @staticmethod
    def parse_model_args(parser):
        parser = SetRankBase.parse_model_args(parser)
        return RerankSeqModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        RerankSeqModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        return SetRankBase.forward(self, feed_dict)