# -*- coding: UTF-8 -*-
# @Author  : Hanyu Li
# @Email   : hanyu-li23@mails.tsinghua.edu.cn

""" MIR
Reference:
    "Multi-Level Interaction Reranking with User Behavior History"
    Xi et al., SIGIR'2022.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseRerankerModel import RerankSeqModel
from models.general import *
from models.sequential import *
from models.developing import *

class SLAttention(nn.Module):#edit from MIR source code
    def __init__(self, v_dim, q_dim, v_seq_len, batch_size, hidden_size,fi=True, ii=True, decay=True):
        super().__init__()
        self.v_dim=v_dim
        self.q_dim=q_dim
        self.v_seq_len=v_seq_len
        self.batch_size=batch_size
        self.fi=fi
        self.ii=ii
        self.decay=decay

        self.w_b = nn.Parameter(torch.randn((1,q_dim, v_dim),dtype=torch.float32)*0.01)
        self.fc_decay1 = nn.Linear(hidden_size,32)
        self.fc_decay2 = nn.Linear(32,1)
        #self.w_v = nn.Parameter(torch.randn((v_dim, v_seq_len),dtype=torch.float32)*0.01)
        #self.w_q = nn.Parameter(torch.randn((q_dim, v_seq_len),dtype=torch.float32)*0.01)
        # parameter that has a dimension with length ‘v_seq_len’ may ruin the permutation invariant property

        self.w_v = nn.Parameter(torch.randn((v_dim, 1),dtype=torch.float32)*0.01)
        self.w_q = nn.Parameter(torch.randn((q_dim, 1),dtype=torch.float32)*0.01)
    
    def forward(self, V, Q, time, usr_prof, his_max):
        v_dim, q_dim = self.v_dim, self.q_dim
        v_seq_len, q_seq_len = self.v_seq_len, his_max
        bat_size = V.shape[0]

        # get affinity matrix
        if self.fi:
            C2=0# not implemented
        if self.ii:
            C1 = torch.matmul(Q, torch.matmul(self.w_b.repeat(bat_size,1,1), torch.transpose(V, 1, 2)))
            if self.fi:
                C1 = C1 + C2
        else:
            C1 = C2

        if self.decay:
            # decay
            pos = time.unsqueeze(2).repeat(1,1,v_seq_len)
            theta = F.leaky_relu(self.fc_decay2(F.leaky_relu(self.fc_decay1(usr_prof))))
            decay_theta = theta.unsqueeze(2).repeat(1, q_seq_len, v_seq_len)
            pos_decay = torch.exp(-decay_theta * pos)
            C = torch.tanh(C1 * pos_decay + C1)
        else:
            C = C1

        # attention map
        hv_1 = torch.reshape(torch.matmul(torch.reshape(V, [-1, v_dim]), self.w_v.repeat(1,self.v_seq_len)), [-1, v_seq_len, v_seq_len])
        hq_1 = torch.reshape(torch.matmul(torch.reshape(Q, [-1, q_dim]), self.w_q.repeat(1,self.v_seq_len)), [-1, q_seq_len, v_seq_len])
        hq_1 = torch.transpose(hq_1, 1, 2)
        h_v = torch.tanh(hv_1 + torch.matmul(hq_1, C))
        h_q = torch.tanh(hq_1 + torch.matmul(hv_1, torch.transpose(C, 1, 2)))
        # h_v = tf.nn.tanh(tf.matmul(hq_1, C))
        # h_q = tf.nn.tanh(tf.matmul(hv_1, tf.transpose(C, perm=[0, 2, 1])))
        a_v = torch.softmax(h_v, dim=-1)
        a_q = torch.softmax(h_q, dim=-1)
        
        v = torch.matmul(a_v, V)
        q = torch.matmul(a_q, Q)

        return v, q

class MIRBase(object): # MIR must have sequential input, but can use general or sequential baseranker
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of item embedding vectors.')
        parser.add_argument('--num_heads', type=int, default=4,
                            help='Number of attention heads.')
        parser.add_argument('--num_hidden_unit', type=int, default=64,
                            help='Number of hidden units in attention and BiLSTM.')
        return parser

    def _base_init(self, args, corpus):
        self.args = args
        self.emb_size = args.emb_size
        self.num_heads = args.num_heads
        self.num_hidden_unit = args.num_hidden_unit
        self.cand_size = args.train_max_pos_item+args.train_max_neg_item

        self.dropout = args.dropout

        self.corpus=corpus

        self._base_define_params()
        self.apply(self.init_weights)

    def _base_define_params(self):
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.intra_set = nn.MultiheadAttention(embed_dim=self.emb_size + self.ranker_emb_size, num_heads=self.num_heads,dropout=self.dropout)
        self.intra_list = torch.nn.LSTM(input_size = self.emb_size + self.ranker_emb_size, hidden_size = self.num_hidden_unit, bidirectional = True, dropout = self.dropout, batch_first = True, num_layers = 1)
        self.ln = nn.LayerNorm(self.emb_size * 4 + self.ranker_emb_size * 4 + self.num_hidden_unit * 2, elementwise_affine=False)
        self.fc1 = nn.Linear(self.emb_size * 4 + self.ranker_emb_size * 4 + self.num_hidden_unit * 2, 500)
        self.fc2 = nn.Linear(500,200)
        self.fc3 = nn.Linear(200,80)
        self.fc4 = nn.Linear(80,1)
        #self.rFF0 = nn.Linear(self.emb_size + self.ranker_emb_size * 2, self.num_hidden_unit, bias=True)
        self.SLAttention = SLAttention(self.emb_size * 2 + self.ranker_emb_size * 2, self.emb_size + self.ranker_emb_size + self.num_hidden_unit * 2, self.cand_size, self.args.batch_size, self.emb_size)


    def forward(self, feed_dict):
        batch_size = feed_dict['item_id'].shape[0]
        history_max = feed_dict['history_items'].shape[1]
        
        i_ids = feed_dict['item_id']  # [batch_size, sample_num]
        his_ids = feed_dict['history_items'] # [batch_size, max_his_length]
        #u_ids = feed_dict['user_id']  # [batch_size, sample_num] Here this should be switched to PV vector, which is from the pretrained model and represents the relationship between u and i

        i_vectors = self.i_embeddings(i_ids)  # [batch_size, sample_num, emb_size]
        i_b_vectors = feed_dict['i_v']#candidate embedding of the base ranker model [batch_size, base_emb_size]
        i_v = torch.cat((i_vectors,i_b_vectors),dim=2) # [batch_size, sample_num, emb_size + base_emb_size]

        his_vectors = self.i_embeddings(his_ids) # [batch_size, history_length, emb_size]
        his_b_vectors = feed_dict['his_v']#the embedding of base ranker model for history items, [batch_size, history_length, base_emb_size]
        his_v = torch.cat((his_vectors,his_b_vectors),dim=2) # [batch_size, history_length, emb_size + base_emb_size]

        seq_v = feed_dict['u_v'][:,0,:]# used as the user profile in MIR, [batch_size, base_emb_size]

        padding_mask = feed_dict['padding_mask']
        his_mask = torch.where(his_ids==0,torch.ones_like(his_ids,device=his_ids.device),torch.zeros_like(his_ids,device=his_ids.device)).bool()

        '''intra-set interaction (candidate set)'''# leak safe
        intra_set=1
        if intra_set:
            i_v=torch.transpose(i_v, 0, 1)
            attn_i = self.intra_set(i_v, i_v, i_v, key_padding_mask=padding_mask, need_weights=False)[0]
            attn_i=torch.transpose(attn_i, 0, 1)
            padding_mask_m = (1-padding_mask.float()).unsqueeze(2).repeat(1,1,2*self.emb_size)
            attn_i = attn_i*padding_mask_m # [batch_size, sample_num, emb_size + base_emb_size]
            i_v = torch.transpose(i_v, 0, 1) # [batch_size, sample_num, emb_size + base_emb_size]
            seq = torch.cat([i_v, attn_i], dim=2) # [batch_size, candidate_len, emb*2] emb = emb_size + base_emb_size
        else:
            seq = torch.cat([i_v, i_v], dim=2)

        '''intra_list interaction (history)'''#usr_seq may leak!
        intra_list=1
        if intra_list:
            bilstm_his,_ = self.intra_list(his_v,None)# [batch_size, his_len, hidden*2]
            usr_seq = torch.cat([bilstm_his, his_v], 2)# [batch_size, his_len, hidden*2 + emb] emb = emb_size + base_emb_size
        else:
            usr_seq = torch.cat([torch.zeros((his_v.shape[0],his_v.shape[1],self.num_hidden_unit*2)).to(his_v.device), his_v], 2)

        set2list=1
        if set2list:
            times=(feed_dict['history_times']>0).float() # if time is not padding
            tmax = torch.max(feed_dict['history_times'],dim=1).values.unsqueeze(1).repeat(1,feed_dict['history_times'].shape[1])-feed_dict['history_times'] # time interval
            tmax = torch.log2(tmax+1) #following MIR github repo
            tmax = tmax+torch.max(tmax,dim=1).values.unsqueeze(1).repeat(1,feed_dict['history_times'].shape[1])+1 #following MIR github repo
            v, q = self.SLAttention(seq, usr_seq, tmax.float()*times, seq_v, history_max) # v: [batch_size, candidate_len, emb*2], q: [batch_size, candidate_len, hidden*2 + emb], emb = emb_size + base_emb_size

            fin = torch.cat([v, q], dim=2) # [batch_size, candidate_len, emb*3 + hidden*2]
        else:
            fin = torch.cat([seq, 0*usr_seq[:,:40,:]], dim=2)#here if usr_seq is no 0, will leak because introduce position information(pos neg related)

        final_embed = torch.cat([i_v, fin], dim=2) # [batch_size, candidate_len, emb*4 + hidden*2]
        final_embed = self.ln(final_embed)
        final_embed = F.dropout(F.relu(self.fc1(final_embed)),self.dropout)
        final_embed = F.dropout(F.relu(self.fc2(final_embed)),self.dropout)
        final_embed = F.dropout(F.relu(self.fc3(final_embed)),self.dropout)
        prediction = self.fc4(final_embed)
        
        return {'prediction': prediction.view(batch_size, -1)}


class MIRGeneral(RerankSeqModel, MIRBase): # MIR must have sequential input, but can use general or sequential baseranker
    reader = 'ImpressionSeqReader'
    runner = 'ImpressionRunner'

    @staticmethod
    def parse_model_args(parser):
        parser = MIRBase.parse_model_args(parser)
        return RerankSeqModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        RerankSeqModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        return MIRBase.forward(self, feed_dict)

class MIRSequential(RerankSeqModel, MIRBase): # MIR must have sequential input, but can use general or sequential baseranker
    reader = 'ImpressionSeqReader'
    runner = 'ImpressionRunner'

    @staticmethod
    def parse_model_args(parser):
        parser = MIRBase.parse_model_args(parser)
        return RerankSeqModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        RerankSeqModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        return MIRBase.forward(self, feed_dict)