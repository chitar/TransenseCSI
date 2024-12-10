import math

import torch
import torch.nn as nn


class AttenEmb(nn.Module):
    def __init__(self,hid,Q_inp):
        super(AttenEmb, self).__init__()
        self.emb_cqi = nn.Linear(1,hid)
        self.emb_m = nn.Linear(1,hid)
        self.emb_Q = nn.Linear(Q_inp, hid)

    def forward(self,Q,cqi,m):

        return self.emb_m(m) + self.emb_cqi(cqi) + self.emb_Q(Q)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Attention(nn.Module):
    def __init__(self,inp_d,hid_d,n_head):
        super(Attention, self).__init__()
        self.attn = nn.Linear(inp_d, hid_d*3)
        self.proj = nn.Linear(hid_d,hid_d)
        self.n_head = n_head
        self.inp_d = inp_d
        self.hid_d = hid_d

    def forward(self,x,his_data):
        '''
        :param x: [batch,n_hid]
        :param his_data:
        :return:
        '''
        q, k, v = self.attn(x).split(self.hid_d, dim=-1)  # batch,hid_d
        nB, nd = q.shape
        if his_data is not None:
            K,V = his_data # batch,n_itr,hid_d
            K = torch.cat([K,k[:,None,:]],dim=1) # (B, n_itr, hid_d)
            V = torch.cat([V,v[:,None,:]],dim=1) # (B, n_itr, hid_d)
        else:
            K,V = k[:,None,:],v[:,None,:]
        K = K.reshape(nB, -1, self.n_head, nd // self.n_head).transpose(1,2)  # (B, nh, n_itr, hd)

        V = V.reshape(nB, -1, self.n_head, nd // self.n_head).transpose(1,2)  # (B, nh, n_itr, hd)
        q = q.reshape(nB, self.n_head, nd // self.n_head, 1)  # (B, nh, hd,1)

        att = (K@q).softmax(dim=-1)  # batch,nh,n_iter,1

        v_next = V.transpose(-1,-2) @ att # batch,nh,hd,1
        v_next = v_next.reshape(nB,-1)# batch,hid_d
        v_next = self.proj(v_next) # batch,hid_d

        # reshape
        K = K.reshape(nB,-1,self.hid_d)
        V = V.reshape(nB, -1, self.hid_d)

        return K,V,v_next


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, inp_d,hid_d,n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hid_d)
        self.attn = Attention(inp_d,hid_d,n_head)
        self.ln_2 = nn.LayerNorm(hid_d)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(hid_d,  hid_d),
            c_proj  = nn.Linear(hid_d, hid_d),
            act     = NewGELU(),
            dropout = nn.Dropout(0),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x,his_data):
        K, V, v_next = self.attn(self.ln_1(x),his_data)
        v_next = x + v_next
        v_next =v_next + self.mlpf(self.ln_2(v_next))
        return  K, V, v_next



class AttenCell(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers=1,n_head=16):
        super(AttenCell, self).__init__()
        self.emb = AttenEmb(hidden_size,32*32*2)
        self.num_layers = num_layers
        layers = []

        for i in range(num_layers):
            layers.append(Block(hidden_size,hidden_size,n_head))

        self.layers = nn.ModuleList(layers)


    def forward(self,Q,cqi,m,his_data):

        his_data = [None for i in range(self.num_layers)] if his_data is None else his_data['data']
        cur_data = []
        Q = torch.view_as_real(Q).contiguous().reshape(Q.shape[0],-1)
        x = self.emb(Q,cqi[:,None],m[:,None].float())
        for i in range(self.num_layers):
            inp = x if i==0 else v_next
            K, V, v_next = self.layers[i](inp,his_data[i])
            cur_data.append((K,V))

        cur_data = {'data':cur_data,'Q_feature':v_next,'H_feature':v_next}
        return cur_data

def Atten_Q_FC(hidden_size,nB):
    return nn.Sequential(
        nn.Linear(hidden_size, 1024),
        nn.ReLU(inplace=True),
        # nn.BatchNorm1d(1024),
        nn.LayerNorm(1024),
        # nn.Linear(1024, 1024),
        # nn.ReLU(inplace=True),
        # # nn.BatchNorm1d(1024),
        # nn.Linear(1024, 1024),
        # nn.ReLU(inplace=True),
        # nn.BatchNorm1d(1024),
        nn.Linear(1024, nB * 32 * 2),
    )

def Atten_CSI_FC(hidden_size,nB):
    return nn.Sequential(
        nn.Linear(hidden_size, 1024),
        nn.ReLU(inplace=True),
        nn.LayerNorm(1024),
        # nn.BatchNorm1d(1024),
        # nn.Linear(1024, 1024),
        # nn.ReLU(inplace=True),
        # # nn.BatchNorm1d(1024),
        # nn.Linear(1024, 1024),
        # nn.ReLU(inplace=True),
        # nn.BatchNorm1d(1024),
        nn.Linear(1024, nB * 2),
    )
