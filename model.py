import torch
import torch.nn as nn
import scipy.io as io
import numpy as np
from quantization import Quantizer,SoftmaxQuantizer,LinearQuantizer
from LSTM_module import LSTMCell,LSTM_CSI_FC,LSTM_Q_FC
from atten_module import AttenCell,Atten_Q_FC,Atten_CSI_FC

def UE_feedback(V,Q,H):

    He = H@Q
    prod = He @ V
    prod = prod.squeeze(dim=1).abs()
    m = prod.max(dim=1)[1]
    v = V[:,m].t()
    cqi = He @ v[:,:,None]
    cqi = cqi.abs().squeeze(-1).squeeze(-1)

    return cqi,m,v






class CSICell(nn.Module):
    def __init__(self,hidden_size,Na = 32, Np=8,n_layer = 2,model='Atten',datapath='',qtFlag='None',Q_mod='eye',nB=16,n_bit=4,n_head=16):
        super(CSICell, self).__init__()
        self.qtFlag = qtFlag
        self.Np = Np
        self.Na = Na
        vData = io.loadmat(datapath)
        vKey = 'M_CB_T1_s' if 'M_CB_T1_s' in vData.keys() else 'T1_CB_woP'
        self.V = torch.from_numpy(np.array(vData[vKey])).type(torch.cfloat).cuda()
        self.hidden_size = hidden_size
        self.cell =LSTMCell(Na * 2, hidden_size,num_layers=n_layer) if model=='LSTM' else AttenCell(Na*2,hidden_size,num_layers=n_layer,n_head=n_head)
        self.Q_fc = LSTM_Q_FC(hidden_size,32) if model=='LSTM' else Atten_Q_FC(hidden_size,32)
        self.csi_fc = LSTM_CSI_FC(hidden_size,nB) if model=='LSTM' else Atten_CSI_FC(hidden_size,nB)
        self.basis = LSTM_Q_FC(hidden_size,nB) if model=='LSTM' else Atten_Q_FC(hidden_size,nB)

        self.nB = nB
        self.quantizer = SoftmaxQuantizer(n_bit=n_bit) if 'softmax' in qtFlag else LinearQuantizer(n_bit=n_bit)
        self.q_mod = Q_mod
        self.Q = nn.Parameter(torch.randn(Na,Na,2))

    def forward(self,H,Q,dec_hcs_pre,R):

        # --convert to complex number
        H = H.reshape(H.shape[0], 1, self.Na, 2)
        H = torch.view_as_complex(H)
        # --initialize
        if Q is None:
            if self.q_mod=='learned':
                Q = torch.view_as_complex(self.Q)[None, :].repeat(H.shape[0], 1, 1)
            elif self.q_mod=='eye':
                Q = torch.eye(H.shape[2], self.Np, dtype=torch.cfloat)[None, :, :].repeat(H.shape[0], 1, 1).cuda()
        else:
            Q = Q.reshape(Q.shape[0], H.shape[2], -1, 2).cuda()
            Q = torch.view_as_complex(Q.contiguous()).cuda()

        # normalize Q
        Q = Q/Q.norm(p='fro',dim=(-2,-1))[:,None,None]

        # --UE's operations
        cqi, m,v = UE_feedback(self.V, Q, H)
        cqi = self.quantizer.apply(cqi,self.qtFlag)
        # -- transmission----

        dec_hcs = self.cell(Q, cqi, m / self.V.shape[1], dec_hcs_pre)

        Q_next = self.Q_fc(dec_hcs['Q_feature'])

        csi_next = self.csi_fc(dec_hcs['H_feature'])

        nBatch = H.shape[0]
        R = self.basis(dec_hcs['H_feature'])
        R = R.reshape(nBatch,32,-1,2)
        R = torch.view_as_complex(R)
        R = R/R.norm(dim=-1,keepdim=True)
        csi_next = torch.view_as_complex(csi_next.reshape(nBatch,-1,2))
        csi_next = (R@csi_next[:,:,None]).squeeze(dim=-1)
        p = Q@v[:,:,None]

        return Q,Q_next, dec_hcs, p, cqi,csi_next,m




class CSIModel(nn.Module):
    def __init__(self,input_size,hidden_size,T,Na,Np,fixM,seed,n_layer=1,model = 'Atten',datapath='',qtFlag='none',Q_mod='eye',nB=16, n_bit=4,n_head=16):
        super(CSIModel, self).__init__()
        np.random.seed(seed)
        self.fixM = fixM
        self.Q = torch.Tensor(np.random.randn(T, Na, Na, 2)).cuda()
        self.T = T
        self.cell = CSICell(hidden_size,Na,Np,n_layer,model,datapath,qtFlag,Q_mod=Q_mod,nB=nB, n_bit=n_bit,n_head=n_head)


    def forward(self,H,R):

        dec_hcs = None
        Q = None
        ps = []
        cqis = []
        csi_pres = []
        Qs = []
        ms = []
        for i in range(self.T):
            Q_pre,Q, dec_hcs, p, cqi,csi_next,m = self.cell(H, Q, dec_hcs,R)

            if self.fixM:
                Q = self.Q[i][None, :, :].repeat(Q.shape[0], 1, 1, 1)
            ps.append(p)
            cqis.append(cqi)
            csi_pres.append(csi_next)
            Qs.append(Q_pre)
            ms.append(m)
        csi_pres = torch.stack(csi_pres)
        Qs = torch.stack(Qs,dim=1)
        ms = torch.stack(ms,dim=1)
        msInd = ms.argmax(dim=-1)
        return csi_pres, ps, cqis, Qs, ms









if __name__ == '__main__':

    data = torch.rand(4,3)

    csiLstm = CSIModel(64, 128, 1, 2, 5, 5)
    out = csiLstm(data)


    print(out.shape)


