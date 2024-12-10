import torch
import torch.nn as nn

class LSTMEmb(nn.Module):
    def __init__(self,hid,Q_inp):
        super(LSTMEmb, self).__init__()
        self.emb_cqi = nn.Linear(1,hid)
        self.emb_m = nn.Linear(1,hid)
        self.emb_Q = nn.Linear(Q_inp, hid)

    def forward(self,Q,cqi,m):

        return self.emb_m(m) + self.emb_cqi(cqi) + self.emb_Q(Q)


class LSTMCell(nn.Module):
    def __init__(self,input_size, hidden_size,num_layers=1):
        super(LSTMCell, self).__init__()
        self.num_layers = num_layers
        self.emb_layer = LSTMEmb(hidden_size,32*32*2)
        layers = []

        for i in range(num_layers):
            # inp = input_size if i==0 else hidden_size
            layers.append(nn.LSTMCell(hidden_size,hidden_size))

        self.layers = nn.ModuleList(layers)


    def forward(self,Q,cqi,m,his_data):

        his_data = [None for i in range(self.num_layers)] if his_data is None else his_data['data']
        cur_data = []
        Q = torch.view_as_real(Q).contiguous().reshape(Q.shape[0], -1)
        h = self.emb_layer(Q,cqi[:,None],m[:,None].float())
        for i in range(self.num_layers):
            # inp = x if i==0 else h
            h,c = self.layers[i](h,his_data[i])
            cur_data.append((h,c))

        cur_data = {'data':cur_data,'Q_feature':h,'H_feature':h}
        return cur_data


def LSTM_Q_FC(hidden_size,nB):
    return nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(1024),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(1024),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(1024),
            nn.LayerNorm(1024),
            nn.Linear(1024, nB*32*2),
        )


def LSTM_CSI_FC(hidden_size,nB):
    return nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(1024),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(1024),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(1024),
            nn.LayerNorm(1024),
            nn.Linear(1024, nB*2),
        )

