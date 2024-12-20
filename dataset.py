import numpy as np
import torch
import random
from torch.utils.data import Dataset
import scipy.io as io

def SNR_dB2linear(dB):
    return 10**(dB/10)


class CSIDataset(Dataset):

    def __init__(self, filepath, train, train_portion,seed=2,SNR_dB=None,noise='None',nB=16):
        self.filepath = filepath
        data = io.loadmat(filepath)
        self.csi = torch.from_numpy(np.array(data['M_DCSnn_csi'])).type(torch.cfloat)
        self.train_portion = train_portion
        self.train = train
        self.SNR = SNR_dB2linear(SNR_dB) if SNR_dB is not None else None
        self.noise = noise
        self.U = None
        self.S = None

        n_samples = self.csi.shape[0]
        indices = [ i for i in range(n_samples) ]
        sep = int(train_portion*n_samples)
        random.seed(seed)
        random.shuffle(indices)
        self.sample_inds = indices[0:int(sep*0.6)] if train=='train' else indices[int(sep*0.6):sep] if train=='valid' else indices[sep:]

        # train samples
        trainCSIs = self.csi[self.sample_inds]
        svds = trainCSIs.svd()

        svalues = svds[1]
        eigenVecs = svds[-1]
        R_H = eigenVecs[0:nB,:]
        R = R_H.t().conj()

        self.R = R
        self.Proj = R@(R_H@R).inverse()@R_H


        print('loaded {} samples for {}'.format(self.__len__(), 'train' if train=='train' else 'valid' if train=='valid' else 'test'))




    def __getitem__(self, index):
        ind = self.sample_inds[index]  # %len(self.sample_inds)]
        csis = self.csi[self.sample_inds]
        h = self.csi[ind]
        noise = 0
        # get nearest k samples
        if self.noise == 'Gaussian':

            delta = h.norm() / np.sqrt(2 * h.shape[-1]) / np.sqrt(self.SNR)
            noise = torch.view_as_complex(delta * torch.randn(h.shape[-1], 2))

        elif self.noise == 'PCA':

            k = 4

            inds = (csis - h[None, :]).norm(dim=-1).sort()[1][0:k]
            hs = self.csi[inds]
            hs_mean = hs.mean(dim=0, keepdim=True)
            U, S, V = torch.linalg.svd((hs - hs_mean).permute(1, 0))
            T = torch.zeros(U.shape[0]) + 0j
            T[0:k] = S

            delta = h.norm() / (S ** 2).sum().sqrt() / np.sqrt(self.SNR)
            rand = delta * torch.randn(T.shape[0])
            diagT = torch.diag(T * rand) + 0j
            noise = (U @ diagT).sum(dim=-1)


        h = h + noise
        h = torch.view_as_real(h)
        return {
            'H': h
        }

    def __len__(self):
        return len(self.sample_inds)


if __name__ == '__main__':
    filepath =  r'./csidata.mat'
    dataset = CSIDataset(filepath, train=True,train_portion=0.2)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, pin_memory=True)

    for data in data_loader:
        H = data['H']
    # validate2
    print(H.shape)
    print('done')