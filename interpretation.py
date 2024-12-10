import torch
import scipy.io as io

def chord_dis(A,B):
    '''
    A:... x m x n
    B:... x m x n
    '''
    A_H = A.transpose(-1,-2).conj()
    B_H = B.transpose(-1,-2).conj()
    AA = A@A_H
    BB = B@B_H

    diff = AA - BB
    dis = diff.norm(p='fro',dim=(-2,-1))
    return dis


datafile = r'/home/qchen/WCL2024/0531src/SeqTrain/NN-Atten-trSNR-30-T-10-quantize-none-Q_mod-learned-seed-1/out_data.pth'


data = torch.load(datafile)
Qs = data['Qs']
H = data['H']
torch.svd(Qs)
Pv = data['P']
svds = torch.svd(Qs)
svd_values = svds.S
svd_vector = svds.U
svdRate = svd_values/svd_values.sum(dim=-1,keepdim=True)
svdRate = svdRate.cpu().numpy()
T = Pv.shape[1]
disHP = chord_dis(H[:,None,:,None].repeat(1,T,1,1),Pv[:,:,:,None])

disPPs = []
for i in range(T-1):
    disPP = chord_dis(Pv[:,i,:,None],Pv[:,i+1,:,None])
    disPPs.append(disPP)
disPPs = torch.stack(disPPs)

angles = []
for i in range(T-1):


    angle = 1


print('done')