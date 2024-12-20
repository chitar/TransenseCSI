import numpy as np
import torch

def NMSE(H,H_hat):
    numerator = (H-H_hat).conj()*(H-H_hat)
    numerator = numerator.sum(dim=(-1)).abs()
    denominator = H.conj()*H
    denominator = denominator.sum(dim=(-1)).abs()

    nmse = (numerator/denominator).mean(dim=(-1))

    return nmse


def NMSEr(H,H_hat):

    theta = H_hat[:,:,None,:].conj()@H[:,:,:,None]
    theta = theta.squeeze(dim=-1).squeeze(dim=-1)
    theta = theta/theta.abs()

    H_hat = H_hat*theta[:,:,None]

    numerator = (H - H_hat).conj() * (H - H_hat)
    numerator = numerator.sum(dim=(-1)).abs()
    denominator = H.conj() * H
    denominator = denominator.sum(dim=(-1)).abs()

    nmse = (numerator / denominator).mean(dim=(-1))

    return nmse



def CORR(H,H_hat):
    '''
    :param H: [Batch,Na]
    :param H_hat: [Batch,Na]
    :return:
    '''
    numerator = (H.conj()*H_hat).sum(dim=-1).abs()
    denominator = H.norm(dim=-1)*H_hat.norm(dim=-1)

    corr = (numerator/denominator).mean(dim=(-1))

    return corr