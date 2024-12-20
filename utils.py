
import torch

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,inp,out):
        return out

    @staticmethod
    def backward(ctx,out_grad):
        return out_grad

def argmaxOneHotSTE(inp,temp=0.1):
    '''
    :param inp: batch x n
    :return:
    '''
    if len(inp.shape)!=2:
        raise NotImplementedError
    softInp = (inp/temp).softmax(dim=-1)
    inds = softInp.max(dim=-1)[1]
    one_hots = torch.nn.functional.one_hot(inds,num_classes=inp.shape[-1])

    output = STE.apply(softInp,one_hots)

    return output


def UE_feedback(V,Q,H):

    He = H@Q
    prod = He @ V
    prod = prod.squeeze().abs()
    onehot = argmaxOneHotSTE(prod).float()
    m = onehot
    v = (V[None,:,:].repeat(onehot.shape[0],1,1)@(onehot[:,:,None]+0j)).squeeze(dim=-1)
    cqi = He @ v[:,:,None]
    cqi = cqi.abs().squeeze()

    return cqi,m,v
