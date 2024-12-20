import numpy as np
import torch

def getUnifY(minimum,maximum,n_bit):

    interval = (maximum - minimum)/(2**n_bit-1)
    y = [ minimum + i*interval for i in range(2**n_bit)]

    y = torch.Tensor(y)

    return y



def getAllCodes(n_bit):
    codes = []
    for i in range(2**n_bit):
        st=bin(i).replace('0b','')
        st = '0' * (n_bit - len(st)) + st
        res = [int(x) for x in st]
        codes.append(res)
    codes = torch.Tensor(codes)
    return codes


def getB(v,x):
    '''
    fix v then solve B
    :param v: [n_bit]
    :return:
    '''
    n_bit = v.shape[0]
    C = getAllCodes(n_bit).cuda()
    Cv = (C @ v[:, None]).squeeze(-1)

    diff = (Cv[None, :] - x[:, None]).abs()

    inds = diff.argmin(dim=-1)

    levels = torch.Tensor([2 ** (n_bit - i - 1) for i in range(n_bit)])

    newB = torch.ones(inds.shape[0], n_bit).cuda()
    baseInds = inds.detach()
    for i in range(n_bit):
        pos = baseInds >= levels[i]
        newB[:, i] = pos.int()
        baseInds = baseInds - levels[i] * newB[:, i]
    return newB

def getv(B,x):
    '''
    fix B then solve v
    :param v: [n_bit]
    :param B: [Batch,n_bit]
    :return:
    '''
    v = (B.t()@B + 0.001*torch.eye(B.shape[-1]).cuda()).inverse()@B.t()@x[:,None]

    return v


def quantizer(v,x,b,T):
    '''
    :param v: [n_bit]
    :param x: [Batch]
    :param b: [] # bias
    :param T:
    :return:
    '''
    x = x - b
    B = None
    for _ in range(T):
        # fix v find B
        B = getB(v,x)
        # fix B find v
        v = getv(B,x).squeeze(-1)
    return v,B


def getUnifV(x,n_bit):
    y = getUnifY(x.min(), x.max(), n_bit)[:, None]
    C = getAllCodes(n_bit)
    v = (C.t() @ C).inverse() @ C.t() @ (y - y.mean())
    return v.squeeze(-1)
def getUnifV2(xmax,xmin,n_bit):
    y = getUnifY(xmin, xmax, n_bit)[:, None]
    C = getAllCodes(n_bit)
    v = (C.t() @ C).inverse() @ C.t() @ (y - y.mean())
    return v.squeeze(-1)



class Quantizer(torch.autograd.Function):
    b = None
    v = None
    def __init__(self,n_bit):
        super(Quantizer, self).__init__()
        self.n_bit = n_bit
    @staticmethod
    def forward(ctx, x, qtFlag):
        alpha = 0.1
        if 'none' in qtFlag:
            q = x
        elif 'train' in qtFlag:
            b = x.mean() if Quantizer.b is None else (1-alpha)*Quantizer.b + alpha*x.mean()
            if 'uniform' in qtFlag:
                v = getUnifV(x,ctx.n_bit).cuda() if Quantizer.v is None else (1-alpha)*Quantizer.v + alpha*getUnifV(x,ctx.n_bit).cuda()
            elif 'opt' in qtFlag:
                v = getUnifV(x, ctx.n_bit).cuda()
                v,B = quantizer(v,x,b,5)
                v = v if Quantizer.v is None else (1-alpha)*Quantizer.v + alpha*v
            B = getB(v, x - b)
            q = B@v[:,None] + b

            Quantizer.v = v
            Quantizer.b = b

        elif 'test' in qtFlag:
            v = Quantizer.v
            b = Quantizer.b

            B = getB(v,x - b)
            q = B @ v[:, None] + b

        return q.squeeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.

        return grad_output, None


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_bar):
        return x_bar

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class LinearQuantizer(torch.autograd.Function):
    def __init__(self, n_bit=6):
        super(LinearQuantizer, self).__init__()
        self.n_bit = n_bit
        self.ste = STE()

    def apply(self, x, qtFlag):

        with torch.no_grad():
            alpha = 0.1
            if 'none' in qtFlag:
                q = x
            elif 'train' in qtFlag:
                b = x.mean() if Quantizer.b is None else (1 - alpha) * Quantizer.b + alpha * x.mean()
                if 'uniform' in qtFlag:
                    v = getUnifV(x, self.n_bit).cuda() if Quantizer.v is None else (
                                                                                              1 - alpha) * Quantizer.v + alpha * getUnifV(
                        x, self.n_bit).cuda()
                elif 'opt' in qtFlag:
                    v = getUnifV(x, self.n_bit).cuda()
                    v, B = quantizer(v, x, b, 5)
                    v = v if Quantizer.v is None else (1 - alpha) * Quantizer.v + alpha * v
                B = getB(v, x - b)
                q = B @ v[:, None] + b


                Quantizer.v = v
                Quantizer.b = b

            elif 'test' in qtFlag:
                v = Quantizer.v
                b = Quantizer.b

                B = getB(v, x - b)
                q = B @ v[:, None] + b

        x = self.ste.apply(x,q.squeeze(-1))

        return x


class SoftmaxQuantizer(torch.autograd.Function):
    def __init__(self, n_bit=6,a=8,v_sup=2):
        super(SoftmaxQuantizer, self).__init__()
        self.n_bit = n_bit
        self.v_sup = v_sup
        self.a = a
        self.ste = STE()
    def apply(self, x, qtFlag):

        if self.v_sup<x.max():
            raise NotImplementedError
        x_norm = x/self.v_sup
        x_soft = 0
        for i in range(1,2**self.n_bit-1+1):
            x_soft += 0.5*(torch.tanh(self.a*(2**self.n_bit * x_norm - i))+1)

        x_hard = torch.round(2**self.n_bit * x_norm - 0.5)

        x_q = self.ste.apply(x_soft, x_hard)
        x_dq = (x_q + 0.5)/2**self.n_bit
        x_bar = x_dq * self.v_sup

        return x_bar


if __name__ == '__main__':
    BATCH = 64
    n_bit = 5

    x = torch.rand(BATCH) +1
    y = getUnifY(x.min(),x.max(),n_bit)[:,None]
    C = getAllCodes(4)
    v = (C.t()@C).inverse()@C.t()@(y-y.mean())
    b = x.mean()

    v,B = quantizer(v,x,b,10)
    print('done')
