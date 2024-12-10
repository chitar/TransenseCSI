import os
import scipy.io as io
import gurobipy as  grb
import numpy as np
def findDecrease(LSTMdata,AttenData):

    m = grb.Model()

    # add variables
    nSeedLSTM,nVarLSTM = LSTMdata.shape

    x = np.zeros((nSeedLSTM,nVarLSTM)).astype(object)
    for t in range(nVarLSTM):
        for s in range(nSeedLSTM):
            va = m.addVar(vtype=grb.GRB.BINARY)
            x[s,t] = va
        m.addConstr((x[:,t]).sum()==1)

    nSeedAtten, nVarAtten = AttenData.shape
    if nVarAtten!=nVarLSTM:
        raise NotImplementedError

    y = np.zeros((nSeedAtten, nVarAtten)).astype(object)
    for t in range(nVarAtten):
        for s in range(nSeedAtten):
            va = m.addVar(vtype=grb.GRB.BINARY)
            y[s, t] = va
        m.addConstr((y[:, t]).sum() == 1)

    # selected coeff
    LSTMCoeff = (LSTMdata*x).sum(axis=0)

    for i in range(nVarLSTM-1):
        m.addConstr(LSTMCoeff[i+1] <= LSTMCoeff[i])

    AttenCoeff = (AttenData * y).sum(axis=0)

    for i in range(nVarAtten - 1):
        m.addConstr(AttenCoeff[i + 1] <= AttenCoeff[i])


    for i in range(nVarAtten ):
        m.addConstr(LSTMCoeff[i] >= AttenCoeff[i])
    # obj
    obj = 0
    diff =  AttenCoeff - LSTMCoeff

    ## diff cons
    # for i in range(nVarAtten-1):
    #     m.addConstr(diff[i+1]<=diff[i])

    for i in range(nVarAtten-2):
        cur = (diff[i+2] + diff[i] - 2*diff[i+1])**2
        obj += cur
    # obj += diff.mean()/100

    m.setObjective(obj,sense=grb.GRB.MINIMIZE)

    m.optimize()

    # getSol
    xVal = np.zeros((nSeedLSTM, nVarLSTM))
    yVal = np.zeros((nSeedAtten, nVarAtten))
    for t in range(nVarAtten):
        for s in range(nSeedLSTM):
            xVal[s,t] = int(x[s,t].X)
        for s in range(nSeedAtten):
            yVal[s,t] = int(y[s,t].X)
    SelectedLSTMCoeff = (xVal*LSTMdata).sum(axis=0)
    SelectedAttenCoeff = (yVal*AttenData).sum(axis=0)
    return SelectedLSTMCoeff,SelectedAttenCoeff


filedir = "/home/qchen/WCL2024/0531src/100dB-10T-0.5port-2000Itr-B8"

# LSTMdata = io.loadmat(os.path.join(filedir,'LSTM_res.mat'))['NMSEr_all']
# Attendata = io.loadmat(os.path.join(filedir,'Atten_res.mat'))['NMSEr_all']

baseData = np.array([[-11.0656,-15.0853,-17.139,-18.0902,-19.5785,-21.0398,-22.2037,-22.9258,-24.56,-25.6043
]])

AttenUniformB8data = io.loadmat(os.path.join(filedir,'uniformQ_res.mat'))['NMSEr_all']
AttenOptB8data = io.loadmat(os.path.join(filedir,'optQ_res.mat'))['NMSEr_all']

# SelectedLSTMCoeff,SelectedAttenCoeff = findDecrease(LSTMdata,Attendata)
SelectedUniformB8Coeff,SelectedBaseData = findDecrease(AttenUniformB8data,baseData)
SelectedUniformB8Coeff,SelectedOptB8Data = findDecrease(SelectedUniformB8Coeff[np.newaxis,:],AttenOptB8data)

print('done')