import numpy as np
import os
import re
import torch
import scipy.io as io

def readConfig(filepath):
    lines = []
    configs = {}
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines:
        ss = line.strip().split(':')
        configs[ss[0].strip()] = ss[1].strip()
    return configs

def readRes(filepath):
    lines = []
    res = {}
    with open(filepath) as f:
        lines = f.readlines()
    newLines = []
    for line in lines:
        if line[0:2] == '  ':
            newLines[-1] = newLines[-1] + line.strip()
        else:
            line = line.replace(':ten','#ten')
            line = line.replace('loss:','loss#')
            newLines.append(line.strip())
    for line in newLines:
        ss = line.strip().split('#')
        if 'tensor' not in line:
            res[ss[0].strip()] = ss[1].strip()
        else:
            ts = ss[1].strip()
            stInd = ts.index(')')+1 if 'nmser' in ss[0] else 0
            data = re.findall("\[.*\]",ts[stInd:])[0]
            res[ss[0].strip()] = eval(data)



    return res


def filterRes(cond,allres,allconfigs):

    Ts = sorted(list(set([int(res['T']) for res in allconfigs])))
    seeds = sorted(list(set([int(res['seed']) for res in allconfigs])))

    nmserRes = np.zeros((len(seeds),len(Ts))) + 1000
    corrRes = np.zeros((len(seeds), len(Ts))) + 1000


    for i in range(len(allres)):
        configs = allconfigs[i]
        # check config
        flag = False
        for j in range(len(cond)):
            key = list(cond.keys())[j]
            if cond[key] != configs[key]:
                flag=True
                break
        if flag:
            continue
        seed = int(allconfigs[i]['seed'])
        T = int(allconfigs[i]['T'])
        nmser = allres[i]['best_val_nmser'][-1]
        corr = allres[i]['best_val_corr'][-1]

        nmserRes[seeds.index(seed),Ts.index(T)] = nmser
        corrRes[seeds.index(seed), Ts.index(T)] = corr

    return {
        'NMSEr':nmserRes,
        'corr':corrRes,
        'T':Ts,
        'seed':seeds
    }

def getAllConfigsAndRes(expDir):
    expList = os.listdir(expDir)
    expList = [os.path.join(expDir, expName) for expName in expList if os.path.isdir(os.path.join(expDir, expName))]

    allRes = []
    allConfigs = []
    for exp in expList:
        configFile = os.path.join(exp, 'config.txt')
        configs = readConfig(configFile)

        seed = re.findall('seed-\d+',exp)[0].replace('-','').replace('seed','')
        configs['seed'] = seed

        allConfigs.append(configs)
        resFile = os.path.join(exp, 'best_test_loss.txt')
        res = readRes(resFile)
        allRes.append(res)

    return allConfigs,allRes


def extractFilteredRes(filter,expDir,savePath,allConfigs,allRes):

    filteredRes = filterRes(filter, allRes, allConfigs)
    resData = {
        'NMSEr_all': filteredRes['NMSEr'],
        'NMSEr_mean': filteredRes['NMSEr'].mean(axis=0),
        'Corr_all': filteredRes['corr'],
        'Corr_mean': filteredRes['corr'].mean(axis=0),
        'Ts': np.array(filteredRes['T']),
        'seeds': np.array(filteredRes['seed'])
    }
    io.savemat(os.path.join(expDir,savePath), resData)



if __name__ == '__main__':

    # expDir = r'100dB-10T-0.5port-2000Itr-N8-supp'
    # allConfigs, allRes = getAllConfigsAndRes(expDir)
    #
    # # filter res
    # transenseCSICond = {'NN':'Atten','Q_mod':'learned',"quantize":'none'}
    # LSTMModelCond = {'NN':'LSTM','Q_mod':'learned',"quantize":'none'}
    #
    # extractFilteredRes(transenseCSICond,expDir,'Atten_res.mat',allConfigs,allRes)
    # extractFilteredRes(LSTMModelCond, expDir, 'LSTM_res.mat', allConfigs, allRes)

    expDir = r'hyperTuning'
    allConfigs, allRes = getAllConfigsAndRes(expDir)
    softmaxB4 = {'NN': 'Atten', 'Q_mod': 'learned', "quantize": 'softmax','n_bit':'4'}
    softmaxB6 = {'NN': 'Atten', 'Q_mod': 'learned', "quantize": 'softmax','n_bit':'6'}
    extractFilteredRes(softmaxB4, expDir, 'softmaxB4.mat', allConfigs, allRes)
    extractFilteredRes(softmaxB6, expDir, 'softmaxB6.mat', allConfigs, allRes)

    print('done')