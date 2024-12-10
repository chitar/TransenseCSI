import os
import multiprocessing
import random
import subprocess

# add env
def shell_source(script):
    """Sometime you want to emulate the action of"source" in bash,
    settings some environment variables. Here is a way to do it."""
    pipe = subprocess.Popen('/bin/bash -c "source %s; env"' % script, stdout=subprocess.PIPE, shell=True)
    output = pipe.communicate()[0]
    for line in output.splitlines():
        line_decode = line.decode()
        # print(line_decode)
        if len(line_decode.split("=")) > 1:
            key, value = line.decode().split("=", 1)
            os.environ.update({key: value})

shell_source('conda_env.sh')

## define variables

epoch = 300
# src = 'train.py'
#src = 'quant_train_uniform.py'
src = 'quant_train.py'
# exp_name = '100dB-10T-0.5port-2000Itr-N8-unlearnedU-B4'
# exp_name = '100dB-10T-0.5port-2000Itr-N8-supp-unlearnedU-B6'
exp_name = '100dB-10T-0.5port-2000Itr-N8-softQuantizer'
trSNR=100
src_exp_dir='100dB-10T-0.5port-2000Itr-N8'
# src_exp_dir=''
loss = "individual"
# NN = 'LSTM'
# Q_mod = 'learned'
## parameters

Ts = list(range(1,11))
random.shuffle((Ts))
Params = [
    ('seed',list(range(1,11))),
    # ('quantize',['none']),
    ('quantize',['softmax']),
    # ('quantize',['uniform','opt']),

    # ('Q_mod',['eye','learned']),
    ('Q_mod',['learned']),
    # ('NN',['LSTM','Atten']),
    ('NN',['Atten']),
    ('n_bit',[4,6]),
    ('T',Ts),

]
## construct cmds

def loopParam(knownParams, allParams, paramInd,cmds):
    if paramInd == len(allParams):
        paraDict = {}
        for name,value in knownParams:
            paraDict[name] = value
        cmd = f"python {src} --NN {paraDict['NN']} --exp_name  {exp_name} --trSNR {trSNR} --epoch {epoch} --T {paraDict['T']} --Q_mod {paraDict['Q_mod']} --seed {paraDict['seed']} --n_bit {paraDict['n_bit']} --quantize {paraDict['quantize']} --loss {loss}"
        if src_exp_dir != '':
            cmd = cmd + f" --src_exp_dir {src_exp_dir}"

        cmds.append(cmd)


    else:
        paraName, targetParams = allParams[paramInd]
        for tp in targetParams:
            loopParam(knownParams+[(paraName, tp)], allParams, paramInd + 1,cmds)

cmds= []
loopParam([],Params,0,cmds)

def run(cmd):

    cmd = cmd + ' > /dev/null'
    os.system(cmd)

    print(f'Finished cmd {cmd}')

if __name__ == '__main__':

    nWorkers = 5
    print(f"Started {nWorkers} workers to process...")
    with multiprocessing.Pool(processes=nWorkers) as pool:
        # add task
        for cmd in cmds:
            pool.apply_async(run, (cmd,))

        # wait
        pool.close()
        pool.join()

    print("All processes are done.")



