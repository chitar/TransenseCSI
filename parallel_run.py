import os
import multiprocessing
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

epoch = 500
exp_name = '30dB3000SeedNormQ'
trSNR=30
NN = 'Atten'
Q_mod = 'learned'
## parameters

Params = [
    ('seed',[1,2,3,4,5]),
    ('quantize',['uniform','opt']),#('quantize',['none']),#
    ('T',list(range(1,11))),

]
## construct cmds

def loopParam(knownParams, allParams, paramInd,cmds):
    if paramInd == len(allParams):
        paraDict = {}
        for name,value in knownParams:
            paraDict[name] = value

        cmds.append(f"python train.py --NN {NN} --exp_name  {exp_name} --trSNR {trSNR} --epoch {epoch} --T {paraDict['T']} --Q_mod {Q_mod} --seed {paraDict['seed']} --quantize {paraDict['quantize']}")


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

    nWorkers = 4
    print(f"Started {nWorkers} workers to process...")
    with multiprocessing.Pool(processes=nWorkers) as pool:
        # add task
        for cmd in cmds:
            pool.apply_async(run, (cmd,))

        # wait
        pool.close()
        pool.join()

    print("All processes are done.")



