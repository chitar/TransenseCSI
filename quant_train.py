

import torch
from dataset import CSIDataset
from model import CSIModel
from criterion import NMSEr,CORR
import argparse
import datetime
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--src_exp_dir', type=str, default=r'exps',
                    help='source experimental directory')
parser.add_argument('--NN', type=str, default=r'Atten',
                    help='Atten,LSTM')
parser.add_argument('--exp_name', type=str, default='quant_exps',
                    help='experiment name')
parser.add_argument('--epoch', type=int, default=1000,
                    help='number of epochs')
parser.add_argument('--train_portion', type=float,default=0.5,
                    help='the training data portion over data set')
parser.add_argument('--lr', type=float,default=1e-6,
                    help='learning rate')
parser.add_argument('--T', type=int,default=10,
                    help='times of communication')
parser.add_argument('--loss', type=str, default='individual',
                    help='weighted,individual')
parser.add_argument('--seed', type=int,default=1,
                    help='seed of training sampling')
parser.add_argument('--noise', type=str,default='Gaussian',
                    help='noise type, can be Gaussian or PCA or None')
parser.add_argument('--trSNR', type=int,default=100,
                    help='noise level of estimation error in testing, -1 means none')
parser.add_argument('--fixM',  action='store_true')
parser.add_argument('--quantize',  type=str,default='softmax',help='none,uniform,opt,softmax')
parser.add_argument('--n_bit',  type=int,default=4)
parser.add_argument('--Q_mod',  type=str,default='learned',help='eye,learned')

args = parser.parse_args()

data_filepath = r'./data/data_DCSnn/data_DCSnn.mat' # r'../../QuaDriGa_data/data_DCSnn_QuaDriGa.mat' #

torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True

time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M%S')

save_dir = f'NN-{args.NN}-trSNR-{args.trSNR}-T-{args.T}-quantize-{args.quantize}-n_bit-{args.n_bit}-Q_mod-{args.Q_mod}-seed-{args.seed}-{time_now}'
print(save_dir)

# find load_model path
src_dir = args.src_exp_dir
src_list = os.listdir(src_dir)
queryName = f'NN-{args.NN}-trSNR-{args.trSNR}-T-{args.T}-quantize-none-Q_mod-{args.Q_mod}-seed-{args.seed}'
queryRes = None
for srcName in src_list:
    if queryName in srcName:
        queryRes= srcName
        break
if queryRes is None:
    print('Not found source dir')
    exit(0)
args.load_model = os.path.join(src_dir,queryRes,'model_best.pth')

save_dir = os.path.join(args.exp_name,save_dir)
os.makedirs(save_dir,exist_ok=True)
# write args
with open(os.path.join(save_dir,'config.txt'),'w') as f:
    kargs = args._get_kwargs()
    for name,value in kargs:
        f.write(f'{name}:  {value}\n')


BATCH_SIZE = 64
LR = args.lr
EPOCHS = args.epoch
PRINT_FRE = 30
TEST_FRE = 5
NUM_WORKERS = 0
nB = 8
if __name__ == '__main__':

    train_dataset = CSIDataset(data_filepath, train='train', train_portion=args.train_portion, seed=1,
                               SNR_dB=args.trSNR, noise=args.noise, nB=nB)
    valid_dataset = CSIDataset(data_filepath, train='valid', train_portion=args.train_portion, seed=1, nB=nB)
    test_dataset = CSIDataset(data_filepath, train='test', train_portion=args.train_portion, seed=1, nB=nB)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
                                               num_workers=NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
                                               num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True,
                                              num_workers=NUM_WORKERS)

    model = CSIModel(input_size=64, hidden_size=512, Na=32, T=args.T, Np=32, n_layer=2, fixM=args.fixM, seed=args.seed,
                     model=args.NN, datapath=data_filepath, qtFlag=args.quantize, Q_mod=args.Q_mod, nB=nB, n_bit=args.n_bit).cuda()

    if args.load_model :
        model.load_state_dict(torch.load(args.load_model))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    min_loss = 9999


    for epoch in range(EPOCHS):

        avg_loss = 0
        avg_corr_loss = 0
        avg_nmser_loss = 0

        # train
        for step, data in enumerate(train_loader):
            model.train()
            if args.quantize:
                model.cell.qtFlag='train'+ args.quantize
            H = data['H'].cuda()
            R = train_dataset.R.cuda()
            H_hat, ps, cqis, Qs,ms = model(H.reshape(H.shape[0],-1),R)

            H = torch.view_as_complex(H.contiguous())

            nmser_loss = NMSEr(H=H[None,:,:],H_hat=H_hat)
            corr_loss = 1-CORR(H=H[None,:,:],H_hat=H_hat)

            w = torch.Tensor([1 for i in range(1,args.T+1)])
            w = w/w.sum()
            rloss = (nmser_loss*w.cuda()).sum() if args.loss == 'weighted' else nmser_loss[-1]

            loss = rloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = H.shape[0]

            avg_loss = (avg_loss * (step * BATCH_SIZE) + loss.item() * batch_size) / (step * BATCH_SIZE + batch_size)

            if step % PRINT_FRE == 0:
                print(
                    f'Epoch: [{epoch}][{step}/{len(train_loader)}] Loss: {avg_loss:.6f} ')


        with open(os.path.join(save_dir, 'last_train_loss.txt'), 'a+') as f:
            f.write('train_loss:{}\n'.format(loss))
            f.write('train_nmser:{}({}dB)\n'.format(nmser_loss, 10 * torch.log10(nmser_loss)))
            f.write('train_corr:{}\n'.format(1 - corr_loss))
            f.write('epoch:{}\n'.format(epoch))

        # valid
        if epoch % TEST_FRE == TEST_FRE - 1:

            model.eval()
            if args.quantize:
                model.cell.qtFlag='test'+ args.quantize
            total_H = None
            total_H_hat = None
            total_Hr = None
            total_P = None
            total_CQI = None
            total_Q = None
            with torch.no_grad():
                for step, data in enumerate(valid_loader):
                    H = data['H'].cuda()

                    R = train_dataset.R.cuda()
                    H_hat, ps, cqis, Qs, ms = model(H.reshape(H.shape[0], -1), R)

                    H = torch.view_as_complex(H.contiguous())

                    total_Q = Qs if total_Q is None else torch.cat([total_Q,Qs],dim=0)

                    total_H = H if total_H is None else torch.cat([total_H, H], dim=0)
                    total_H_hat = H_hat if total_H_hat is None else torch.cat([total_H_hat, H_hat], dim=1)

                    P = torch.stack(ps, dim=1).squeeze()
                    CQI = torch.stack(cqis, dim=1)
                    total_P = P if total_P is None else torch.cat([total_P, P], dim=0)
                    total_CQI = CQI if total_CQI is None else torch.cat([total_CQI, CQI], dim=0)


                val_corr_loss = 1 - CORR(H=total_H, H_hat=total_H_hat)
                val_nmser_loss = NMSEr(H=total_H[None,:,:], H_hat=total_H_hat)

                w = torch.Tensor([1 for i in range(1, args.T + 1)])
                w = w / w.sum()
                val_loss = (val_nmser_loss*w.cuda()).sum() if args.loss == 'weighted' else val_nmser_loss[-1]
                loss_print = val_nmser_loss[-1] if args.loss == 'individual' else val_nmser_loss
                print('Val: meanLoss {:.6f}  nmser_loss {} '.format(
                    val_loss, 10 * torch.log10(loss_print),))

                if val_loss <= min_loss :
                    min_loss = val_loss
                    # save model
                    torch.save(model.state_dict(), os.path.join(save_dir, 'model_best.pth'))
                    # save data

                    with open(os.path.join(save_dir, 'best_val_loss.txt'), 'w') as f:
                        f.write('best_val_loss:{}\n'.format(val_loss))
                        f.write('best_val_nmser:{}({}dB)\n'.format(val_nmser_loss,10*torch.log10(val_nmser_loss)))
                        f.write('best_val_corr:{}\n'.format(1-val_corr_loss))
                        f.write('best_epoch:{}\n'.format(epoch))

                with open(os.path.join(save_dir, 'last_val_loss.txt'), 'a+') as f:
                    f.write('val_loss:{}\n'.format(val_loss))
                    f.write('val_nmser:{}({}dB)\n'.format(val_nmser_loss,10*torch.log10(val_nmser_loss)))
                    f.write('val_corr:{}\n'.format(1 - val_corr_loss))
                    f.write('epoch:{}\n'.format(epoch))



    # test

    model.load_state_dict(torch.load(os.path.join(save_dir, 'model_best.pth')))
    model.eval()
    if args.quantize:
        model.cell.qtFlag = 'test' + args.quantize
    total_H = None
    total_H_hat = None
    total_Hr = None
    total_P = None
    total_CQI = None
    total_Q = None
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            H = data['H'].cuda()

            R = train_dataset.R.cuda()
            H_hat, ps, cqis, Qs, ms = model(H.reshape(H.shape[0], -1), R)

            H = torch.view_as_complex(H.contiguous())

            total_Q = Qs if total_Q is None else torch.cat([total_Q, Qs], dim=0)

            total_H = H if total_H is None else torch.cat([total_H, H], dim=0)
            total_H_hat = H_hat if total_H_hat is None else torch.cat([total_H_hat, H_hat], dim=1)

            P = torch.stack(ps, dim=1).squeeze()
            CQI = torch.stack(cqis, dim=1)
            total_P = P if total_P is None else torch.cat([total_P, P], dim=0)
            total_CQI = CQI if total_CQI is None else torch.cat([total_CQI, CQI], dim=0)

        val_corr_loss = 1 - CORR(H=total_H, H_hat=total_H_hat)
        val_nmser_loss = NMSEr(H=total_H[None, :, :], H_hat=total_H_hat)

        w = torch.Tensor([1 for i in range(1, args.T + 1)])
        w = w / w.sum()
        val_loss = (val_nmser_loss * w.cuda()).sum() if args.loss == 'weighted' else val_nmser_loss[-1]
        loss_print = val_nmser_loss[-1] if args.loss == 'individual' else val_nmser_loss
        print('Test: meanLoss {:.6f}  nmser_loss {} '.format(
            val_loss, 10 * torch.log10(loss_print), ))


        # save data

        with open(os.path.join(save_dir, 'best_test_loss.txt'), 'w') as f:
            f.write('best_test_loss:{}\n'.format(val_loss))
            f.write('best_val_nmser:{}({}dB)\n'.format(val_nmser_loss,
                                                       10 * torch.log10(val_nmser_loss)))
            f.write('best_val_corr:{}\n'.format(1 - val_corr_loss))
