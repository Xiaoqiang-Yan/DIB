import torch
from scipy.spatial.distance import squareform, pdist
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
import datetime
from lightly.loss import NTXentLoss

from utils import calculate_MI, UD_constraint

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# MNIST-USPS
# BDGP
# hand
# ESP
# Flickr
# Caltech-3V
# Caltech-4V
# Caltech-5V
Dataname = 'MNIST-USPS'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--pretrain_epoc", default=200)    
parser.add_argument("--train_epoc", default=100)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--compress_rate", default=0.01)
parser.add_argument("--clustering_rate", default=0.01)
parser.add_argument("--alpha",default=0.01)
parser.add_argument("--beta",default=0.01)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
if args.dataset == "MNIST-USPS":
    args.train_epoc = 100
    seed = 10
if args.dataset == "BDGP":
    args.train_epoc = 20
    seed = 10
if args.dataset == "hand":
    args.train_epoc = 90
    seed = 10
if args.dataset == "ESP":
    args.train_epoc = 80
    seed = 10
if args.dataset == "Flickr":
    args.train_epoc = 40
    seed = 10
if args.dataset == "Caltech-3V":
    args.train_epoc = 100
    seed = 10
if args.dataset == "Caltech-4V":
    args.train_epoc = 100
    seed = 10
if args.dataset == "Caltech-5V":
    args.train_epoc = 100
    seed = 10



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def train(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    cross_entropy = torch.nn.CrossEntropyLoss()
    NTX_loss = NTXentLoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs = model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(NTX_loss(hs[v], hs[w]))
                loss_list.append(NTX_loss(qs[v], qs[w]))
            loss_list.append(mes(xs[v], xrs[v]))

            UDC = UD_constraint(qs[v]).to(device)
            loss_list.append(cross_entropy(qs[v], UDC))

            with torch.no_grad():
                feature_z = zs[v].cpu().detach().numpy()
                k_z = squareform(pdist(feature_z, 'euclidean'))
                sigma_z = np.mean(np.sort(k_z[:, :10], 1))

                input = xs[v].cpu().detach().numpy()
                input = input.reshape(xs[v].shape[0],-1)
                k_x = squareform(pdist(input, 'euclidean'))
                sigma_x = np.mean(np.sort(k_x[:, :10], 1))

                feature_h = hs[v].cpu().detach().numpy()
                k_h = squareform(pdist(feature_h, 'euclidean'))
                sigma_h = np.mean(np.sort(k_h[:, :10], 1))

                label = qs[v].cpu().detach().numpy()
                k_q = squareform(pdist(label, 'euclidean'))
                sigma_q = np.mean(np.sort(k_q[:, :10], 1))

            IXZ = calculate_MI(xs[v], zs[v], sigma_x, sigma_z, batch_size=args.batch_size)
            IHS = - calculate_MI(hs[v], qs[v], sigma_h, sigma_q, batch_size=args.batch_size)
            loss_list.append(args.alpha * IXZ + args.beta * IHS)



        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))




accs = []
nmis  = []
purs  = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
for i in range(T):
    print("ROUND:{}".format(i+1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    epoch = 1
    while epoch <= args.pretrain_epoc:
        pretrain(epoch)
        epoch += 1
    while epoch <= args.pretrain_epoc + args.train_epoc:
        train(epoch)
        if epoch == args.pretrain_epoc + args.train_epoc:
            acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num)
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
            print('Saving..')
            accs.append(acc)
            nmis.append(nmi)
            purs.append(pur)
        epoch += 1
