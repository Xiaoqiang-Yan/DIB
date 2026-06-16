import torch
import numpy as np

def pairwise_distances(x, batch_size):
    '''
    calculate the pairwise distances
    :param x:
    :param batch_size:
    :return: pairwise distances
    '''
    x = x.view(batch_size,-1)
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def calculate_gram_mat(x, sigma, batch_size):
    '''
    calculate the gram matrix
    :param x:
    :param sigma:
    :param batch_size:
    :return: gram matrix
    '''
    dist= pairwise_distances(x, batch_size)
    return torch.exp(-dist /sigma)

def reyi_entropy(x, sigma, batch_size):
    '''
    calculate the renyi entropy
    :param x:
    :param sigma:
    :param batch_size:
    :return: renyi entropy
    '''
    alpha = 1.01
    k = calculate_gram_mat(x, sigma, batch_size)
    k = k/torch.trace(k) 
    eigv = torch.abs(torch.linalg.eigh(k, UPLO='L')[0])
    eig_pow = eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, s_x, s_y, batch_size):
    '''
    calculate the joint renyi entropy
    :param x:
    :param y:
    :param s_x:
    :param s_y:
    :param batch_size:
    :return: joint renyi entropy
    '''
    alpha = 1.01
    x = calculate_gram_mat(x, s_x, batch_size)
    y = calculate_gram_mat(y, s_y, batch_size)
    k = torch.mul(x,y)
    k = k/torch.trace(k)
    eigv = torch.abs(torch.linalg.eigh(k, UPLO='L')[0])
    eig_pow =  eigv**alpha
    entropy = (1/(1-alpha))*torch.log2(torch.sum(eig_pow))

    return entropy

def calculate_MI(x,y,s_x,s_y, batch_size=128):
    '''
    calculate the mutual information
    :param x:
    :param y:
    :param s_x:
    :param s_y:
    :param batch_size:
    :return: mutual information
    '''
    Hx = reyi_entropy(x, sigma=s_x, batch_size=batch_size)
    Hy = reyi_entropy(y, sigma=s_y, batch_size=batch_size)
    Hxy = joint_entropy(x,y,s_x,s_y, batch_size=batch_size)
    Ixy = Hx+Hy-Hxy
    
    return Ixy

def UD_constraint(classer):
    CL = classer.detach().cpu().numpy()
    N, K = CL.shape
    CL = CL.T
    r = np.ones((K, 1)) / K
    c = np.ones((N, 1)) / N
    CL **= 10
    inv_K = 1. / K
    inv_N = 1. / N
    err = 1e3
    _counter = 0
    while err > 1e-2 and _counter < 75:
        r = inv_K / (CL @ c)
        c_new = inv_N / (r.T @ CL).T
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    CL *= np.squeeze(c)
    CL = CL.T
    CL *= np.squeeze(r)
    CL = CL.T
    try:
        argmaxes = np.nanargmax(CL, 0)
    except:
        argmaxes = np.argmax(CL, 0)
    newL = torch.LongTensor(argmaxes)
    return newL


