import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys
import math
assert sys.version_info >= (3, 5)
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import p_sample_loop_w_Condition, p_sample_loop_w_Condition_DDIM
from statsmodels.distributions.empirical_distribution import ECDF

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = "serif"

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
np.random.seed(42)
from channel_models import *


def Constellation_InOut_m(enc, ch_model, ch_model_type, device, alphas, betas, alphas_bar_sqrt,
                          one_minus_alphas_bar_sqrt,
                          num_steps, M, num_samples, noise_std, max_amp, m=0, PreT = False,
                          denoising_alg = 'DDPM', traj = range(1), IS_RES = False, pred_type='epsilon'):
    labels = m * torch.ones(1)
    msg_1h = F.one_hot(labels.long(), num_classes=M)                    
    msg_1h = msg_1h.float().reshape(-1,M) 
    x = enc(msg_1h).repeat(num_samples,1)
    
    n = x.shape[1] # channel dimension

    if ch_model_type == 'AWGN':
        channel_model = ch_AWGN
    elif ch_model_type == 'Rayleigh':
        channel_model = ch_Rayleigh_AWGN_n
    elif ch_model_type == 'SSPA':
        channel_model = ch_SSPA

    else:
        raise Exception('Unrecognized channel model. Available models: AWGN, Rayleigh, SSPA.')

    y = channel_model(x,noise_std,device)

    if PreT:
        c = x
    else:
        c = torch.cat((labels.to(device), x ),dim=1)
    if denoising_alg == 'DDPM':
        x_seq = p_sample_loop_w_Condition(ch_model, x.size(), num_steps, alphas, betas, alphas_bar_sqrt,
                                          one_minus_alphas_bar_sqrt, c, pred_type = pred_type)
    elif denoising_alg == 'DDIM':
        alphas_prod = torch.cumprod(alphas, 0)
        x_seq = p_sample_loop_w_Condition_DDIM(ch_model, x.size(), traj, alphas_prod, alphas_bar_sqrt,
                                               one_minus_alphas_bar_sqrt, c, pred_type = pred_type)

    y_gen = x_seq[-1]
    if IS_RES: 
        y_gen = y_gen + c

    y_gen = y_gen

    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y_gen = y_gen.detach().cpu().numpy()

    print("The scatter plot of the generated data in comparison to the original source; message", m+1, flush=True)

    if n//2 ==1 :
        fig = plt.figure(figsize=(4,4))
        plt.scatter(y_gen[:,0], y_gen[:,1], edgecolor=(1,0,0,0.8), label='generated', facecolor=(1,1,1,0.5))
        plt.scatter(y[:,0], y[:,1], edgecolor=(0,0.3,0.3,0.7), label='ground truth', facecolor=(1,1,1,0.5))
        plt.scatter(x[:,0], x[:,1], edgecolor=(0,0,0,0.8), label='channel input',facecolor=(1,1,1,0.5) )
        plt.xlim([-max_amp, max_amp])
        plt.ylim([-max_amp, max_amp])
        plt.xlabel('$Re(y)$')
        plt.ylabel('$Im(y)$')
        title = 'message '+ str(m+1)
        plt.title(title)
        plt.legend()
    else:
        fig, ax = plt.subplots(1,n//2,figsize=(10,4))
        for j in range(n // 2):
            ax[j].scatter(y_gen[:,2*j], y_gen[:,2*j+1], edgecolor=(1,0,0,0.8), label='generated', facecolor=(1,1,1,0.5))
            ax[j].scatter(y[:,2*j], y[:,2*j+1], edgecolor=(0,0.3,0.3,0.7), label='ground truth', facecolor=(1,1,1,0.5))
            ax[j].scatter(x[:,2*j], x[:,2*j+1], edgecolor=(0,0,0,0.8), label='channel input',facecolor=(1,1,1,0.5) )
            ax[j].set_xlim([-max_amp, max_amp])
            ax[j].set_ylim([-max_amp, max_amp])
            ax[j].set_xlabel('$Re(y_'+ str(j+1)+ ')$')
            ax[j].set_ylabel('$Im(y_'+ str(j+1)+')$')
            title = '$m = '+ str(m+1)+'$'
            ax[j].set_title(title)
            ax[j].legend()       
    plt.show()
    return     


def Constellation_InOut(enc, ch_model, ch_model_type, device, alphas, betas, alphas_bar_sqrt,
                        one_minus_alphas_bar_sqrt,
                        num_steps, M, num_samples, noise_std, max_amp, PreT = False,
                        denoising_alg = 'DDPM', traj = range(1), cmap_str = 'nipy_spectral',
                        IS_RES = False, pred_type = 'epsilon'):

    labels = torch.empty((num_samples*M,1), dtype = int)
    for i in range(M):
        labels[num_samples*i:num_samples*(i+1),:] = torch.full([num_samples,1],i,dtype=int)

    m = F.one_hot(labels, num_classes=M)                    
    m = m.float().reshape(-1,M)
    x = enc(m) #to(device)
    n = x.shape[1] # channel dimension
    colors = 100//M*np.arange(M)/100. #np.arange(1, 100, 100//M, dtype=int)[:M]
    cmap = plt.cm.get_cmap(cmap_str)
    colors_rgba = cmap(colors)


    if ch_model_type == 'AWGN':
        channel_model = ch_AWGN
    elif ch_model_type == 'Rayleigh':
        channel_model = ch_Rayleigh_AWGN_n
    elif ch_model_type == 'SSPA':
        channel_model = ch_SSPA
    else:
        raise Exception('Unrecognized channel model. Available models: AWGN, Rayleigh, SSPA.')

    y = channel_model(x,noise_std,device)

    if PreT:
        c = x
    else:
        c = torch.cat((labels.to(device), x ),dim=1) #repeat(num_samples*M,1)
        #mx = torch.cat((m, x),dim=1)
    if denoising_alg == 'DDPM':
        x_seq = p_sample_loop_w_Condition(ch_model, x.size(), num_steps, alphas, betas, alphas_bar_sqrt,
                                          one_minus_alphas_bar_sqrt, c, pred_type = pred_type)
    elif denoising_alg == 'DDIM':
        alphas_prod = torch.cumprod(alphas, 0)
        x_seq = p_sample_loop_w_Condition_DDIM(ch_model, x.size(), traj, alphas_prod, alphas_bar_sqrt,
                                               one_minus_alphas_bar_sqrt, c, pred_type = pred_type)
    else:
        assert(ValueError)

    if IS_RES:
        y_gen = x_seq[-1] + c
    else: y_gen = x_seq[-1]

    y_gen = y_gen

    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y_gen = y_gen.detach().cpu().numpy()
    if n // 2 == 1:
        fig = plt.figure(figsize=(7, 7))
    else:
        fig, ax = plt.subplots( 1, n // 2, figsize=( 3 *  n // 2, 3))
    
    if n//2 ==1 :        
        plt.scatter(y_gen[:,0], y_gen[:,1], edgecolor = np.repeat(colors_rgba, num_samples , axis=0),
                    marker="^", s = 25, facecolor='none', label = 'generated')
        plt.scatter(y[:,0], y[:,1], color = np.repeat(colors_rgba, num_samples , axis=0), marker="x",
                    s = 25,  label='ground truth') #c=np.repeat(colors, num_samples), facecolor='none')
        plt.scatter(x[:,0], x[:,1], color = np.repeat(colors_rgba, num_samples , axis=0),
                    edgecolor = 'black', label= 'channel input', s = 25 )
        plt.legend()

    else:
        for j in range(n // 2):
            ax[j].scatter(y_gen[:,2*j], y_gen[:,2*j+1], edgecolor = np.repeat(colors_rgba, num_samples , axis=0),
                          marker="^", s = 25, facecolor='none', label = 'generated')
            ax[j].scatter(y[:,2*j], y[:,2*j+1], color = np.repeat(colors_rgba, num_samples , axis=0),
                          marker="x", s = 25,  label='ground truth')
            ax[j].scatter(x[:,2*j], x[:,2*j+1], color = np.repeat(colors_rgba, num_samples , axis=0),
                          edgecolor = 'black', label= 'channel input', s = 25 )
            ax[j].legend()

    if n // 2 == 1:
        plt.xlim([-max_amp, max_amp])
        plt.ylim([-max_amp, max_amp])
        plt.xlabel('$Re(y)$')
        plt.ylabel('$Im(y)$')
        title = 'Channel Output Constellation'
        plt.title(title)
    else:
        for j in range(n // 2):
            ax[j].set_xlim([-max_amp, max_amp])
            ax[j].set_ylim([-max_amp, max_amp])
            ax[j].set_xlabel('$Re(y_' + str(j + 1) + ')$')
            ax[j].set_ylabel('$Im(y_' + str(j + 1) + ')$')
            title = 'Channel Output Constellation'
            ax[j].set_title(title)
    plt.show()
    
    fig.savefig('figures/constellation_inout_'+ch_model_type+'.pdf', format='pdf')
    return            

def Constellation_Enc(enc, M):
    # Show the constellation of the learned encoder
    # Arrays for plotting
    bins = 30 #np.linspace(0.0, noise_std*4, 20)

    labels = torch.arange(0,M, dtype=int)
    m = F.one_hot(labels, num_classes=M)                    
    m = m.float().reshape(-1,M)
    x = enc(m).detach().cpu().numpy()
    
    n = x.shape[1]
    
    print("Constellation of learned encoder", flush=True)
    if n//2==1:
        plt.figure(figsize=(4,4))
        for j in range(M):
            plt.scatter(x[j,0],x[j,1])
            plt.xlabel('$Re(x)$')
            plt.ylabel('$Im(x)$')
    else:
        fig, axes = plt.subplots(1,n//2,figsize=(15,4))
        for i in range(n //2):
            for j in range(M):
                axes[i].scatter(x[j,2*i],x[j,2*i+1])
                axes[i].set_xlabel('$Re(x_' + str(i+1) + ')$')
                axes[i].set_ylabel('$Im(x_'+ str(i+1) +')$')
                axes[i].set_title("channel #"+ str(i+1))

    plt.show()
    
    return 
    
def ECDF_histogram_m(enc, ch_model, ch_model_type, device, alphas, betas, alphas_bar_sqrt,
                     one_minus_alphas_bar_sqrt,
                     num_steps, M, num_samples, batch_size, noise_std, max_amp, m=0,
                    PreT = False, denoising_alg = 'DDPM', traj = range(1), pred_type='epsilon'):
    figsize=(6.5, 2.5)
    # Test: Channel generation - ECDF of the channel output

    # Check the distribution of the amplitude for one dimension WLOG

    # Arrays for plotting
    z = np.linspace(0,max_amp,100)
    bins = 30   #np.linspace(0.0, noise_std*4, 20)
    alphas = alphas.to(device)
    betas = betas.to(device)
    one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)


    y_amp = []

    labels = torch.full([num_samples,1],m,dtype=int)
    
    msg_1h = F.one_hot(labels, num_classes=M)    
    msg_1h = msg_1h.float().reshape(-1,M)
    x = enc(msg_1h)
    
    # True channel output y

    if ch_model_type == 'AWGN':
        channel_model = ch_AWGN
    elif ch_model_type == 'Rayleigh':
        channel_model = ch_Rayleigh_AWGN_n
    elif ch_model_type == 'SSPA':
        channel_model = ch_SSPA
    else:
        raise Exception('Unrecognized channel model. Available models: AWGN, Rayleigh, SSPA.')

    y = channel_model(x,noise_std,device)

    y_amp.append(torch.sqrt(torch.sum(torch.pow(y,2), dim=1)))
    y_amp = [i.detach().cpu().numpy() for i in y_amp]
    y_amp = np.asarray(y_amp).reshape((-1,))
    
    y_gen_amp = []
    for i_btch in range(num_samples // batch_size):

        # Generated channel output y
        labels_batch = labels[batch_size*i_btch:batch_size*(i_btch+1)].to(device)
        x_batch = x[batch_size*i_btch:batch_size*(i_btch+1)].to(device)
        
        if PreT:
            c = x_batch
        else:
            c = torch.cat((labels_batch.reshape([-1,1]).to(device),x_batch), dim=1).to(device)

        if denoising_alg == 'DDPM':
            x_seq = p_sample_loop_w_Condition(ch_model, x_batch.size(), num_steps, alphas, betas, alphas_bar_sqrt,
                                          one_minus_alphas_bar_sqrt, c, pred_type=pred_type)
        elif denoising_alg == 'DDIM':
            alphas_prod = torch.cumprod(alphas, 0)
            x_seq = p_sample_loop_w_Condition_DDIM(ch_model, x_batch.size(), traj, alphas_prod, alphas_bar_sqrt,
                                               one_minus_alphas_bar_sqrt, c, pred_type=pred_type)
        else:
            assert(ValueError)

        y_gen = x_seq[-1].detach().cpu()
        y_gen = y_gen

        # Change them into the polar coordinate and append in lists        
        y_gen_amp.append(torch.sqrt(torch.sum(torch.pow(y_gen,2), dim=1) ).numpy() )

    # List of tensors -> numpy array

    y_gen_amp = np.asarray(y_gen_amp).reshape((-1,))

    print("The evaluation of generated channel in comparison to the original source; m =", m+1, flush=True)
    # Compute the empirical CDF of the ground truth random variables and the generated random variables
    y_ecdf = ECDF(y_amp)
    y_gen_ecdf = ECDF(y_gen_amp)
    fig, axes = plt.subplots(1,2,figsize= figsize,  constrained_layout=True)


    axes[1].plot(z, y_gen_ecdf(z), label = 'Generated', color = (1,0,0,0.8))
    axes[1].plot(z, y_ecdf(z), '--', label = 'True', color = (0.3,0.3,0.3,0.7))
    axes[1].legend()
    axes[1].set_xlabel('$|y|$')
    axes[1].set_ylabel('Empirical CDF')

    axes[0].hist(y_gen_amp, bins=bins, edgecolor=(1,0,0,0.8), label='Generated',facecolor=(1,1,1,0.5))
    axes[0].hist(y_amp, bins=bins, edgecolor=(0.3,0.3,0.3,0.7), linestyle = '--', label='True', facecolor=(1,1,1,0.5))
    axes[0].set_ylabel('Frequency')
    
    fig.tight_layout()
    plt.legend()
    plt.savefig('figures/ECDF_histogram_'+ch_model_type+'.pdf', bbox_inches = 'tight')
    plt.show()
    return


def ECDF_histogram(enc, ch_model, ch_model_type, device, alphas, betas, alphas_bar_sqrt,
                   one_minus_alphas_bar_sqrt, num_steps, M,
                   num_samples, batch_size, noise_std, max_amp, PreT = False, denoising_alg = 'DDPM',
                   traj = range(1), pred_type='epsilon'):
    # Test: Channel generation - ECDF of the channel output

    figsize=(3, 3)
    # Check the distribution of the amplitude for one dimension WLOG

    # Arrays for plotting
    z = np.linspace(0, max_amp , 100)
    bins = 30  # np.linspace(0.0, noise_std*4, 20)
    alphas = alphas.to(device)
    betas = betas.to(device)
    one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)

    y_amp = []

    labels = torch.empty((num_samples * M, 1), dtype=int)
    for i in range(M):
        labels[num_samples * i:num_samples * (i + 1), :] = torch.full([num_samples, 1], i, dtype=int)
    m = F.one_hot(labels, num_classes=M)
    m = m.float().reshape(-1, M)
    x = enc(m)

    n = x.shape[1]

    # True channel output y
    if ch_model_type == 'AWGN':
        channel_model = ch_AWGN
    elif ch_model_type == 'Rayleigh':
        channel_model = ch_Rayleigh_AWGN_n
    elif ch_model_type == 'SSPA':
        channel_model = ch_SSPA
    else:
        raise Exception('Unrecognized channel model. Available models: AWGN, Rayleigh, SSPA.')

    y = channel_model(x,noise_std,device).detach().cpu().numpy()

    y_amp.append(torch.sqrt(torch.sum(torch.pow(y, 2), dim=1)))
    y_amp = np.asarray(y_amp).reshape((-1,))

    for i in range(M):
        y_gen_amp = []
        for i_btch in range(num_samples // batch_size):

            # Generated channel output y
            m_batch = m[num_samples * i + batch_size * i_btch:num_samples * i + batch_size * (i_btch + 1), :].to(device)
            labels_batch = labels[num_samples * i + batch_size * i_btch:num_samples * i + batch_size * (i_btch + 1)].to(
                device)
            x_batch = x[num_samples * i + batch_size * i_btch:num_samples * i + batch_size * (i_btch + 1)].to(
                device)

            if PreT:
                c = x_batch
            else:
                c = torch.cat((labels_batch.reshape([-1, 1]).to(device), x_batch), dim=1).to(device)
                #mx = torch.cat((m, x),dim=1)
            if denoising_alg == 'DDPM':
                x_seq = p_sample_loop_w_Condition(ch_model, x_batch.size(), num_steps, alphas, betas, alphas_bar_sqrt,
                                          one_minus_alphas_bar_sqrt, c, pred_type=pred_type)
            elif denoising_alg == 'DDIM':
                alphas_prod = torch.cumprod(alphas, 0)
                x_seq = p_sample_loop_w_Condition_DDIM(ch_model, x_batch.size(), traj, alphas_prod, alphas_bar_sqrt,
                                               one_minus_alphas_bar_sqrt, c, pred_type=pred_type)

            y_gen = x_seq[-1]

            # Change them into the polar coordinate and append in lists
            y_gen_amp.append(torch.sqrt(torch.sum(torch.pow(y_gen, 2), dim=1)).detach().cpu().numpy())

        # List of tensors -> numpy array
        y_gen_amp = np.asarray(y_gen_amp).reshape((-1,))

        print("The evaluation of generated channel in comparison to the original source; m =", i + 1, flush=True)
        # Compute the empirical CDF of the ground truth random variables and the generated random variables
        y_ecdf = ECDF(y_amp[num_samples * i:num_samples * (i + 1)])
        y_gen_ecdf = ECDF(y_gen_amp)
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].plot(z / M, y_gen_ecdf(z), label='generated')
        axes[0].plot(z / M, y_ecdf(z), label='ground truth')
        axes[0].legend()
        axes[0].set_xlabel('$|y|/M$')
        axes[0].set_ylabel('Empirical CDF')

        axes[1].hist(y_gen_amp / M, bins=bins, edgecolor=(1, 0, 0, 0.8), label='generated', facecolor=(1, 1, 1, 0.5))
        axes[1].hist(y_amp[num_samples * i:num_samples * (i + 1)] / M, bins=bins, edgecolor=(0.3, 0.3, 0.3, 0.7),
                     label='ground truth', facecolor=(1, 1, 1, 0.5))
        plt.legend()
        plt.show()
    return


def show_denoising_m(channel_gen, encoder, device, alphas, betas, alphas_bar_sqrt,
                     one_minus_alphas_bar_sqrt, M =16, m = 0,
                     num_samples_dn = 1000, num_steps = 50, max_amp =3, PreT = False,
                     denoising_alg = 'DDPM', traj = range(1), pred_type = 'epsilon'):
    print('The denoising process is plotted for the message '+str(m+1)+' and the first two dimensions.')
    if m > M :
        raise Exception("The message index m should be smaller than or equal to " +str(M)+".")

    labels = m * torch.ones(1)
    msg_1h = F.one_hot(labels.long(), num_classes=M)                    
    msg_1h = msg_1h.float().reshape(-1,M) 
    x = encoder(msg_1h).repeat(num_samples_dn,1)
    #n = x.shape[1]
    
    if PreT:
        c = x
    else:
        c = torch.cat((labels.repeat(num_samples_dn,1).to(device), x ),dim=1)
        #mx = torch.cat((m, x),dim=1)
    if denoising_alg == 'DDPM':
        x_seq = p_sample_loop_w_Condition(channel_gen, x.size(), num_steps, alphas, betas, alphas_bar_sqrt,
                                          one_minus_alphas_bar_sqrt, c, pred_type = pred_type)
    elif denoising_alg == 'DDIM':
        alphas_prod = torch.cumprod(alphas, 0)
        x_seq = p_sample_loop_w_Condition_DDIM(channel_gen, x.size(), traj, alphas_prod, alphas_bar_sqrt,
                                               one_minus_alphas_bar_sqrt, c, pred_type = pred_type)
    
    len_seq = len(x_seq) - 1
    max_norm = 0.
    fig, axs = plt.subplots(1, 10, figsize=(28, 3))
    fig1, axs1 = plt.subplots(1, 10, figsize=(28, 3))
    for i in range(1, 11):
        cur_x = x_seq[i * len_seq // 10].detach().cpu()
        axs[i-1].scatter(cur_x[:, 0], cur_x[:, 1], color='white', edgecolor='gray', s=5);
        #axs[i-1].set_axis_off(); 
        axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*len_seq//10)+'})$')
        amp = torch.norm(cur_x, dim=1).numpy()
        max_norm = max( max_norm, np.max(amp))
        axs[i-1].set_xlim([-max_amp, max_amp])
        axs[i-1].set_ylim([-max_amp, max_amp])
        axs1[i-1].hist(amp,bins=20,edgecolor='gray');
        #axs1[i-1].set_axis_off(); 
        
        axs1[i-1].set_title('$q(\mathbf{x}_{'+str(i*num_steps//10)+'})$')
   
    for i in range(1,11):
        axs1[i-1].set_xlim([0, max_norm])
    plt.show()
    
    
def show_diffusion_m(channel_model, q_x, encoder, device, noise_std, M=16, m = 0, num_samples_df=1000,
                     num_steps=50, max_amp = 3):
    if m>M:
        raise Exception("The message index m should be smaller than or equal to " +str(M)+".")

    print("Diffusion process for message " + str(m+1) + " in the first two dimensions.")
    msg_1h = torch.zeros([num_samples_df, M]).to(device)
    msg_1h[:,m] = 1 # One hot encoding for message with index 1. 

    dataset_test_diff = channel_model(encoder(msg_1h) , noise_std, device)

    fig, axs = plt.subplots(1, 10, figsize=(28, 3))
    for i in range(10):
        q_i = q_x(dataset_test_diff, torch.tensor([i * num_steps//10]).to(device))
        q_i = torch.Tensor.cpu(q_i).detach().numpy()
        axs[i].scatter(q_i[:, 0], q_i[:, 1],color='white',edgecolor='gray', s=5);
        #axs[i].set_axis_off(); axs[i].set_title('$q(\mathbf{x}_{'+str(i*10)+'})$')
        axs[i].set_xlim([-max_amp, max_amp])
        axs[i].set_ylim([-max_amp, max_amp])
    plt.show()

    print("Adjust beta according to the diffusion results. If it needs to be more noisy, increase beta. If it gets noisy too soon, decrease beta.")

       
    
def Constellation_InOut_s(enc, ch_model, ch_model_type, device, alphas, betas, alphas_bar_sqrt,
                          one_minus_alphas_bar_sqrt,
                          num_steps, M, num_samples, noise_std, max_amp, PreT = False,
                          denoising_alg = 'DDPM', traj = range(1), pred_type = 'epsilon'):
    # Test: Channel generation - scatter plot
    x_maxabs = max_amp
    y_maxabs = max_amp

    labels = torch.empty((num_samples*M,1), dtype = int)
    for i in range(M):
        labels[num_samples*i:num_samples*(i+1),:] = torch.full([num_samples,1],i,dtype=int)

    m = F.one_hot(labels.long(), num_classes=M)                    
    m = m.float().reshape(-1,M)
    x = enc(m).to(device)
    n = x.shape[1] # channel dimension
    

    if ch_model_type == 'AWGN':
        channel_model = ch_AWGN
    elif ch_model_type == 'Rayleigh':
        channel_model = ch_Rayleigh_AWGN_n
    elif ch_model_type == 'SSPA':
        channel_model = ch_SSPA
    else:
        raise Exception('Unrecognized channel model. Available models: AWGN, Rayleigh, SSPA.')

    y = channel_model(x,noise_std,device).detach().cpu().numpy()

    if PreT:
        c = x
    else:
        c = torch.cat((labels.reshape([-1,1]).to(device),x), dim=1).to(device)
        
    if denoising_alg == 'DDPM':
        x_seq = p_sample_loop_w_Condition(ch_model, x.size(), num_steps, alphas, betas, alphas_bar_sqrt,
                                          one_minus_alphas_bar_sqrt, c, pred_type=pred_type)
    elif denoising_alg == 'DDIM':
        alphas_prod = torch.cumprod(alphas, 0)
        x_seq = p_sample_loop_w_Condition_DDIM(ch_model, x.size(), traj,  alphas_prod, alphas_bar_sqrt,
                                               one_minus_alphas_bar_sqrt, c, pred_type=pred_type)

    y_gen = x_seq[-1]#[num_steps]

    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y_gen = y_gen.detach().cpu().numpy()
    if n // 2 == 1:
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(n // 2, 2, figsize=(7,9))
        fig.tight_layout()
    for i in range(M):
        if n//2 ==1 :
            ax[0].scatter(y[num_samples*i:num_samples*(i+1),0], y[num_samples*i:num_samples*(i+1),1], s=5)  #edgecolor=(0,0.3,0.3,0.7), facecolor=(1,1,1,0.5)
            ax[1].scatter(y_gen[num_samples*i:num_samples*(i+1),0], y_gen[num_samples*i:num_samples*(i+1),1], s=5)
            ax[0].scatter(x[num_samples*i,0], x[num_samples*i,1], edgecolor=(0,0,0,0.8), label='channel input',facecolor=(1,1,1,0.5) , s=10)
            ax[1].scatter(x[num_samples*i,0], x[num_samples*i,1], edgecolor=(0,0,0,0.8), label='channel input',facecolor=(1,1,1,0.5) , s=10)

        else:
            for j in range(n // 2):
                ax[j,0].scatter(y_gen[num_samples*i:num_samples*(i+1),2*j], y_gen[num_samples*i:num_samples*(i+1),2*j+1])
                ax[j,1].scatter(y[num_samples*i:num_samples*(i+1),2*j], y[num_samples*i:num_samples*(i+1),2*j+1])  #edgecolor=(0,0.3,0.3,0.7), facecolor=(1,1,1,0.5)
                ax[j,0].scatter(x[num_samples*i,2*j], x[num_samples*i,2*j+1], edgecolor=(0,0,0,0.8), label='channel input',facecolor=(1,1,1,0.5) )
                ax[j,1].scatter(x[num_samples*i,2*j], x[num_samples*i,2*j+1], edgecolor=(0,0,0,0.8), label='channel input',facecolor=(1,1,1,0.5) )


    if n // 2 == 1:
        ax[0].set_xlim([-x_maxabs, x_maxabs])
        ax[0].set_ylim([-y_maxabs, y_maxabs])
        ax[1].set_xlim([-x_maxabs, x_maxabs])
        ax[1].set_ylim([-y_maxabs, y_maxabs])
        ax[0].set_title('True Channel Output')
        ax[1].set_title('Generated Channel Output')
        
        plt.savefig('figures/channel_output_constell_'+ch_model_type+'.pdf', bbox_inches= 'tight')
        extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig('figures/generated_ch_output_'+ch_model_type+'.pdf', bbox_inches=extent.expanded(1.2, 1.2))
        extent = ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig('figures/true_ch_output_'+ch_model_type+'.pdf', bbox_inches=extent.expanded(1.2, 1.2))

        
        #plt.legend()
        plt.show()
    else:
        for j in range(n // 2):
            ax[j,0].set_xlim([-x_maxabs, x_maxabs])
            ax[j,0].set_ylim([-y_maxabs, y_maxabs])
            ax[j,0].set_xlabel('$Re(y_' + str(j + 1) + ')$')
            ax[j,0].set_ylabel('$Im(y_' + str(j + 1) + ')$')
            title = 'Generated Channel Output'
            ax[j,0].set_title(title)
            ax[j,1].set_xlim([-x_maxabs, x_maxabs])
            ax[j,1].set_ylim([-y_maxabs, y_maxabs])
            ax[j,1].set_xlabel('$Re(y_' + str(j + 1) + ')$')
            ax[j,1].set_ylabel('$Im(y_' + str(j + 1) + ')$')
            title = 'True Channel Output'
            ax[j,1].set_title(title)
            plt.show()
    return   


