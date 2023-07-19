# Loss Function for Diffusion Model
# Original Source: https://github.com/acids-ircam/diffusion_models

import torch
import numpy as np
import sys
import math
assert sys.version_info >= (3, 5)
import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.rc('axes', labelsize=14)
#mpl.rc('xtick', labelsize=12)
#mpl.rc('ytick', labelsize=12)
#from scipy import special, integrate
#from scipy.spatial import distance
import torch.nn.functional as F
from torch import nn
np.random.seed(42)
#from sklearn.neighbors import NearestNeighbors


def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start

    elif schedule == "cosine":
        i = torch.linspace(0, 1, n_timesteps+1)
        f_t = torch.cos( (i + 0.008) / (1+0.008) * math.pi / 2 ) ** 2
        alphas_bar = f_t / f_t[0]
        alphas_bar_before = alphas_bar[:-1]
        alphas_bar = alphas_bar[1:]

        betas = 1 - alphas_bar / alphas_bar_before
        for i in range(n_timesteps):
            if betas[i] > 0.999:
                betas[i] = 0.999
    elif schedule == "cosine-zf":
        i = torch.linspace(0, 1, n_timesteps+1)
        f_t = torch.cos( (i + 0.008) / (1+0.008) * math.pi / 2 ) ** 2
        alphas_bar = f_t / f_t[0]
        alphas_bar_before = alphas_bar[:-1]
        alphas_bar = alphas_bar[1:]

        betas = 1 - alphas_bar / alphas_bar_before

    return betas

    return betas

def extract(input, t, shape):
    #    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape).to(input.device)

def q_posterior_mean_variance(x_0, x_t, t,posterior_mean_coef_1,posterior_mean_coef_2,posterior_log_variance_clipped):
    shape = x_0.shape
    coef_1 = extract(posterior_mean_coef_1, t, shape)
    coef_2 = extract(posterior_mean_coef_2, t, shape)
    mean = coef_1 * x_0 + coef_2 * x_t
    var = extract(posterior_log_variance_clipped, t, shape)
    return mean, var

def p_mean_variance(model, x, t):
    # Go through model
    out = model(x, t)
    # Extract the mean and variance
    mean, log_var = torch.split(out, 2, dim=-1)
    var = torch.exp(log_var)
    return mean, log_var
'''
def p_sample(model, x, t,alphas,betas,one_minus_alphas_bar_sqrt):
    t = torch.tensor([t])
    shape = x.shape
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, shape)) / extract(one_minus_alphas_bar_sqrt, t, shape))
    # Model output
    eps_theta = model(x, t)
    # Final values
    mean = (1 / extract(alphas, t, shape).sqrt()) * (x - (eps_factor * eps_theta))
    # Generate z
    z = torch.randn_like(x)
    # Fixed sigma
    sigma_t = extract(betas, t, shape).sqrt() # Simplified version
    #sigma_t =  ( 1 - a ** 2 / a_next ** 2) ** 0.5 * am1_next / am1 # full version
    sample = mean + sigma_t * z
    return (sample)

def p_sample_loop(model, shape, n_steps,alphas,betas,one_minus_alphas_bar_sqrt):
    device = model.this_device
    cur_x = torch.randn(shape).to(device)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i,alphas,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq
'''
def p_sample_w_Condition(model, z, t, alphas, betas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, c):
    device = model.this_device
    z = z.to(device)
    c = c.to(device)
    shape = z.shape
    t = torch.tensor([t]).to(device)

    ## The commented lines below are necessary only when the full version of sigma_t is used.
    #a = extract(alphas_bar_sqrt, t, shape)
    #am1 = extract(one_minus_alphas_bar_sqrt, t, shape)
    #if t > 0:
    #    a_next = extract(alphas_bar_sqrt, t - 1, shape)
    #    am1_next = extract(one_minus_alphas_bar_sqrt, t - 1, shape)
    #else:
    #    a_next = torch.ones_like(z)
    #    am1_next = torch.zeros_like(z)

    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, shape)) / extract(one_minus_alphas_bar_sqrt, t, shape))
    eps_factor = eps_factor.to(device)
    # Model output
    eps_theta = model(z, t, c)
    # Final values
    mean = (1 / extract(alphas, t, shape).sqrt().to(device)) * (z - (eps_factor * eps_theta))
    # Generate epsilon
    e = torch.randn_like(z).to(device)
    # Fixed sigma
    sigma_t = extract(betas, t, shape).sqrt().to(device) # Simplified version
    #sigma_t =  ( 1 - a ** 2 / a_next ** 2) ** 0.5 * am1_next / am1 # full version

    sample = mean.to(device) + sigma_t.to(device) * e
    sample = sample.to(device)
    return (sample)


def p_sample_w_Condition_v(model, z, t, alphas, betas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, c):  # stochastic sampling with condition

    device = model.this_device
    z = z.to(device)
    c = c.to(device)
    shape = z.shape
    t = torch.tensor([t]).to(device)

    a = extract(alphas_bar_sqrt, t, shape)
    am1 = extract(one_minus_alphas_bar_sqrt, t, shape)

    if t > 0:
        a_next = extract(alphas_bar_sqrt, t - 1, shape)
        am1_next = extract(one_minus_alphas_bar_sqrt, t - 1, shape)
    else:
        a_next = torch.ones_like(z)
        am1_next = torch.zeros_like(z)


    # Model output
    v_theta = model(z, t, c)
    e = torch.randn_like(z)

    # Generate Sample
    e_hat = am1 * z + a * v_theta
    x_hat = a * z - am1 * v_theta
    sample = a_next * x_hat + am1_next ** 2 * a / am1 / a_next * e_hat + (
            1 - a ** 2 / a_next ** 2) ** 0.5 * am1_next / am1 * e

    return sample


def p_sample_loop_w_Condition(model, shape, n_steps, alphas, betas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, c,
                              pred_type='epsilon', is_light = False):
    device = model.this_device
    cur_x = torch.randn(shape).to(device)
    c = c.to(device)
    alphas = alphas.to(device)
    betas = betas.to(device)
    alphas_bar_sqrt = alphas_bar_sqrt.to(device)
    one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)

    if pred_type == 'epsilon':
        p_sample_func = p_sample_w_Condition
    elif pred_type == 'v':
        p_sample_func = p_sample_w_Condition_v
    if not is_light:
        x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample_func(model, cur_x, i, alphas, betas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, c)
        if not is_light:
            x_seq.append(cur_x)
    if not is_light:
        return x_seq
    else: return cur_x

def p_sample_w_Condition_DDIM(model, xt, i, j, alphas_prod, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, shape, device, c):
    at = extract(alphas_prod, torch.tensor([i]).to(device), shape).to(device)
    if j > -1:
        at_next = extract(alphas_prod, torch.tensor([j]), shape).to(device)
    else:
        at_next = torch.ones(shape).to(device)  # a_0 = 1

    et = model(xt, torch.tensor([i]).to(device), c)
    #        x0_t =  (xt - et * (1 - at).sqrt()) / at.sqrt()
    #        c2 = (1 - at_next).sqrt()
    #        xt_next = at_next.sqrt() * x0_t + c2 * et
    xt_next = at_next.sqrt() * (xt - et * (1 - at).sqrt()) / at.sqrt() + (1 - at_next).sqrt() * et
    return xt_next

def p_sample_w_Condition_DDIM_v(model, xt, i, j, alphas_prod, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, shape, device, c):
    a = extract(alphas_bar_sqrt, torch.tensor([i]).to(device), shape).to(device)
    am1 = extract(one_minus_alphas_bar_sqrt, torch.tensor([i]).to(device), shape).to(device)
    if j > -1:
        a_next = extract(alphas_bar_sqrt, torch.tensor([j]), shape).to(device)
        am1_next = extract(one_minus_alphas_bar_sqrt, torch.tensor([j]), shape).to(device)
    else:
        a_next = torch.ones(shape).to(device)  # a_0 = 1
        am1_next = torch.zeros(shape).to(device)  # 1-sqrt(a_0) = 0
    z = x_seq[-1].to(device)
    v_theta = model(z, torch.tensor([i]).to(device), c)

    # Generate Sample
    x_hat = a * z - am1 * v_theta
    z_next = a_next * x_hat + am1_next * (z - a * x_hat) / am1

def p_sample_loop_w_Condition_DDIM(model, shape, traj, alphas_prod, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, c,
                                   pred_type='epsilon'):
    device = model.this_device
    cur_x = torch.randn(shape).to(device) # randomly sampled x_T
    c = c.to(device)
    alphas_prod = alphas_prod.to(device)
    alphas_bar_sqrt = alphas_bar_sqrt.to(device)
    one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)


    traj_next = [-1] + list(traj[:-1]) # t-1

    x_seq = [cur_x]

    if pred_type == 'epsilon':
        p_sample_func_DDIM = p_sample_w_Condition_DDIM
    elif pred_type == 'v':
        p_sample_func_DDIM = p_sample_w_Condition_DDIM_v

    for i, j in zip(reversed(traj), reversed(traj_next)): # i = t, j = t-1
        xt = x_seq[-1].to(device)
        xt_next = p_sample_func_DDIM(model, xt, i, j, alphas_prod, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,
                                     shape, device, c)
        x_seq.append(xt_next)

    return x_seq

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(torch.tensor(np.sqrt(2.0 / np.pi)) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, means, log_scales):
    # Assumes data is integers [0, 255] rescaled to [-1, 1]
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
    log_one_minus_cdf_min = torch.log(torch.clamp(1 - cdf_min, min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(x < -0.999, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(torch.clamp(cdf_delta, min=1e-12))))
    return log_probs

def normal_kl(mean1, logvar1, mean2, logvar2):
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
    return kl

def q_sample(x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt ,noise=None):
    shape = x_0.shape
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(alphas_bar_sqrt, t, shape)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, shape)
    return (alphas_t * x_0 + alphas_1_m_t * noise)
'''
def loss_variational(model, x_0,alphas_bar_sqrt, one_minus_alphas_bar_sqrt,posterior_mean_coef_1,posterior_mean_coef_2,posterior_log_variance_clipped,n_steps):
    batch_size = x_0.shape[0]
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,))
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()
    # Perform diffusion for step t
    x_t = q_sample(x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
    # Compute the true mean and variance
    true_mean, true_var = q_posterior_mean_variance(x_0, x_t, t,posterior_mean_coef_1,posterior_mean_coef_2,posterior_log_variance_clipped)
    # Infer the mean and variance with our model
    model_mean, model_var = p_mean_variance(model, x_t, t)
    # Compute the KL loss
    kl = normal_kl(true_mean, true_var, model_mean, model_var)
    kl = torch.mean(kl.view(batch_size, -1), dim=1) / np.log(2.)
    # NLL of the decoder
    decoder_nll = -discretized_gaussian_log_likelihood(x_0, means=model_mean, log_scales=0.5 * model_var)
    decoder_nll = torch.mean(decoder_nll.view(batch_size, -1), dim=1) / np.log(2.)
    # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
    output = torch.where(t == 0, decoder_nll, kl)
    return output.mean(-1)

def noise_estimation_loss(model, x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):
    shape = x_0.shape
    batch_size = shape[0]
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,))
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()
    # x0 multiplier
    a = extract(alphas_bar_sqrt, t, shape)
    # eps multiplier
    am1 = extract(one_minus_alphas_bar_sqrt, t, shape)
    e = torch.randn_like(x_0)
    # model input
    x = x_0 * a + e * am1
    output = model(x, t)
    return (e - output).square().mean()
'''

def noise_estimation_loss(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps, n):
    device = model.this_device
    shape = x_0.shape
    batch_size = x_0.shape[0]
    x = x_0[:, 0:n].to(device)
    c = x_0[:, n:].to(device)

    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,)).to(device)
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()
    # x0 multiplier
    a = extract(alphas_bar_sqrt, t, shape).to(device)
    # eps multiplier
    am1 = extract(one_minus_alphas_bar_sqrt, t, shape).to(device)
    e = torch.randn_like(x).to(device)
    # model input
    y = x * a + e * am1
    output = model(y, t, c)
    return (e - output).square().mean()


def v_estimation_loss(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps, n,
                      losstype='no_weighting'):  # , prediction_var = 'x'):

    assert isinstance(losstype, str)
    #    assert isinstance(prediction_var, str)

    device = model.this_device
    shape = x_0.shape
    batch_size = x_0.shape[0]
    x = x_0[:, 0:n].to(device)
    c = x_0[:, n:].to(device)

    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,)).to(device)
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()
    # x0 multiplier
    a = extract(alphas_bar_sqrt, t, shape).to(device)
    # eps multiplier
    am1 = extract(one_minus_alphas_bar_sqrt, t, shape).to(device)
    e = torch.randn_like(x).to(device)

    # model input - the noisy sample
    y = x * a + e * am1
    output = model(y, t, c)

    # loss weighting
    if losstype == 'SNR':
        weight_t = a ** 2 / am1 ** 2
    elif losstype == 'SNRp1':
        weight_t = a ** 2 / am1 ** 2 + torch.ones_like(x)
    elif losstype == 'SNRtr':
        weight_t = a ** 2 / am1 ** 2
        weight_t[weight_t < 1] = 1
    elif losstype == 'no_weighting':
        weight_t = 1
    else:
        raise ValueError('Unrecognized weight type. Possible options are SNR, SNRp1, SNRtr, and no_weighting.')

    # loss corresponding to the model output

    v = e * a - x * am1

    #    if prediction_var == 'e': # epsilon_prediction
    #        v_hat = output * a - x * am1
    #    elif prediction_var == 'x':  # x_prediction
    #        v_hat = e * a - output * am1
    #    elif prediction_var == 'v':
    #        v_hat = output
    #    else: raise ValueError('Unrecognized variable for prediction. Possible options are e, x, v. e represents epsilon.' )

    loss = (v - output).square()

    return (weight_t * loss).mean()  # (v - output).square().mean()

def EbNo_to_noise(ebnodb, rate):
    '''Transform EbNo[dB]/snr to noise power'''
    ebno = 10**(ebnodb/10)
    noise_std = 1/np.sqrt(2*rate*ebno)
    return noise_std

def SNR_to_noise(snrdb):
    '''Transform EbNo[dB]/snr to noise power'''
    snr = 10**(snrdb/10)
    noise_std = 1/np.sqrt(2*snr)
    return noise_std

def B_Ber_m(input_msg, msg):
    '''Calculate the Batch Symbol Error Rate'''
    batch, bits = msg.size()

    pred_error = torch.ne(input_msg, torch.round(msg))
    bber = torch.sum(pred_error)/(batch*bits)
    return bber

def Interleaver_set(dataset):
    '''Interleave the whole codeword set at once, set length needs to be a mupltiple of 255'''
    output=[dataset[i:i + 255].transpose() for i in range(0, len(dataset), 255)]
    return np.vstack(output)

def prediction_errors(input_msg, msg):
    pred_error = torch.ne(torch.flatten(input_msg), msg.argmax(dim=1))
    return pred_error

def block_error(prediction_matrix, erasures_matrix, erasure=False):
    if erasure:
        errors_wo_erasures = torch.logical_and(prediction_matrix, torch.logical_not(erasures_matrix))
        errors = torch.sum(errors_wo_erasures, dim=1)
        erasures = torch.sum(erasures_matrix, dim=1)
        #if torch.sum(erasures>1)>1: print(torch.sum(erasures>1))
        errors_tot = torch.sum(2*errors + erasures > 32)
        block_error = errors_tot/255

    else:
        errors = torch.sum(prediction_matrix, dim=1)
        block_error =torch.sum(errors>16)/255

    return block_error

def SER(input_msg, msg, erasure_bound=0.9):

    '''Calculate the Symbol Error Rate'''
    batch, _ = msg.size()

    # count erasures and give indices without erasures
    smax = torch.nn.Softmax(dim=1)
    out, _ = torch.topk(smax(msg), 1, axis=1)
    indices_erasures = (out <= erasure_bound).flatten()#.nonzero()
    #erasures = out.size()[0]-indices_non_erasures.size()[0]

    #erasures = torch.count_nonzero(out < erasure_bound).item()

    # cut erasures from prediction
    #msg_wo_er = torch.index_select(msg, 0, indices_non_erasures.flatten())
    #input_msg_wo_er = torch.index_select(torch.flatten(input_msg), 0, indices_non_erasures.flatten()) 

    #print("input_msg", input_msg)
    #print("msg", msg.argmax(dim=1))

    input_msg_wo_er = torch.flatten(input_msg)
    msg_wo_er = msg

    pred_error = torch.ne(input_msg_wo_er, msg_wo_er.argmax(dim=1))
    block_er = torch.sum(pred_error)
    ser = block_er/batch


    return ser, pred_error, indices_erasures

def Block_ER(input_msg, msg):
    '''Calculate the Batch Symbol Error Rate'''
    batch, bits = msg.size()

    pred_error = torch.ne(input_msg, torch.round(msg))
    ber,_ = torch.max(pred_error, dim=1)
    bber = torch.sum(ber)
    if bber < 17:
        bler = 0.
    else:
        bler = 1.
    return bler

def random_sample(batch_size=32, M=16):
    msg = torch.randint(0,2, size=(batch_size, 1))
    return msg

def test_encoding(encoder, M=16, n=1):
    inp = np.arange(0,M)
    coding = encoder(inp).detach()
    fig = plt.figure(figsize=(4,4))
    plt.plot(coding[:,0], coding[:, 1], "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18, rotation=0)
    plt.grid(True)
    plt.gca().set_ylim(-2, 2)
    plt.gca().set_xlim(-2, 2)
    plt.show()

def test_noisy_codeword(data):
    rcvd_word = data[1:2000]
    fig = plt.figure(figsize=(4,4))
    plt.plot(rcvd_word[:,0], rcvd_word[:, 1], "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18, rotation=0)
    plt.grid(True)
    plt.gca().set_ylim(-2, 2)
    plt.gca().set_xlim(-2, 2)
    plt.show()

def minimum_distance(encoder, M, n):
    inp = np.arange(0,M)
    code = encoder(inp).detach()
    code_flat = np.reshape(code, (M,n))
    distmat = distance.cdist(code_flat,code_flat, 'euclidean')+np.identity(M)*np.exp(100)
    return np.min(distmat)

def plot_loss(encoder, M, n, step, epoch, mean_loss, X_batch, y_pred, plot_encoding):
    template = 'Iteration: {}, Epoch: {}, Loss: {:.5f}, Batch_BER: {:.5f}'
    if step % 500 == 0:
        print(template.format(step, epoch, mean_loss, B_Ber_m(X_batch, y_pred)))
        if plot_encoding:
            test_encoding(encoder, M=16, n=1)

def plot_batch_loss(epoch, mean_loss, X_batch, y_pred):
    template_outer_loop = 'Interim result for Epoch: {}, Loss: {:.5f}, Batch_BER: {:.5f}'
    print(template_outer_loop.format(epoch, mean_loss, B_Ber_m(X_batch, y_pred)))

# Approximate 16 QAM Error

def SIXT_QAM_sim(ebnodb):
    return (3.0/2)*special.erfc(np.sqrt((4.0/10)*10.**(ebnodb/10)))

def BPSK_rayleigh(ebnodb, n, k):
    ebn0 = 10.**(ebnodb/10)
    return 0.5 * (1-np.sqrt((n/k)*ebn0 / ((n/k)*ebn0+1)))

def MQAM_rayleigh_approx(M, ebnodb):
    ebno = 10.**(ebnodb/10)
    esno = 4*ebno
    #Goldsmith, p.185, 6.3.2, Eqn 6.61, alphaM=4, betaM=3/(M-1)
    a=3.25
    b=3/(M-1)
    e=b*esno
    return (a/2)*(1-np.sqrt(0.5*e / (1+0.5*e) ) ), a/(2*b*esno)

def rayleigh_error(ebnodb):
    esno=4*10.**(ebnodb/10)
    def awgn_ser(esno):
        return 1.5*special.erfc(np.sqrt(0.1*esno))
    def weighted_ser(x):
        weight=2*x*np.exp(-x*np.conj(x))
        return weight*awgn_ser(esno*x*np.conj(x))
    r,e = integrate.quad(weighted_ser,0,np.inf)
    return r

def PSK_1h(m, avg_power): # m is one-hot encdoed tensor 
    M = m.size()[1] # message cardinality
    label = torch.argmax(m,dim=1)

    x= torch.zeros([m.size()[0],2]) # complex channel
    rad = torch.linspace(0,2*math.pi,steps=M+1)[:M] # phase

    x[:,0] = math.sqrt(avg_power/2)*torch.cos(rad[label])
    x[:,1] = math.sqrt(avg_power/2)*torch.sin(rad[label])
    return x

def NN_det_1h(r_seq, table):
    m = torch.zeros([r_seq.size()[0],table.size()[0]])
    #sym = torch.zeros([r_seq.size()[0],table.size()[1]])
    for i in range(r_seq.size()[0]):
        distance = torch.sum((table - r_seq[i,:]) ** 2  ,dim=1)
        m[i,torch.argmin(distance, dim=0)] = 1
        #sym[i,:] = table[torch.argmin(distance, dim=0), :]
    return m

def NN_det_old(r_seq, table):
    sym = torch.zeros([r_seq.size()[0],table.size()[1]])
    for i in range(r_seq.size()[0]):
        distance = torch.sum((table - r_seq[i,:]) ** 2  ,dim=1)
        sym[i,:] = table[torch.argmin(distance, dim=0), :]
    return sym

def NN_det(r_seq, table):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(table)
    _, indices = nbrs.kneighbors(r_seq)
    return indices

def qam16_mapper_n(m):
    # m takes in a vector of messages
    x = np.linspace(-3,3,4)
    y = np.meshgrid(x,x)
    z = np.array(y).reshape(2,16)
    z = z / np.std(z)

    return np.array([[z[0][i],z[1][i]] for i in m])



class LoadModels:
    def __init__(self, path, Load_model_gen,  model_gen, tag_load_gen, Load_model_AE, model_enc, model_dec, tag_load_AE):
        self.Load_model_gen = Load_model_gen
        self.model_gen = model_gen
        self.dir_gen = path + '/channel_gen' + tag_load_gen
        self.Load_model_AE = Load_model_AE
        self.model_enc = model_enc
        self.model_dec = model_dec
        self.dir_enc = path + '/encoder' + tag_load_AE
        self.dir_dec = path + '/decoder' + tag_load_AE

    def load(self):
        if self.Load_model_gen:
            self.model_gen.load_state_dict(torch.load(self.dir_gen))
            self.model_gen.eval()
        if self.Load_model_AE:
            self.model_enc.load_state_dict(torch.load(self.dir_enc))
            self.model_enc.eval()
            self.model_dec.load_state_dict(torch.load(self.dir_dec))
            self.model_dec.eval()
        return self.model_gen, self.model_enc, self.model_dec

class SaveModels:
    def __init__(self, path, Save_model_gen,  model_gen, tag_save_gen, Save_model_AE, model_enc, model_dec, tag_save_AE):
        self.Save_model_gen = Save_model_gen
        self.model_gen = model_gen
        self.dir_gen = path + '/channel_gen' + tag_save_gen
        self.Save_model_AE = Save_model_AE
        self.model_enc = model_enc
        self.model_dec = model_dec
        self.dir_enc = path + '/encoder' + tag_save_AE
        self.dir_dec = path + '/decoder' + tag_save_AE

    def save_gen(self):
        if self.Save_model_gen:
            torch.save(self.model_gen.state_dict(), self.dir_gen)
    def save_AE(self):
        if self.Save_model_AE:
            torch.save(self.model_enc.state_dict(), self.dir_enc)
            torch.save(self.model_dec.state_dict(), self.dir_dec)

		
 
