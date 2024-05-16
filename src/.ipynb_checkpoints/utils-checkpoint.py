# Loss Function for Diffusion Model
# Original Source: https://github.com/acids-ircam/diffusion_models

import torch
import numpy as np
import sys
import math
assert sys.version_info >= (3, 5)
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
np.random.seed(42)


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

def extract(input, t, shape):
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

def q_sample(x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt ,noise=None):
    shape = x_0.shape
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(alphas_bar_sqrt, t, shape)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, shape)
    return (alphas_t * x_0 + alphas_1_m_t * noise)

def noise_estimation_loss(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps, n):
    device = model.this_device
    shape = x_0.shape
    batch_size = x_0.shape[0]
    x = x_0[:, 0:n].to(device)
    c = x_0[:, n:].to(device)

    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size,)).long().to(device)
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
    t = torch.randint(0, n_steps, size=(batch_size,)).long().to(device)
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

    v = e * a - x * am1

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

		
 
