# The channel models are defined here.

import torch
import numpy as np

def ch_AWGN(ch_input, AWGN_std, device):
    ch_input = ch_input.to(device)
    ch_output = ch_input + torch.normal(mean=0, std=AWGN_std, size = ch_input.size()).to(device)
    return ch_output


def ch_Rayleigh_AWGN(ch_input, Rayleigh_param, AWGN_std, device):
    # y = hx + z
    # h follows a Rayleigh distribution. 
    ch_input = ch_input.to(device)
    h = torch.sqrt(torch.normal(mean=0, std=Rayleigh_param, size=ch_input.size()) **2 + torch.normal(mean=0, std=Rayleigh_param, size=ch_input.size()) **2 ).to(device)
    z = torch.normal(mean=0, std=AWGN_std, size= ch_input.size()).to(device) 
    
    ch_output = h * ch_input + z 
    
    return ch_output
 
    
def ch_Rayleigh_AWGN_n(codeword, noise_std, dvc):
    codeword = codeword.to(dvc)
    i,j = codeword.size()
    h_ray = (1/np.sqrt(2))*torch.sqrt( torch.randn((i,j), device = dvc)**2 + torch.randn((i,j), device = dvc)**2)
    
    channel_noise = noise_std*torch.randn((i,j), device = dvc)
                
    return codeword + torch.div(channel_noise, h_ray)


def ch_SSPA(x, sigma_n, device, p = 3.0, A_0 = 1.5, v = 5.): # A_0 limiting output amplitude, v is small signal gain.
    assert x.size(1) % 2 == 0 

    # x= ([1 2 3 4] ,[5 6 7 8])
    dim = int(x.size(1) //2)
    x_2d = x.reshape(-1,2) #x_2d =([1 2], [3 4] , [5 6], [7 8])
    A = torch.sum(x_2d ** 2, dim=1) ** 0.5 # Amplitude
    A_ratio = v / (1+ (v*A/A_0)**(2*p) )**(1/2/p) # g_A / A
    x_amp_2d = torch.mul(A_ratio.reshape(-1,1), x_2d)
    x_amp = x_amp_2d.reshape(-1, 2*dim)
    y = x_amp + sigma_n * torch.randn_like(x)
    
    return y
