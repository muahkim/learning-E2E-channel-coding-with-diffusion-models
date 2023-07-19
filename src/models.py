#An Implementation of Diffusion Network Model
#Oringinal source: https://github.com/acids-ircam/diffusion_models

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from channel_models import *

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out, device =self.this_device )
        self.embed = nn.Embedding(n_steps, num_out, device = self.this_device) # for each step, the distribution is distinct. Given the number of steps, the embedding returns a gamma for it.
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        x = x.type(torch.FloatTensor).to(self.this_device)
        out = self.lin(x).to(self.this_device)
        gamma = self.embed(y).to(self.this_device)
        out = gamma.view(-1, self.num_out).to(self.this_device) * out # Why is it multiplicative? not additive?
        return out
        
class ConditionalModel(nn.Module):
    def __init__(self, n_steps):
        super(ConditionalModel, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.lin1 = ConditionalLinear(2, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, 2, device =self.this_device)
    
    def forward(self, x, y):
        x = x.type(torch.FloatTensor).to(self.this_device)
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        return self.lin4(x)
    
    
class ConditionalLinear_w_Condition(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear_w_Condition, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out, device = self.this_device)
        self.embed = nn.Embedding(n_steps, num_out, device = self.this_device) # for each step, the distribution is distinct. Given the number of steps, the embedding returns a gamma for it.
        self.embed.weight.data.uniform_()

    def forward(self, x, y, c):
        x = x.type(torch.FloatTensor).to(self.this_device)
        c = c.type(torch.FloatTensor).to(self.this_device)
        out = self.lin(torch.cat((x,c), dim=1)).to(self.this_device)
        gamma = self.embed(y).to(self.this_device)
        out = gamma.view(-1, self.num_out).to(self.this_device) * out # Why is it multiplicative? not additive?
        return out

    
class ConditionalModel_w_Condition(nn.Module):
    def __init__(self, n_steps, M, n, N):
        super(ConditionalModel_w_Condition, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.lin1 = ConditionalLinear_w_Condition(2*n+M, N, n_steps) # hidden layer 128
        self.lin2 = ConditionalLinear(N, N, n_steps)
        self.lin3 = ConditionalLinear(N, N, n_steps)
        self.lin4 = nn.Linear(N, n, device = self.this_device)
    
    def forward(self, x, y, c):
        x = x.type(torch.FloatTensor).to(self.this_device)
        c = c.type(torch.FloatTensor).to(self.this_device)

        x = F.softplus(self.lin1(x, y, c))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        return self.lin4(x)


class ConditionalLinear_w_Msg(nn.Module):
    def __init__(self, num_in, num_out, n_steps, M):
        super(ConditionalLinear_w_Msg, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.num_out = num_out
        self.M = M
        self.lin = nn.Linear(num_in, num_out, device = self.this_device)
        self.embed_t = nn.Embedding(n_steps, num_out, device = self.this_device)
        self.embed_t.weight.data.uniform_()
        self.embed_m = nn.Embedding(M, num_out, device = self.this_device)
        self.embed_m.weight.data.uniform_()

    def forward(self, x, y, m):
        x = x.type(torch.FloatTensor).to(self.this_device)
        m = m.type(torch.IntTensor).to(self.this_device)
        out = self.lin(x).to(self.this_device)# self.lin(torch.cat((x,ch_in), dim=1)).to(self.this_device)
        gamma_t = self.embed_t(y).to(self.this_device)
        gamma_m = self.embed_m(m).to(self.this_device)
        out = gamma_t.view(-1, self.num_out).to(self.this_device) * gamma_m.view(-1, self.num_out).to(self.this_device) * out
        return out
    
class ConditionalLinear_w_ChIn(nn.Module):
    def __init__(self, num_in, num_out, n_steps, n):
        super(ConditionalLinear_w_ChIn, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.num_out = num_out
#        self.n = n
        self.lin = nn.Linear(num_in, num_out, device = self.this_device)
        self.embed_t = nn.Embedding(n_steps, num_out, device = self.this_device) 
        self.embed_t.weight.data.uniform_()
        self.lin_c = nn.Linear(n, num_out, device = self.this_device) 
        self.lin_c.weight.data.uniform_()

    def forward(self, x, y, c):
        x = x.type(torch.FloatTensor).to(self.this_device)
        c = c.type(torch.FloatTensor).to(self.this_device)
        out = self.lin(x).to(self.this_device)# self.lin(torch.cat((x,ch_in), dim=1)).to(self.this_device)
        gamma_t = self.embed_t(y).to(self.this_device)
        gamma_c = self.lin_c(c).to(self.this_device)
        out = gamma_t.view(-1, self.num_out).to(self.this_device) * gamma_c.view(-1, self.num_out).to(self.this_device) * out 
        return out

    
class ConditionalModel_w_ChIn(nn.Module):
    def __init__(self, n_steps, M, n):
        super(ConditionalModel_w_ChIn, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.M = M
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.lin1 = ConditionalLinear_w_Msg(2*n, 128, n_steps, M)
        self.lin2 = ConditionalLinear_w_Msg(128, 128, n_steps, M)
        self.lin3 = ConditionalLinear_w_Msg(128, 128, n_steps, M)
        self.lin4 = nn.Linear(128, n, device = self.this_device)
    
    def forward(self, x, y, c):
        x = x.type(torch.FloatTensor).to(self.this_device)
        c = c.type(torch.FloatTensor).to(self.this_device)
        m = c[:,0].to(self.this_device)
        
        y = y.reshape([-1,1]).to(self.this_device)
        ch_in = c[:,1:]
        xch_in = torch.cat((x, ch_in), dim=1).to(self.this_device)
        x = F.softplus(self.lin1(xch_in, y, m))
        x = F.softplus(self.lin2(x, y, m))
        x = F.softplus(self.lin3(x, y, m))
        return self.lin4(x)


    
# Binary Input Encoders/decoders: 
class Encoder(nn.Module):
    def __init__(self, k, M, n, **extra_kwargs):
        super(Encoder, self).__init__()
        
        use_cuda =  torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        self._f = nn.Sequential(            
            nn.Linear(k, M),
            nn.ELU(),
            nn.Linear(M,M),
            nn.ELU(),
            nn.Linear(M, n)#,norm_layer
        )
        
    def power_constraint(self, codes):
        codes_mean = torch.mean(codes)
        codes_std  = torch.std(codes)
        codes_norm = (codes-codes_mean)/ codes_std
        
        return codes_norm
        
    def forward(self, inputs):
        inputs = inputs.type(torch.FloatTensor).to(self.this_device)
        _x = self._f(inputs)
        x = self.power_constraint(_x)
        return x
    
    
class Decoder(nn.Module):
    def __init__(self,k, M ,n, **extra_kwargs):
        super(Decoder, self).__init__()
        self._f = nn.Sequential(
               nn.Linear(n, M),
               nn.ELU(),
               nn.Linear(M, M),
               nn.ELU(),
               nn.Linear(M, k),
               nn.Sigmoid()
                )
        
    def forward(self, inputs):
        x = self._f(inputs)
        return x
    
# one-hot encoders/decoders:
class Encoder_1h(nn.Module):
    def __init__(self, M, n, P_in, **extra_kwargs):
        super(Encoder_1h, self).__init__()

        use_cuda =  torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.P_in = P_in
        self._f = nn.Sequential(
            nn.Linear(M, M),
            nn.ELU(),
            nn.Linear(M,M),
            nn.ELU(),
            #nn.Linear(M,n),
            #nn.ELU(),
            nn.Linear(M, n)#,norm_layer
        )

    def power_constraint(self, codes):
        codes_mean = torch.mean(codes)
        codes_std  = torch.std(codes)
        #norm = torch.mean(torch.sqrt(torch.sum(codes**2, dim=1)))
        #P_in = 10**((0-30)/10) # Input power in W. 1e-3
        codes_norm = (codes-codes_mean)/ codes_std* np.sqrt(self.P_in/2)

        return codes_norm
    
    def forward(self, inputs):
        inputs = inputs.to(self.this_device)
        _x = self._f(inputs)
        x = self.power_constraint(_x)
        return x

class Encoder_1h_n(nn.Module): #normalized
    def __init__(self, M, n, **extra_kwargs):
        super(Encoder_1h_n, self).__init__()
        
        use_cuda =  torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        
        self._f = nn.Sequential(
            nn.Linear(M, M),
            nn.ELU(),
            nn.Linear(M,M),
            nn.ELU(),
            #nn.Linear(M,n),
            #nn.ELU(),
            nn.Linear(M, n)#,norm_layer
        )
        
    def power_constraint(self, codes):
        codes_mean = torch.mean(codes)
        codes_std  = torch.std(codes)
        codes_norm = (codes-codes_mean)/ codes_std
        
        return codes_norm 
        
    def forward(self, inputs):
        inputs = inputs.to(self.this_device)
        _x = self._f(inputs)
        x = self.power_constraint(_x)
        return x

class Encoder_1h_2(nn.Module): #power constraint
    def __init__(self, M, n, P_in, **extra_kwargs):
        super(Encoder_1h_2, self).__init__()

        use_cuda =  torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")
        self.P_in = P_in
        self.M = M
        self._f = nn.Sequential(
            nn.Linear(M, M),
            nn.ELU(),
            nn.Linear(M,M),
            nn.ELU(),
            #nn.Linear(M,n),
            #nn.ELU(),
            nn.Linear(M, n)#,norm_layer
        )
    def power(self):
        inputs = torch.eye(self.M).to(self.this_device)
        codes = self._f(inputs)
        P = torch.mean(torch.sum(codes**2, dim=1))
        return P

    def power_constraint(self, codes):
        P = self.power() #torch.mean(torch.sum(codes**2, dim=1))
        codes_norm = codes * np.sqrt(self.P_in) / torch.sqrt(P)
        return codes_norm

    def forward(self, inputs):
        inputs = inputs.to(self.this_device)
        _x = self._f(inputs)
        x = self.power_constraint(_x)
        return x

class Decoder_1h(nn.Module):
    def __init__(self, M ,n, **extra_kwargs):
        super(Decoder_1h, self).__init__()
        self._f = nn.Sequential(
               #nn.Linear(n, n),
               #nn.ELU(),
               nn.Linear(n, M),
               nn.ELU(),
               nn.Linear(M, M),
               nn.ELU(),
               nn.Linear(M, M),
                )
        
    def forward(self, inputs):
        x = self._f(inputs)
        return x



    
class Channel(torch.nn.Module):
    def __init__(self, enc, dec):
        super(Channel, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec

    def forward(self, inputs, noise_std):
        #End-to-End Framework
        codeword  = self.enc(inputs)
        
        i,j = codeword.size()
        channel_noise = noise_std*torch.randn((i, j)).to(self.device) 
        
        rec_signal = codeword + channel_noise
        dec_signal = self.dec(rec_signal)

        return dec_signal

    
    
class Channel_ray_n(torch.nn.Module):
    def __init__(self, enc, dec):
        super(Channel_ray_n, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec

    def forward(self, inputs, noise_std):
        #End-to-End Framework
        codeword  = self.enc(inputs)
        
        i,j = codeword.size()
        
        h_ray = (1/np.sqrt(2))*torch.sqrt( torch.randn((i, j), device = self.device)**2 + torch.randn((i, j), device = self.device)**2)
        channel_noise = noise_std*torch.randn((i, j), device = self.device)
        
        rec_signal = codeword + torch.div(channel_noise, h_ray)
        dec_signal = self.dec(rec_signal)

        return dec_signal
    
class Channel_ray(torch.nn.Module):
    def __init__(self, enc, dec):
        super(Channel_ray, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec

    def forward(self, inputs, noise_std, Ray_param):
        #End-to-End Framework
        codeword  = self.enc(inputs)
        
        i,j = codeword.size()
        
        h_ray = (1/np.sqrt(2))*Ray_param*torch.sqrt( torch.rand(size = codeword.size(), device = self.device)**2 +torch.rand(size = codeword.size(), device = self.device)**2)
        channel_noise = noise_std*torch.randn((i, j), device = self.device) 
        
        rec_signal = codeword*h_ray + channel_noise
        dec_signal = self.dec(rec_signal)

        return dec_signal
    

class Channel_SSPA(torch.nn.Module):
    def __init__(self, enc, dec):
        super(Channel_SSPA, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.enc = enc
        self.dec = dec
    def forward(self, inputs, noise_std):
        codeword = self.enc(inputs)

        rec_signal = ch_SSPA(codeword, noise_std, self.device)
        dec_signal = self.dec(rec_signal)

        return dec_signal
        
