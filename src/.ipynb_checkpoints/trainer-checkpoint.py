import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils 


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 500
    M=16
    learning_rate = 1e-3
    rate = 4/7
    num_workers = 0 # for DataLoader


    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, config, one_hot):
        self.model = model
        self.train_dataset = train_dataset
        #self.test_dataset = test_dataset
        self.config = config
        self.one_hot = one_hot

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
    
    def train(self):
        model, config = self.model, self.config
        optimizer = torch.optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

        loss_CE = nn.CrossEntropyLoss()
        
        def run_epoch():
            lr = config.learning_rate
            batch_size = config.batch_size
            M = config.M
            data = self.train_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=False, batch_size=batch_size,
                                num_workers=config.num_workers)            
            losses, batch_BER, batch_SER  = [], [], []
            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)
            
            for it, x in pbar:
                labels = x.reshape((batch_size,1)).to(self.device)
                x = F.one_hot(labels.long(), num_classes=M)                    
                x = x.float().reshape(batch_size,M)

                optimizer.zero_grad()            
                output = model(x, config.noise_std)
                if self.one_hot:
                    loss = loss_CE(output, labels.long().squeeze(1))
                else:
                    loss =  F.binary_cross_entropy(output, x)
                loss = loss.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                losses.append(loss.item())
                optimizer.step()
                
                if self.one_hot:
                    SER, _, _ = utils.SER(labels.long(), output, 0.9)
                    batch_SER.append(SER.item())
                    if it%100==0:
                        pbar.set_description(f"epoch {epoch+1}: loss {np.mean(losses):.2e} SER {np.mean(batch_SER):.2e}")
                else:
                    batch_BER.append(utils.B_Ber_m(x, output).item())
                    pbar.set_description(f"epoch {epoch+1}: loss {np.mean(losses):.2e} BER {np.mean(batch_BER):.2e}")          
               
            
        for epoch in range(config.max_epochs):
            run_epoch()
    
    def test(self, snr_range, one_hot, erasure_bound):
        model, config = self.model, self.config
        ser = []
                
        def run_epoch():
            data = self.train_dataset
            batch_size = config.batch_size
            M = config.M
            loader = DataLoader(data, shuffle=True, pin_memory=False, batch_size=batch_size,
                                num_workers=config.num_workers)            
            batch_BER = [] 
            block_ER = []
            erasures = []
            batch_SER = []
            
            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)
            
            for it, x in pbar:
                if self.one_hot:
                    labels = x.reshape((batch_size,1)).to(self.device)
                    x=F.one_hot(labels.long(), num_classes=M)
                    x = x.float().reshape(batch_size,M)
                else:
                    x = x.float().to(self.device)
                noise_std_ = utils.EbNo_to_noise(snr_range[epoch], config.rate)
                #noise_std_ = utils.SNR_to_noise(snr_range[epoch])
              
                output = model(x,  noise_std_)

                if self.one_hot:
                    SER, _, _ = utils.SER(labels.long(), output, erasure_bound)
                    batch_SER.append(SER.item())
                    #block_ER.append(bler)
                    #erasures.append(Erasures)
                    if it%100==0:
                        pbar.set_description(f"SNR {snr_range[epoch]}  iter {it}: SER {np.mean(batch_SER):.3e}")
                else:
                    batch_BER.append(utils.B_Ber_m(x.float(), output).item())
                    block_ER.append(utils.Block_ER(x.float(), output))
                    pbar.set_description(f"SNR {snr_range[epoch]} iter {it}: avg. BER {np.mean(batch_BER):.8f} BLER {np.mean(block_ER):.7e}")
       
            return np.mean(batch_SER)

        num_samples = self.train_dataset.size(dim=0)
        for epoch in range(len(snr_range)):

            temp1 = 0
            it_in_while = 0

            while temp1 * num_samples *(it_in_while) < 1000: # To guarantee we have enough samples for the Monte-Carlo method.
                if it_in_while > 1 :
                    print("The number of samples is not enough. One more epoch is running. Total # of samples used: " , num_samples * it_in_while)
                temp2 = run_epoch()
                temp1 = (it_in_while * temp1 + temp2)/(it_in_while+1) # taking average of the error probability.
                it_in_while += 1
            ser.append(temp1)
        
        return ser
    
    
    
class TrainerConfig_DDM:
    # optimization parameters
    max_epochs = 10
    dataset_size = 1000000
    batch_size = 500
    noise_std = 1
    learning_rate = 1e-3
    M=16 
    n=7
    num_steps = 50
    betas = torch.zeros([num_steps])    
    rate = 4/7
    num_workers = 0 # for DataLoader
    optim_gen = optim.Adam
#    TRANSFER_LEARNING= False
    IS_RES = False
    pred_type = 'epsilon'
    

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.alphas = 1 - self.betas.to('cpu')
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_prod_p = torch.cat([torch.tensor([1]), self.alphas_prod[:-1]], 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        #one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

class Trainer_DDM:
    def __init__(self, channel_gen, ema, device, channel_model, tconf_gen):
        self.channel_gen = channel_gen
        self.ema = ema
        self.device = device
        self.channel_model = channel_model
        self.tconf_gen = tconf_gen
        
    def train_PreT(self): # pretraining algorithm
        channel_gen = self.channel_gen
        ema = self.ema
        device = self.device
        channel_model = self.channel_model
        tconf_gen = self.tconf_gen
        n = tconf_gen.n

        IS_RES = tconf_gen.IS_RES
        
        optimizer_ch = tconf_gen.optim_gen(channel_gen.parameters(), lr=tconf_gen.learning_rate)
        dataset_gen = torch.randn(tconf_gen.dataset_size, n)
        alphas_bar_sqrt = tconf_gen.alphas_bar_sqrt.to(device)
        one_minus_alphas_bar_sqrt = tconf_gen.one_minus_alphas_bar_sqrt.to(device)

        if tconf_gen.pred_type == 'epsilon':
            estimation_loss = utils.noise_estimation_loss
        elif tconf_gen.pred_type == 'v':
            estimation_loss = utils.v_estimation_loss
        else:
            assert(ValueError)


        loss_ep_gen = []
        for t in range(tconf_gen.max_epochs):
            loader = DataLoader(dataset_gen, shuffle=True, pin_memory=False, batch_size=tconf_gen.batch_size) 
            loss_batch, batch_BER, batch_SER  = [], [], []
            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5) 

            for it, x in pbar: 
                x = x.to(device)
                y = channel_model(x, tconf_gen.noise_std, device) 
                if IS_RES:
                    y = y - x
                    
                yx = torch.cat((y , x),dim=1)
                # Create EMA model
                loss = estimation_loss(channel_gen, yx, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, tconf_gen.num_steps, n)

                optimizer_ch.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(channel_gen.parameters(), 1.)
                optimizer_ch.step()
                # Update the exponential moving average
                ema.update(channel_gen)

                loss_batch.append(loss.item())

                if it % 100 == 0:
                    pbar.set_description(f"epoch {t+1}: loss {np.mean(loss.item()):.2e}")
          
            loss_ep_gen.append(np.mean(loss_batch))
        return loss_ep_gen
 
    def train_itr(self, encoder): # iterative training
        
        channel_gen = self.channel_gen
        ema = self.ema
        device = self.device
        channel_model = self.channel_model
        tconf_gen = self.tconf_gen
        M = tconf_gen.M
        n = tconf_gen.n

        optimizer_ch = tconf_gen.optim_gen(channel_gen.parameters(), lr=tconf_gen.learning_rate)
            
        dataset_gen = np.random.randint(M, size=tconf_gen.dataset_size) 
        
        alphas_bar_sqrt = tconf_gen.alphas_bar_sqrt.to(device)
        one_minus_alphas_bar_sqrt = tconf_gen.one_minus_alphas_bar_sqrt.to(device)

        if tconf_gen.pred_type == 'epsilon':
            estimation_loss = utils.noise_estimation_loss
        elif tconf_gen.pred_type == 'v':
            estimation_loss = utils.v_estimation_loss
        else:
            assert(ValueError)

        loss_ep_gen = []
        for t in range(tconf_gen.max_epochs):
            loader = DataLoader(dataset_gen, shuffle=True, pin_memory=False, batch_size=tconf_gen.batch_size) 

            loss_batch, batch_BER, batch_SER  = [], [], []
            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5) 

            for it, m in pbar: 
                labels = m.reshape((-1,1))
                m = F.one_hot(labels.long(), num_classes=M)                    
                m = m.float().reshape(-1,M).to(device)

                x = encoder(m) 
                y = channel_model(x, tconf_gen.noise_std, device) 
                yx = torch.cat((y , x ),dim=1)
                
                # Create EMA model
                loss = estimation_loss(channel_gen, yx, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, tconf_gen.num_steps, n)

                optimizer_ch.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(channel_gen.parameters(), 1.)
                optimizer_ch.step()
                
                # Update the exponential moving average
                ema.update(channel_gen)

                loss_batch.append(loss.item())

                if it % 100 == 0:
                    pbar.set_description(f"epoch {t+1}: loss {np.mean(loss.item()):.2e}")
                  
            loss_ep_gen.append(np.mean(loss_batch))
        return loss_ep_gen


class TrainerConfig_AE_w_DDM:
    # optimization parameters
    max_epochs = 10
    dataset_size = 1000000
    batch_size = 500
    noise_std = 1
    learning_rate = 1e-3
    M=16 
    n=7
    rate = 4/7
    num_steps = 50
    #alphas = torch.zeros([num_steps])
    betas = torch.zeros([num_steps])
    denoising_alg = 'DDPM'
    pred_type = 'epsilon'
    traj = range(num_steps)
    channel_model = 'None'

    optim_AE = optim.NAdam
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.alphas = 1 - self.betas.to('cpu')
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_prod_p = torch.cat([torch.tensor([1]), self.alphas_prod[:-1]], 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        #one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
        
class Trainer_AE_w_DDM:
    def __init__(self, encoder, decoder, channel_gen, device, tconf_AE):
        self.encoder = encoder
        self.decoder = decoder
        self.channel_gen = channel_gen
        self.device = device
        self.tconf_AE = tconf_AE
        
    def train_PreT(self):#(encoder, decoder, channel_gen, device, tconf_AE):    
        encoder = self.encoder
        decoder = self.decoder 
        channel_gen = self.channel_gen 
        device = self.device 
        tconf_AE = self.tconf_AE
        
        
        M = tconf_AE.M
        n = tconf_AE.n
    
        denoising_alg = tconf_AE.denoising_alg
        one_minus_alphas_bar_sqrt = tconf_AE.one_minus_alphas_bar_sqrt.to(device)
        betas = tconf_AE.betas.to(device)
        alphas = tconf_AE.alphas.to(device)
        
        dataset_AE = np.random.randint(M, size=tconf_AE.dataset_size)
        loss_CE = nn.CrossEntropyLoss()
        optimizer_AE = tconf_AE.optim_AE(list(encoder.parameters()) + list(decoder.parameters()), lr = tconf_AE.learning_rate)

        loss_ep_AE = []

        for t in range(tconf_AE.max_epochs):
            loader = DataLoader(dataset_AE, shuffle=True, pin_memory=False, batch_size=tconf_AE.batch_size)#, num_workers=config.num_workers)            
            loss_batch, batch_BER, batch_SER  = [], [], []
            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

            for it, m in pbar: #enumerate(loader): 
                labels = m.reshape((-1,1)).to(device)
                optimizer_AE.zero_grad()

            # Forward Pass:
                # Encoding
                m = F.one_hot(labels.long(), num_classes=M)                    
                m = m.float().reshape(-1,M)
                x = encoder(m)
                c = x 
                
                # Channel
                if denoising_alg == 'DDPM':
                    x_seq = utils.p_sample_loop_w_Condition(channel_gen, x.size(), tconf_AE.num_steps,
                                                            tconf_AE.alphas, tconf_AE.betas,
                                                             tconf_AE.alphas_bar_sqrt,
                                                            tconf_AE.one_minus_alphas_bar_sqrt, c,
                                                            pred_type=tconf_AE.pred_type)
                elif denoising_alg == 'DDIM':
                    x_seq = utils.p_sample_loop_w_Condition_DDIM(channel_gen, x.size(), tconf_AE.traj,
                                                                 tconf_AE.alphas_prod, tconf_AE.alphas_bar_sqrt,
                                                                 tconf_AE.one_minus_alphas_bar_sqrt, c,
                                                                pred_type=tconf_AE.pred_type)
                y = x_seq[-1]

                # Decoding
                m_d = decoder(y)

            # Backward pass: 
                # Compute the loss and the gradient
                loss = loss_CE(m_d, labels.long().squeeze(1))
                loss = loss.mean()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(decoder.parameters()), 1.)
                optimizer_AE.step()


                loss_batch.append(loss.item())

                SER_tmp, _, _ = utils.SER(labels.long(), m_d, 0.9)
                batch_SER.append(SER_tmp.item())

                if it%100==0:
                    pbar.set_description(f"epoch {t+1}: loss {np.mean(loss_batch):.2e} SER {np.mean(batch_SER):.2e}")        

            # Save epoch loss
            loss_ep_AE.append(np.mean(loss_batch))
        return loss_ep_AE

   

    def train_PreT_enc(self):  # (encoder, decoder, channel_gen, device, tconf_AE):
        encoder = self.encoder
        decoder = self.decoder
        channel_gen = self.channel_gen
        device = self.device
        tconf_AE = self.tconf_AE

        M = tconf_AE.M
        n = tconf_AE.n

        denoising_alg = tconf_AE.denoising_alg
        one_minus_alphas_bar_sqrt = tconf_AE.one_minus_alphas_bar_sqrt.to(device)
        betas = tconf_AE.betas.to(device)
        alphas = tconf_AE.alphas.to(device)

        dataset_AE = np.random.randint(M, size=tconf_AE.dataset_size)
        loss_CE = nn.CrossEntropyLoss()
        optimizer_AE = tconf_AE.optim_AE(encoder.parameters(), lr=tconf_AE.learning_rate)

        loss_ep_AE = []
        SER_ep = []

        for t in range(tconf_AE.max_epochs):
            loader = DataLoader(dataset_AE, shuffle=True, pin_memory=False, batch_size=tconf_AE.batch_size)
            loss_batch, batch_BER, batch_SER = [], [], []
            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

            for it, m in pbar:
                labels = m.reshape((-1, 1)).to(device)
                optimizer_AE.zero_grad()

                # Forward Pass:
                # Encoding
                m = F.one_hot(labels.long(), num_classes=M)
                m = m.float().reshape(-1, M)
                x = encoder(m)
                c = x

                # Channel

                if denoising_alg == 'DDPM':
                    x_seq = utils.p_sample_loop_w_Condition(channel_gen, x.size(), tconf_AE.num_steps
                                                            , tconf_AE.alphas, tconf_AE.betas,
                                                            tconf_AE.alphas_bar_sqrt,
                                                            tconf_AE.one_minus_alphas_bar_sqrt, c,
                                                            pred_type=tconf_AE.pred_type)
                    y = x_seq[-1]
                elif denoising_alg == 'DDIM':
                    x_seq = utils.p_sample_loop_w_Condition_DDIM(channel_gen, x.size(), tconf_AE.traj,
                                                                 tconf_AE.alphas_prod, tconf_AE.alphas_bar_sqrt,
                                                                 tconf_AE.one_minus_alphas_bar_sqrt, c,
                                                                 pred_type=tconf_AE.pred_type)
                    y = x_seq[-1]

                    # Decoding
                m_d = decoder(y)

                # Backward pass:
                # Compute the loss and the gradient
                loss = loss_CE(m_d, labels.long().squeeze(1))
                loss = loss.mean()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.)
                optimizer_AE.step()

                loss_batch.append(loss.item())

                SER_tmp, _, _ = utils.SER(labels.long(), m_d, 0.9)
                batch_SER.append(SER_tmp.item())

                if it % 100 == 0:
                    pbar.set_description(f"loss {np.mean(loss_batch):.2e} SER {np.mean(batch_SER):.2e}")

                    # Save epoch loss
            loss_ep_AE.append(np.mean(loss_batch))
            SER_ep.append(np.mean(batch_SER))
        return loss_ep_AE, SER_ep

    def train_dec(self):
        encoder = self.encoder
        decoder = self.decoder 
        device = self.device 
        tconf_AE = self.tconf_AE
        
        
        M = tconf_AE.M
        n = tconf_AE.n
        
        dataset_AE = np.random.randint(M, size=tconf_AE.dataset_size)
        loss_CE = nn.CrossEntropyLoss()
        optimizer_AE = tconf_AE.optim_AE(decoder.parameters(), lr = tconf_AE.learning_rate)

        loss_ep_AE = []
        SER_ep = []

        for t in range(tconf_AE.max_epochs):
            loader = DataLoader(dataset_AE, shuffle=True, pin_memory=False, batch_size=tconf_AE.batch_size)#, num_workers=config.num_workers)            
            loss_batch, batch_BER, batch_SER  = [], [], []
            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

            for it, m in pbar: #enumerate(loader): 
                labels = m.reshape((-1,1)).to(device)
                optimizer_AE.zero_grad()

            # Forward Pass:
                # Encoding
                m = F.one_hot(labels.long(), num_classes=M)                    
                m = m.float().reshape(-1,M)
                x = encoder(m)
                y = tconf_AE.channel_model(x, tconf_AE.noise_std, device)             
               
                # Decoding
                m_d = decoder(y)

            # Backward pass: 
                # Compute the loss and the gradient
                loss = loss_CE(m_d, labels.long().squeeze(1))
                loss = loss.mean()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.)
                optimizer_AE.step()


                loss_batch.append(loss.item())

                SER_tmp, _, _ = utils.SER(labels.long(), m_d, 0.9)
                batch_SER.append(SER_tmp.item())

                if it%100==0:
                    pbar.set_description(f"loss {np.mean(loss_batch):.2e} SER {np.mean(batch_SER):.2e}")        

            # Save epoch loss
            loss_ep_AE.append(np.mean(loss_batch))
            SER_ep.append(np.mean(batch_SER))
        return loss_ep_AE, SER_ep    

    def train_itr(self):#(encoder, decoder, channel_gen, device, tconf_AE):    
            encoder = self.encoder
            decoder = self.decoder 
            channel_gen = self.channel_gen 
            device = self.device 
            tconf_AE = self.tconf_AE

            M = tconf_AE.M
            n = tconf_AE.n

            one_minus_alphas_bar_sqrt = tconf_AE.one_minus_alphas_bar_sqrt.to(device)
            betas = tconf_AE.betas.to(device)
            alphas = tconf_AE.alphas.to(device)

            dataset_AE = np.random.randint(M, size=tconf_AE.dataset_size)
            loss_CE = nn.CrossEntropyLoss()
            optimizer_AE = tconf_AE.optim_AE(list(encoder.parameters()) + list(decoder.parameters()), lr = tconf_AE.learning_rate)

            loss_ep_AE = []
            SER_ep = []

            for t in range(tconf_AE.max_epochs):
                loader = DataLoader(dataset_AE, shuffle=True, pin_memory=False, batch_size=tconf_AE.batch_size)#, num_workers=config.num_workers)            
                loss_batch, batch_BER, batch_SER  = [], [], []
                pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

                for it, m in pbar: #enumerate(loader): 
                    labels = m.reshape((-1,1)).to(device)
                    optimizer_AE.zero_grad()

                # Forward Pass:
                    # Encoding
                    m = F.one_hot(labels.long(), num_classes=M)                    
                    m = m.float().reshape(-1,M)
                    x = encoder(m)
                    c = x
                    # Channel

                    x_seq = utils.p_sample_loop_w_Condition(channel_gen, [x.size()[0], n], tconf_AE.num_steps, alphas,
                                                            betas, one_minus_alphas_bar_sqrt, c,
                                                            pred_type=tconf_AE.pred_type, is_light=True)
                    y = x_seq # Generated data after 100 time steps

                    # Decoding
                    m_d = decoder(y)

                # Backward pass: 
                    # Compute the loss and the gradient
                    loss = loss_CE(m_d, labels.long().squeeze(1))
                    loss = loss.mean()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(decoder.parameters()), 1.)
                    optimizer_AE.step()


                    loss_batch.append(loss.item())

                    SER_tmp, _, _ = utils.SER(labels.long(), m_d, 0.9)
                    batch_SER.append(SER_tmp.item())

                    if it%100==0:
                        pbar.set_description(f"epoch {t+1}: loss {np.mean(loss_batch):.2e} SER {np.mean(batch_SER):.2e}")        

                # Save epoch loss
                loss_ep_AE.append(np.mean(loss_batch))
                SER_ep.append(np.mean(batch_SER))
            return loss_ep_AE, SER_ep

    def test(self, channel_model):
        encoder = self.encoder
        decoder = self.decoder 
        
        device = self.device 
        tconf_AE = self.tconf_AE
        
        
        M = tconf_AE.M
        n = tconf_AE.n
        noise_std = tconf_AE.noise_std
        
        dataset_AE = np.random.randint(M, size=tconf_AE.dataset_size)

        SER_ep_AE = []
        
        for t in range(1):
            loader = DataLoader(dataset_AE, shuffle=True, pin_memory=False, batch_size=tconf_AE.batch_size)           
            batch_SER  = []
            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=0.5)

            for it, m in pbar: 
                labels = m.reshape((-1,1)).to(device)
            # Forward Pass:
                # Encoding
                m = F.one_hot(labels.long(), num_classes=M)                    
                m = m.float().reshape(-1,M)
                x = encoder(m)
 
                # Channel
                y = channel_model(x, noise_std, device) 

                # Decoding
                m_d = decoder(y)

                SER_tmp, _, _ = utils.SER(labels.long(), m_d, 0.9)
                batch_SER.append(SER_tmp.item())

                if it%100==0:
                    pbar.set_description(f"Validation SER {np.mean(batch_SER):.2e}")        

            # Save epoch loss
            SER_ep_AE.append(np.mean(batch_SER))
        return np.mean(SER_ep_AE)
