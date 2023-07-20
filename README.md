# learning-E2E-channel-coding-with-diffusion-models
This repository contains the simulations used in the paper having the same title as the repository name.
It is open to public, so please play around or use for your research, study, education, etc.
In the repository, there are notebook files enabling an end-to-end (E2E) framework for each of three channel models: AWGN channel, real Rayleigh fading channel, and a channel model with solid state power amplifier (SSPA). 
The E2E framwork's goal is to learn a pair of a neural encoder and a neural decoder achieving a small symbol error rate.
In real world, the channel models are not clearly known nor differentiable, 
so we learn the channel distribution by using a diffusion model and replace the channel block by a synthesized channel sampled by the learned diffusion model.
Please refer to our paper for more details.
This repository has the source code that can reproduce the simulations in the paper and some visualizations that can help understanding.


## Implementation Environment
Python >= 3.7

pytorch >= 1.6

CUDA (I don't know exactly, but I used) 11.6

## Parameters and Options Available
### Communication Channel
* Channel models: AWGN, Rayleigh, SSPA
* Prameters: cardinality of message set 'M', block length 'n', Training Eb/N0 'TRAINING_Eb/N0'.  

### Difusion-Denoising Models
* Prediction variable and loss: 'epsilon', 'v'
* Sampling algorithm: 'DDPM', 'DDIM'
* Diffusion noise (beta) scheduling: sigmoid, cosine, no scheduling (constant). 
* Parameters: # of diffusion steps 'num_steps', step size of skipped sampling 'skip'.

## Sources
#### Our paper about this study

Conference version: https://scholar.google.com/citations?view_op=view_citation&hl=ko&user=VOl55dwAAAAJ&citation_for_view=VOl55dwAAAAJ:IjCSPb-OGe4C

Full journal version: 

#### Diffusion model tutorial: https://github.com/azad-academy/denoising-diffusion-model

#### Autoencoder-based communication system: https://github.com/Fritschek/Modular-AE-with-Pytorch

