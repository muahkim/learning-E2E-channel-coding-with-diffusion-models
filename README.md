# learning-E2E-channel-coding-with-diffusion-models
This repository contains the simulations used in the paper having the same title as the repository name.
It is open to the public, so please feel free to play around or use it for your research, study, education, etc.
In the repository, there are notebook files enabling an end-to-end (E2E) framework for each of three channel models: AWGN channel, real Rayleigh fading channel, and a channel model with solid-state power amplifier (SSPA). 
The E2E framework's final goal is to learn a pair of a neural encoder and a neural decoder achieving a small symbol error rate.
In the real world, the channel models are not clearly known nor differentiable, 
so we learn the channel distribution by using a diffusion model and replace the channel block with a synthesized channel sampled by the learned diffusion model.
Please refer to our papers for more details. You can find the links of them below. 
This repository has the source code that can reproduce the simulations in the paper and some visualizations that can help understanding.


## Implementation Environment
Python >= 3.7

PyTorch >= 1.6

CUDA (I don't know exactly, but I used) 11.6

## Parameters and Options Available
The source codes are in the .ipynb format and are straightforward. You can easily find the parameters and simulation settings in the top cells and change them as you want. The adjustable parameters and available options are listed below.  

### Communication Channel
* Channel models: AWGN, Rayleigh, SSPA
* Parameters: cardinality of the message set 'M', block length 'n', Training Eb/N0 'TRAINING_Eb/N0'.  

### Difusion-Denoising Models
* Prediction variable and loss: 'epsilon', 'v'
* Sampling algorithm: 'DDPM', 'DDIM'
* Diffusion noise (beta) scheduling: sigmoid, cosine, no scheduling (constant). 
* Parameters: # of diffusion steps 'num_steps', step size of skipped sampling 'skip'.

## Sources
Our papers about this study: [conference version](https://scholar.google.com/citations?view_op=view_citation&hl=ko&user=VOl55dwAAAAJ&citation_for_view=VOl55dwAAAAJ:IjCSPb-OGe4C), [full journal version](https:) 

[azad-academy/denoising-diffusion-model](https://github.com/azad-academy/denoising-diffusion-model): We referred to the simple implementation of the diffusion models with MLPs.


