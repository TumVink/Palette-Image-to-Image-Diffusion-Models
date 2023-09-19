import torch
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os,glob
from PIL import Image

mpl.rcParams['figure.figsize'] = (12, 8)

root_dir = '/mnt/data/BCI/'
split = 'train'
format = 'HE'

img_dir = os.path.join(root_dir, split, format)
files = glob.glob(img_dir + '/*.png')
I = np.asarray(Image.open(files[2]))



img = torch.FloatTensor(I/255)
#plt.imshow(img)
#plt.imsave('original.png', img.numpy())

def input_T(input):
    # [0,1] -> [-1,+1]
    return 2 * input - 1


def output_T(input):
    # [-1,+1] -> [0,1]
    return (input + 1) / 2


def show(ax,input):
    ax.imshow(output_T(input).clip(0, 1))
    #plt.savefig('noisy imgs/foo.png')

def save(f,img_name):
    img_name = str(img_name) + '.png'
    f.savefig(img_name)



img_ = input_T(img)
# show(img_)

num_timesteps=1000
betas=torch.linspace(1e-4,2e-2,num_timesteps)

alphas=1-betas
alphas_sqrt=alphas.sqrt()
alphas_cumprod=torch.cumprod(alphas,0)
alphas_cumprod_sqrt=alphas_cumprod.sqrt()


def forward_step(t, condition_img, return_noise=False):
    """
        forward step: t-1 -> t
    """
    assert t >= 0

    mean = alphas_sqrt[t] * condition_img
    std = betas[t].sqrt()

    # sampling from N
    if not return_noise:
        return mean + std * torch.randn_like(img)
    else:
        noise = torch.randn_like(img)
        return mean + std * noise, noise


def forward_jump(t, condition_img, condition_idx=0, return_noise=False):
    """
        forward jump: 0 -> t
    """
    assert t >= 0

    mean = alphas_cumprod_sqrt[t] * condition_img
    std = (1 - alphas_cumprod[t]).sqrt()

    # sampling from N
    if not return_noise:
        return mean + std * torch.randn_like(img)
    else:
        noise = torch.randn_like(img)
        return mean + std * noise, noise


#plt.figure(figsize=(12, 8))
N=8 # number of computed states between x_0 and x_T
M=1 # number of samples taken from each distribution

for idx in range(N):
    t_step = int(idx * (num_timesteps / N))

    f,ax = plt.subplots(1,1)
    show(ax,alphas_cumprod_sqrt[t_step] * img_)
    # plt.title(r'$\mu_t=\sqrt{\bar{\alpha}_t}x_0$') if idx == 0 else None
    # plt.ylabel("t: {:.2f}".format(t_step / num_timesteps))
    plt.title("Noise Steps: "+str(t_step))
    plt.xticks([])
    plt.yticks([])
    save(f,idx)

    # for sample in range(M):
    #     x_t = forward_jump(t_step, img_)
    #
    #     plt.subplot(N, 1 + M, 2 + (1 + M) * idx + sample)
    #     show(x_t)
    #     plt.axis('off')

#plt.tight_layout()