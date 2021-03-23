import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt

import numpy as np
import SimpleITK as sitk
import os
import numpy as np
from glob import glob
import time
import shutil
from PIL import Image

def extract_amp_spectrum(trg_img):

    fft_trg_np = np.fft.fft2( trg_img, axes=(-2, -1) )
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    return amp_target

def amp_spectrum_swap( amp_local, amp_target, L=0.1 , ratio=0):
    
    a_local = np.fft.fftshift( amp_local, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_target, axes=(-2, -1) )

    _, h, w = a_local.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_local[:,h1:h2,w1:w2] = a_local[:,h1:h2,w1:w2] * ratio + a_trg[:,h1:h2,w1:w2] * (1- ratio)
    a_local = np.fft.ifftshift( a_local, axes=(-2, -1) )
    return a_local

def freq_space_interpolation( local_img, amp_target, L=0 , ratio=0):
    
    local_img_np = local_img 

    # get fft of local sample
    fft_local_np = np.fft.fft2( local_img_np, axes=(-2, -1) )

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_ = amp_spectrum_swap( amp_local, amp_target, L=L , ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp( 1j * pha_local )
    local_in_trg = np.fft.ifft2( fft_local_, axes=(-2, -1) )
    local_in_trg = np.real(local_in_trg)

    return local_in_trg

def draw_image(image):
    plt.imshow(image, cmap='gray')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)

    plt.xticks([])
    plt.yticks([])
    
    return 0

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

im_local = Image.open("demo_samples/fundus_client4.jpg")
im_trg_list = [Image.open("demo_samples/fundus_client1.png"),
         Image.open("demo_samples/fundus_client2.jpg"),
         Image.open("demo_samples/fundus_client3.jpg")]

im_local = im_local.resize( (384,384), Image.BICUBIC )
im_local = np.asarray(im_local, np.float32)
im_local = im_local.transpose((2, 0, 1))

plt.figure(figsize=(18,3))
    
for client_idx,im_trg in enumerate(im_trg_list):
    im_trg = im_trg.resize( (384,384), Image.BICUBIC )
    im_trg = np.asarray(im_trg, np.float32)
    im_trg = im_trg.transpose((2, 0, 1))

    L = 0.003

    # visualize local data, target data, amplitude spectrum of target data
    plt.figure(figsize=(18,3))
    plt.subplot(1,8,1)
    draw_image((im_local / 255).transpose((1, 2, 0)))    
    plt.xlabel("Local Image", fontsize=12)

    plt.subplot(1,8,2)
    draw_image((im_trg / 255).transpose((1, 2, 0)))
    plt.xlabel("Target Image (Client {})".format(client_idx), fontsize=12)
    
    # amplitude spectrum of target data
    amp_target = extract_amp_spectrum(im_trg)
    amp_target_shift = np.fft.fftshift( amp_target, axes=(-2, -1) )
    
    plt.subplot(1,8,3)
    draw_image(np.clip((np.log(amp_target_shift)/ np.max(np.log(amp_target_shift))).transpose((1, 2, 0)), 0, 1))
    plt.xlabel("Target Amp (Client {})".format(client_idx), fontsize=12)
    
    # continuous frequency space interpolation
    for idx, i in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
        plt.subplot(1,8,idx+4)
        local_in_trg = freq_space_interpolation(im_local, amp_target, L=L, ratio=1-i)
        local_in_trg = local_in_trg.transpose((1,2,0))
        draw_image((np.clip(local_in_trg / 255, 0, 1)))
        plt.xlabel("Interpolation Rate: {}".format(i), fontsize=12)
    plt.show()

