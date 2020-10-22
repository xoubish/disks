from __future__ import print_function
import argparse
import os
import random
from skimage.transform import downscale_local_mean
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np

import astropy.wcs as wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import ndimage
from PIL import Image
from photutils import create_matching_kernel
from skimage.transform import downscale_local_mean
from astropy.convolution import convolve

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F


from astropy.convolution import convolve

from skimage.feature import blob_dog, blob_log, blob_doh

from peak_finding import (
    PeakFinder,
    find_peaks,
)

import cv2



device = torch.device("cpu")

batchSize = 64          # input batch size
imageSize = 64           # the height / width of the input image to network
ngf = 64
ndf = 64
nz = 1000
ngpu = 0    #number of GPUs to use
netG = ''
manualSeed = random.randint(1, 10000)
torch.manual_seed(manualSeed)
nc = 1
goodsfits = './data/goodss_all_acs_wfc_f775w_060mas_v1.5_drz.fits'
goodscat = './data/gds.fits'
#goodsfits = '/Users/shemmati/Desktop/GOODS/goodss_all_acs_wfc_f775w_060mas_v1.5_drz.fits'
#goodscat='/Users/shemmati/Dropbox/WFIRST_WPS/CANDELS_fits/gds.fits'
psfhigh='../psfs/psf_i.fits'
psflow='../psfs/PSF_subaru_i.fits'

class Shoobygen(nn.Module):

    def __init__(self,ngpu):
        super(Shoobygen, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            
            nn.Conv2d(1, ngf * 4, 7, 2, 2, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            
            nn.ConvTranspose2d( ngf*4, ngf * 2, 6, 3, 2,dilation=2, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, nc, 3, 2, 2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            output1 = output[:,:,:,:]
        else:
            output = self.main(input)
            output1 = output[:,:,:,:]

        return output1
netS = Shoobygen(ngpu).to(device)
netS.load_state_dict(torch.load('netG_epoch_599.pth',map_location='cpu'))

def radec2xy(ra,dec,wc):
    coords = SkyCoord(ra,dec, unit='deg')
    a=wcs.utils.skycoord_to_pixel(coords, wc, origin=0,mode=u'wcs')
    return a[0],a[1]
    
def cut(ra,dec,andaze,filename):
    '''gets coordinates of the galaxy and the filter to return a cutout
    (also called a postage stamp) of the galaxy with given size'''
    hdr = pyfits.getheader(filename)#'/Users/shemmati/Desktop/GOODS/goodsn_all_wfc3_ir_f160w_060mas_v1.0_drz.fits')#
    w = wcs.WCS(hdr)
    x,y=radec2xy(ra,dec,w)
    x,y=np.int(x),np.int(y)
    im=pyfits.getdata(filename)[y-andaze:y+andaze,x-andaze:x+andaze]
    return im

def brightest_center(im, r = 20):
    
    '''This function is to check whether the central object of the 
    image is the brightest compared to its neighbors in the given cutout.
    Central is defined with a 10x10 pixel square in the center'''
    
    a0,a1 = np.unravel_index(np.argmax(im, axis=None), im.shape)
    ans = False
    if ((a0>((im.shape[0]-r)/2)) & (a0<((im.shape[0]+r)/2)) & (a1>((im.shape[1]-r)/2)) & (a1<((im.shape[0]+r)/2))):
        ans = True
    
    return ans

import scipy
def go_lowres(galax,out_size=21, noise_sigma=0.05,psfhigh=psfhigh,psflow=psflow):
    '''This function is to take high resolution galaxy cutout and go to 
    a lower pixelscale, resolution and more noise'''
    
    psf = pyfits.getdata(psfhigh)
    psf = downscale_local_mean(psf,(3,3))
    psf = psf[7:-8,7:-8]
    psf_hsc = pyfits.getdata(psflow)
    psf_hsc = psf_hsc[1:42,1:42]
    kern = create_matching_kernel(psf,psf_hsc)

    img = convolve(galax,kern)

    outp = np.array(Image.fromarray(img).resize((out_size,out_size)))
    outp = outp*9.0

    im = outp+np.random.normal(0,noise_sigma,outp.shape)    
    return im


def go_lowres_tens(galax,psfhigh=psfhigh,psflow=psflow):
    device = torch.device("cpu")
    tfms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    im = tfms(galax)
    gals = np.repeat(im[:,:,:,np.newaxis],1,axis = 3)
    gals =torch.tensor(gals, device="cpu").float()
    gals = gals.permute(3,0,1,2)
    inputs = gals
    real_cpu = inputs.to(device)

    psf = pyfits.getdata(psfhigh)
    psf = downscale_local_mean(psf,(3,3))
    psf = psf[7:-8,7:-8]#[22:-22,22:-22]
    psf_hsc = pyfits.getdata(psflow)
    psf_hsc = psf_hsc[1:42,1:42]
    kern = create_matching_kernel(psf,psf_hsc)
    psfh = np.repeat(kern[:,:, np.newaxis], 1, axis=2)
    psfh = np.repeat(psfh[:,:,:,np.newaxis],1,axis = 3)
    kernel = torch.Tensor(psfh)
    kernel = kernel.permute(2,3,0,1)
    kernel =  kernel.float()
    kernel = kernel.to(device)
    im = real_cpu+0.25*torch.rand_like(real_cpu)
    downsampled = F.upsample(im,scale_factor=1/3,mode='bilinear')
    img = F.conv2d(downsampled, kernel,padding=int(((kernel.shape[3])-1)/2))
    return img[:,:,:,:]

def str2coord(string):
    x,y=[],[]
    if len(string)>8:
        d = string.split('[')
        d = ''.join(d)
        d2 = d.split(']')
        d2 = ''.join(d2)
        d3 = d2.split(',')
        for i in range(np.int(len(d3)/2)):
            x.append(np.int(d3[i]))
            y.append(np.int(d3[np.int(len(d3)/2)+i]))
    return x,y

def MatchGan(x,y,x2,y2):
    num=0
    if len(x2)>0:
        for i in range(len(x)):
            dis = distance(x2,np.repeat(y[i],len(x2)),y2,np.repeat(x[i],len(y2))) #distance of all Ganres to initial sources
            if np.min(dis)<5:
                num+=1
    num = min(num,len(x2))
    return num

def MatchLow(x,y,x2,y2):
    num=0
    if len(x2)>0:
        xlo = np.array(x2)*3.0
        ylo = np.array(y2)*3.0
        for i in range(len(x)):
            dis = distance(xlo,np.repeat(y[i],len(xlo)),ylo,np.repeat(x[i],len(ylo))) #distance of all Ganres to initial sources
            if np.min(dis)<10:
                num+=1
            dis = []
    num = min(num,len(x2))
    return num

def distance(x1,x2,y1,y2):
    return (np.sqrt((x1-x2)**2+(y1-y2)**2))

def magdis(f1,f2):
    r =np.ones_like(f1)
    sel1 = (f1>f2)
    r[sel1] = f1[sel1]/f2[sel1]
    sel2 = (f2>=f1)
    r[sel2]=f2[sel2]/f1[sel2]
    return r  

def galblend(gals=1, lim_hmag=24, plot_it=True,goodscat=goodscat,goodsfits=goodsfits,psfhigh=psfhigh,psflow=psflow):
    
    '''This is to put together two candels GOODS_S galaxies into a single 64x64 cutout.
    I make sure in each cutout the central galaxy is brightest object in the cutout so 
    in rescaling and adding two components still show up. Also, one galaxy is put at 
    the center and the second in some random distance from it. both cutouts are rotated with
    a random angle. Cutouts are from HST H band for now.'''
    
    ## reading GOODS-S catalog and initial selection on objects
    gs = pyfits.getdata(goodscat)
    sel1 = (gs['zbest']>0.1)&(gs['zbest']<5.0)&(gs['CLASS_STAR']<0.95)&(gs['Hmag']<lim_hmag)&(gs['FWHM_IMAGE']>2)&(gs['FWHM_IMAGE']<10) &(gs['DECdeg']>-27.8)
    
   
    ra, dec,red,iflux,fwhm = gs['RA_1'][sel1],gs['DEC_1'][sel1],gs['zbest'][sel1],gs['ACS_F775W_FLUX'][sel1],gs['FWHM_IMAGE'][sel1]
    z2,flux2,x2,y2 = np.zeros(gals),np.zeros(gals),np.zeros(gals),np.zeros(gals)
    s2 = np.zeros(gals)
    im = np.zeros([64,64])
   
    data1 = np.zeros([80,80])
    da1 = np.zeros([64,64])
    while not(brightest_center(data1)):
        n = np.int(np.random.uniform(0,len(ra)-1))
        data1 = cut(ra[n],dec[n],40,goodsfits)
        z2[0] = red[n]
        flux2[0] = iflux[n] 
        s2[0] = fwhm[n]
    
    angle = np.random.uniform(0,180)
    s = ndimage.rotate(data1,angle,mode='nearest',reshape=False)
    da1 = s[8:-8,8:-8]
    y2[0],x2[0] = np.unravel_index(da1.argmax(), da1.shape)
    im += da1
    
    da2 = np.zeros([gals-1,64,64])
    
    for boz in range(gals-1):
        data2 = np.zeros([140,140])
        
        while not(brightest_center(data2)):
            n = np.int(np.random.uniform(0,len(ra)-1))
            data2 = cut(ra[n],dec[n],70,goodsfits)
            
            z2[boz+1],flux2[boz+1],s2[boz+1] = red[n],iflux[n],fwhm[n]
        #p,t = np.int(np.random.uniform(5,10)),np.int(np.random.uniform(5,10))
        p,t = np.int(np.random.randint(0,15)),np.int(np.random.randint(0,15))
        #s = data2[10+t:-10+t,10+p:-10+p]
        s = data2[30+t:-30+t,30+p:-30+p]
        
        angle = np.random.uniform(0,180)
        so = ndimage.rotate(s,angle,mode='nearest',reshape=False)
        da2[boz] = so[8:-8,8:-8]
        y2[boz+1],x2[boz+1] = np.unravel_index(da2[boz].argmax(), da2[boz].shape)
        im += da2[boz]
    
        
    dada = np.arcsinh(im)
    rescaled = (255.0 / (dada.max()+0.05) * (dada - dada.min())).astype(np.uint8)
    final_im = Image.fromarray(rescaled)
        
     #### Detect sources on high res blend sample
    
    psf = pyfits.getdata(psfhigh)
    #num = find_peaks(image=im, kernel = psf,thresh=3*np.mean(im))
    num = find_peaks(image=rescaled-np.mean(rescaled), kernel = psf,thresh=np.mean(rescaled))
    x_esh,y_esh = [],[]
    for boz in range(len(num)):
        if (1<num[boz][0]<64)&(1<num[boz][1]<64):
            x_esh.append(num[boz][0])
            y_esh.append(num[boz][1])
      
    ### Reduce resolution and pixel scale to Subaru and add some noise
    lowres = go_lowres(im)
    dadalow = np.arcsinh(lowres)
    rescaledlow = (255.0 / (dadalow.max()+1) * (dadalow - dadalow.min())).astype(np.uint8)

    psflo = pyfits.getdata(psflow)
    num = find_peaks(image=lowres, kernel = psflo,thresh=3*np.mean(lowres))
    x_esh_low,y_esh_low = [],[]
    for boz in range(len(num)):
        if (1<num[boz][0]<21)&(1<num[boz][1]<21):
            x_esh_low.append(num[boz][0])
            y_esh_low.append(num[boz][1])


    #### Increase resolution with the trained GAN       
    lowres_tensor = go_lowres_tens(rescaled)       
    fake = netS(lowres_tensor)
    fd = fake.detach()
    fd = fd.cpu()
    GANres = fd[0,0,:,:].numpy()
    
    ### Detect sources on GANres
    num = find_peaks(image=GANres-np.mean(GANres), kernel = psf,thresh=0.2)#np.mean(GANres))
    x_esh_fd,y_esh_fd = [],[]
    for boz in range(len(num)):
        if (1<num[boz][0]<64)&(1<num[boz][1]<64):
            x_esh_fd.append(num[boz][0])
            y_esh_fd.append(num[boz][1])

    
    if plot_it:
        plt.figure(figsize=(12,4))
        n = gals+1
        
        plt.subplot(1,n+2,1)
        plt.imshow(da1,origin='lower')
        plt.scatter(x2[0],y2[0],marker='x',color='r')
        plt.text(2,55,'z='+str(z2[0]),color='y',fontsize=20)
        plt.axis('off')

        for boz in range(gals-1):
            plt.subplot(1,n+2,2+boz)
            plt.imshow(da2[boz],origin='lower')
            plt.text(2,55,' z='+str(z2[boz+1]),color='y',fontsize=20)
            plt.scatter(x2[boz+1],y2[boz+1],marker='x',color='r')
            plt.axis('off')

        plt.subplot(1,n+2,n)
        plt.imshow(rescaled,origin='lower')
        plt.text(2,55,'Sum',color='y',fontsize=20)
        plt.plot(y_esh,x_esh,'ro')
        plt.axis('off')

        plt.subplot(1,n+2,n+1)
        plt.imshow(rescaledlow,origin='lower')
        plt.text(1.5,18,'Lowres',color='y',fontsize=20)
        plt.plot(y_esh_low,x_esh_low,'ro')
        plt.axis('off')

        plt.subplot(1,n+2,n+2)
        plt.imshow(fd[0,0,:,:],origin='lower')
        plt.text(2,55,'GAN res',color='y',fontsize=20)
        plt.plot(y_esh_fd,x_esh_fd,'ro')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        
    return im,da1,da2[0,:,:],lowres,fd[0,0,:,:],psf,psflo,[x2,y2,z2,flux2,s2,[[x_esh,y_esh],[x_esh_low,y_esh_low],[x_esh_fd,y_esh_fd]]]
