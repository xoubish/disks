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
import scipy

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
#goodsfits = './data/goodss_all_acs_wfc_f775w_060mas_v1.5_drz.fits'
#goodscat = './data/gds.fits'
goodsfits_b = '/Users/shemmati/Desktop/GOODS/goodss_all_acs_wfc_f435w_060mas_v1.5_drz.fits'
goodsfits_v = '/Users/shemmati/Desktop/GOODS/goodss_all_acs_wfc_f606w_060mas_v1.5_drz.fits'
goodsfits_i = '/Users/shemmati/Desktop/GOODS/goodss_all_acs_wfc_f775w_060mas_v1.5_drz.fits'
goodsfits_z1 = '/Users/shemmati/Desktop/GOODS/goodss_all_acs_wfc_f814w_060mas_v1.5_drz.fits'
goodsfits_z = '/Users/shemmati/Desktop/GOODS/goodss_all_acs_wfc_f850l_060mas_v1.5_drz.fits'
goodsfits_j = '/Users/shemmati/Desktop/GOODS/goodss_all_wfc3_ir_f125w_060mas_v1.0_drz.fits'
goodsfits_h = '/Users/shemmati/Desktop/GOODS/goodss_all_wfc3_ir_f160w_060mas_v1.0_drz.fits'

goodsfits = [goodsfits_b,goodsfits_v,goodsfits_i,goodsfits_z1,goodsfits_z,goodsfits_j,goodsfits_h]
goodscat='/Users/shemmati/Dropbox/WFIRST_WPS/CANDELS_fits/gds.fits'

psfhigh_b = '../psfs/psf_b.fits'
psfhigh_v = '../psfs/psf_v.fits'
psfhigh_i = '../psfs/psf_i.fits'
psfhigh_z1 = '../psfs/psf_z.fits'
psfhigh_z = '../psfs/psf_z.fits'
psfhigh_j = '../psfs/psf_j.fits'
psfhigh_h = '../psfs/psf_h.fits'

psflow = '../psfs/PSF_subaru_i.fits'


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


def distance(x1,x2,y1,y2):
    return (np.sqrt((x1-x2)**2+(y1-y2)**2))

def magdis(f1,f2):
    r =np.ones_like(f1)
    sel1 = (f1>f2)
    r[sel1] = f1[sel1]/f2[sel1]
    sel2 = (f2>=f1)
    r[sel2]=f2[sel2]/f1[sel2]
    return r  

def galblend(gals=1, lim_hmag=25, plot_it=True,goodscat=goodscat, goodsfits = goodsfits):
    
    '''This is to put together two candels GOODS_S galaxies into a single 64x64 cutout.
    I make sure in each cutout the central galaxy is brightest object in the cutout so 
    in rescaling and adding two components still show up. Also, one galaxy is put at 
    the center and the second in some random distance from it. both cutouts are rotated with
    a random angle. Cutouts are from HST H band for now.'''
    
    ## reading GOODS-S catalog and initial selection on objects
    gs = pyfits.getdata(goodscat)
    sel1 = (gs['zbest']>0.1)&(gs['zbest']<5.0)&(gs['CLASS_STAR']<0.95)&(gs['Hmag']<lim_hmag)&(gs['FWHM_IMAGE']>1)&(gs['FWHM_IMAGE']<20) &(gs['DECdeg']>-27.8)
    
   
    ra, dec,red,iflux,fwhm = gs['RA_1'][sel1],gs['DEC_1'][sel1],gs['zbest'][sel1],gs['ACS_F775W_FLUX'][sel1],gs['FWHM_IMAGE'][sel1]
    z2,flux2,x2,y2 = np.zeros(gals),np.zeros(gals),np.zeros(gals),np.zeros(gals)
    s2 = np.zeros(gals)
    im = np.zeros([7,64,64])
 
    data1 = np.zeros([7,80,80])
    da1 = np.zeros([7,64,64])
    
    while not(brightest_center(data1[2,:,:])):
        n = np.int(np.random.uniform(0,len(ra)-1))
        data1[2,:,:] = cut(ra[n],dec[n],40,goodsfits[2])
    
    angle = np.random.uniform(0,180)
    for w in range(7):
        data1[w,:,:] = cut(ra[n],dec[n],40,goodsfits[w])
        s = ndimage.rotate(data1[w,:,:],angle,mode='nearest',reshape=False)
        da1 = s[8:-8,8:-8]
        im[w,:,:] += da1

    y2[0],x2[0] = np.unravel_index(da1.argmax(), da1.shape)
    z2[0] = red[n]
    flux2[0] = iflux[n] 
    s2[0] = fwhm[n]
    
        
    for boz in range(gals-1):
        data2 = np.zeros([7,140,140])
        
        while not(brightest_center(data2[2,:,:])):
            n = np.int(np.random.uniform(0,len(ra)-1))
            data2[2,:,:] = cut(ra[n],dec[n],70,goodsfits[2])
        
        p,t = np.int(np.random.randint(0,15)),np.int(np.random.randint(0,15))
        angle = np.random.uniform(0,180)
        for w in range(7):
            data2[w,:,:] = cut(ra[n],dec[n],70,goodsfits[w])
            s = data2[w,30+t:-30+t,30+p:-30+p]
            so = ndimage.rotate(s,angle,mode='nearest',reshape=False)
            da2 = so[8:-8,8:-8]
            im[w,:,:] += da2
        
        z2[boz+1],flux2[boz+1],s2[boz+1] = red[n],iflux[n],fwhm[n]
        y2[boz+1],x2[boz+1] = np.unravel_index(da2.argmax(), da2.shape)

    if plot_it:
        plt.figure(figsize=(15,3))
        for w in range(7):
            plt.subplot(1,7,w+1)
            plt.imshow(im[w,:,:],origin='lower')
            plt.plot(x2,y2,'rx')
            plt.axis('off')
            
    return im,[x2,y2,z2,flux2,s2]
