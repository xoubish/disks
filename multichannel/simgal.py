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

psfhigh = [psfhigh_b,psfhigh_v,psfhigh_i,psfhigh_z1,psfhigh_z,psfhigh_j,psfhigh_h]
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

def simgal(gals=1, lim_hmag=25, plot_it=True,goodscat=goodscat, goodsfits = goodsfits,psfhigh=psfhigh,psflow=psflow):
    
    '''This is to put together two candels GOODS_S galaxies into a single 64x64 cutout.
    I make sure in each cutout the central galaxy is brightest object in the cutout so 
    in rescaling and adding two components still show up. Also, one galaxy is put at 
    the center and the second in some random distance from it. both cutouts are rotated with
    a random angle. Cutouts are from HST H band for now.'''
    
    ## reading GOODS-S catalog and initial selection on objects
    gs = pyfits.getdata(goodscat)
    sel1 = (gs['zbest']>0.1)&(gs['zbest']<5.0)&(gs['CLASS_STAR']<0.95)&(gs['Hmag']<lim_hmag)&(gs['FWHM_IMAGE']>2)&(gs['FWHM_IMAGE']<15) &(gs['DECdeg']>-27.8)#(gs['DECdeg']>-27.8)
    
   
    ra, dec,red,iflux,fwhm = gs['RA_1'][sel1],gs['DEC_1'][sel1],gs['zbest'][sel1],gs['ACS_F775W_FLUX'][sel1],gs['FWHM_IMAGE'][sel1]

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
        da1[w,:,:] = s[8:-8,8:-8]
        im[w,:,:] += da1[w,:,:]
    
    dada2 = np.zeros([gals,7,64,64])    
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
            dada2[boz,w,:,:] = so[8:-8,8:-8]
            im[w,:,:] += dada2[boz,w,:,:]

    return im[:,:,:]