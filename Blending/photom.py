import numpy as np
import matplotlib.pyplot as plt
import fitsio
import ngmix
import mof
import os
import scipy.optimize
import ipdb
import priors as pedar
import moflib

def get_image(objid,image_path='./images',bkgd_sub = True):
    filename = f"low_res_{objid}.fits"
    image_file = os.path.join(image_path,filename)
    image = fitsio.read(image_file)
    # no background subtraction drives ngmix CRAZY.
    if bkgd_sub:
        bckgd = np.mean([image[0,:],image[-1,:],image[:,0],image[:,-1]])
        noise = np.std([image[0,:],image[-1,:],image[:,0],image[:,-1]])
        #print(f"noise: {noise:04}")
        image = image - bckgd
    return image

def get_psf(objid,image_path = '../psfs',downsample = True):
    psf_filename = os.path.join(image_path,"PSF_subaru_i.fits")
    psfim = fitsio.read(psf_filename)
    # Downsample the psf by a factor of 3.
    return psfim

def fit_psf(psfObs):
    lm_pars = {'ftol': 1.0e-5,
                'xtol': 1.0e-5}
    psf_boot = ngmix.bootstrap.PSFRunner(psfObs,'turb',1.0,lm_pars=lm_pars)
    psf_boot.go()
    gm_psf = psf_boot.fitter.get_gmix()
    return gm_psf


def make_observation(objid, image_noise = .055,psf_noise = 1e-3, scale = 1.0):
    gal_image = get_image(objid)
    psf_image = get_psf(objid)
    weight_image = np.ones_like(gal_image) + 1./image_noise**2
    psf_weight_image = np.ones_like(psf_image) + 1./psf_noise**2

    jj_im = ngmix.jacobian.DiagonalJacobian(scale=scale,col=0,row=0)
    jj_psf = ngmix.jacobian.DiagonalJacobian(scale=scale/3.,col=(psf_image.shape[0]-1)/2.,row=(psf_image.shape[0]-1)/2.)
    psf_obs = ngmix.Observation(psf_image,weight=psf_weight_image,jacobian=jj_psf)
    psf_gmix = fit_psf(psf_obs)
    psf_obs.set_gmix(psf_gmix)
    gal_obs = ngmix.Observation(gal_image, weight=weight_image,jacobian=jj_im, psf=psf_obs)
    
    return gal_obs

def get_priors_double(entry):

    # prior on ellipticity.  The details don't matter, as long
    # as it regularizes the fit.  This one is from Bernstein & Armstrong 2014
    
    g_sigma = 0.3
    g_prior = ngmix.priors.GPriorBA(g_sigma)
    # 2-d gaussian prior on the center
    # row and column center (relative to the center of the jacobian, which would be zero)
    # and the sigma of the gaussians
        
    # units same as jacobian, probably arcsec
    row1, col1 = entry['x1'], entry['y1']
    row2, col2 = entry['x2'], entry['y2']  
    row_sigma, col_sigma = .3, .3 
    cen_prior1 = ngmix.priors.CenPrior(row1, col1, row_sigma, col_sigma)
    cen_prior2 = ngmix.priors.CenPrior(row2, col2, row_sigma, col_sigma)    
        
    # T prior.  This one is flat, but another uninformative you might
    # try is the two-sided error function (TwoSidedErf)
        
    Tminval = 0.0 # arcsec squared
    Tmaxval = 400
    T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval)
    
    # FracDev prior for bdf models:
    FD_minval = 0.
    FD_maxval = 1.0
    FD_prior = ngmix.priors.FlatPrior(FD_minval, FD_maxval)
        
    # similar for flux.  Make sure the bounds make sense for
    # your images
        
    Fminval = -1e2
    Fmaxval = 1e4
    F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval)
    
    # now make a joint prior.  This one takes priors
    # for each parameter separately
    priors = pedar.PriorBDFSepMulti(
        [cen_prior1,cen_prior2],
        g_prior,
        T_prior,
        FD_prior,
        F_prior)
    
    return priors

def get_priors_single(entry):

    # prior on ellipticity.  The details don't matter, as long
    # as it regularizes the fit.  This one is from Bernstein & Armstrong 2014
    
    g_sigma = 0.3
    g_prior = ngmix.priors.GPriorBA(g_sigma)
    # 2-d gaussian prior on the center
    # row and column center (relative to the center of the jacobian, which would be zero)
    # and the sigma of the gaussians
        
    # units same as jacobian, probably arcsec
    row, col = entry['x_lowres'], entry['y_lowres']
    row_sigma, col_sigma = .3, .3 
    cen_prior = ngmix.priors.CenPrior(row, col, row_sigma, col_sigma)
        
    # T prior.  This one is flat, but another uninformative you might
    # try is the two-sided error function (TwoSidedErf)
        
    Tminval = 0.0 # arcsec squared
    Tmaxval = 400
    T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval)
    
    # FracDev prior for bdf models:
    FD_minval = 0.
    FD_maxval = 1.0
    FD_prior = ngmix.priors.FlatPrior(FD_minval, FD_maxval)
    
    # similar for flux.  Make sure the bounds make sense for
    # your images
        
    Fminval = -1e2
    Fmaxval = 1e4
    F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval)
    
    # now make a joint prior.  This one takes priors
    # for each parameter separately
    priors = pedar.PriorBDFSepMulti(
        [cen_prior],
        g_prior,
        T_prior,
        FD_prior,
        F_prior)
    
    return priors

def fit_mof(entry,obs,max_tries = 5, max_chi2 = 25):
    prior = get_priors_double(entry)
    npars_per_object = 7 # Only for BDF fits
    acceptable_fit = False
    n_tries = 0
    while (n_tries <= max_tries) & (not acceptable_fit):
        guess = prior.sample()
        fitter = moflib.MOF(obs,model='bdf',nobj=2,prior=prior)
        fitter.go(guess=guess)
        # Did it work?
        result = fitter.get_result()
        if 'chi2per' in result.keys():
            pars = fitter.get_result()['pars']
            chi2 = result['chi2per']
            #print(f"{chi2}")
            if chi2 < max_chi2:
                acceptable_fit = True
                #print ("good enough")
            else:
                n_tries = n_tries+1
        else:
            n_tries = n_tries+1

    params1 = parameter_array(result['pars'][:npars_per_object])
    params1['chisq'] = chi2
    errs1 = parameter_array(result['pars_err'][:npars_per_object])
    params2 = parameter_array(result['pars'][npars_per_object:])
    errs2 = parameter_array(result['pars_err'][npars_per_object:])    
    params2['chisq'] = chi2
            
    #image = fitter.make_image()
    #fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
    #ax1.imshow(obs.image,origin='lower')
    #ax1.plot(entry['y1']-0.3333,entry['x1']-0.3333,'r+')
    #ax1.plot(entry['y2']-0.3333,entry['x2']-0.3333,'r+')    
    #ax2.imshow(image,origin='lower')    
    #ax2.plot(entry['y1']-0.3333,entry['x1']-0.3333,'r+')
    #ax2.plot(entry['y2']-0.3333,entry['x2']-0.3333,'r+')    
    #ax3.imshow(obs.image-image,origin='lower')
    #ax3.plot(entry['y1']-0.3333,entry['x1']-0.3333,'r+')
    #ax3.plot(entry['y2']-0.3333,entry['x2']-0.3333,'r+')
    #chi2per = result['chi2per']
    #fig.suptitle(f"chi2per: {result['chi2per']:.08}")
    #fig.savefig(f"fit_hires-{entry['id']}.png")
    #plt.close(fig)
    return params1,errs1,params2,errs2


def fit_single(entry,obs,max_tries = 5, max_chi2 = 25):
    prior = get_priors_single(entry)
    npars_per_object = 7    
    acceptable_fit = False
    n_tries = 0
    while (n_tries <= max_tries) & (not acceptable_fit):
        guess = prior.sample()
        fitter = moflib.MOF(obs,model='bdf',nobj=1,prior=prior)
        fitter.go(guess=guess)
        # Did it work?
        result = fitter.get_result()
        pars = fitter.get_result()['pars']
        if 'chi2per' in result.keys():
            chi2 = result['chi2per']
            #print(f"{chi2}")
            if chi2 < max_chi2:
                acceptable_fit = True
                #print ("good enough")
            else:
                n_tries = n_tries+1
        else:
            n_tries = n_tries+1
    params = parameter_array(result['pars'][:npars_per_object])
    errs = parameter_array(result['pars_err'][:npars_per_object])    
    params['chisq'] = chi2
    image = fitter.make_image()
    #fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
    #ax1.imshow(obs.image,origin='lower')
    #ax1.plot(entry['y1']-0.3333,entry['x1']-0.3333,'r+')
    #ax1.plot(entry['y2']-0.3333,entry['x2']-0.3333,'r+')
    #ax1.plot(entry['y_lowres']-0.3333,entry['x_lowres']-0.3333,'o')    
    
    #ax2.imshow(image,origin='lower')    
    #ax2.plot(entry['y1']-0.3333,entry['x1']-0.3333,'r+')
    #ax2.plot(entry['y2']-0.3333,entry['x2']-0.3333,'r+')    
    #ax2.plot(entry['y_lowres']-0.3333,entry['x_lowres']-0.3333,'o')    

    #ax3.imshow(obs.image-image,origin='lower')
    #ax3.plot(entry['y1']-0.3333,entry['x1']-0.3333,'r+')
    #ax3.plot(entry['y2']-0.3333,entry['x2']-0.3333,'r+')
    #ax3.plot(entry['y_lowres']-0.3333,entry['x_lowres']-0.3333,'o')    
    
   # chi2per = result['chi2per']
    #fig.suptitle(f"chi2per: {result['chi2per']:.08}")
    #fig.savefig(f"fit_lowres-{entry['id']}.png")
    #plt.close(fig)
    return params,errs
    
def parameter_array(pars):
    array = {}
    array['xcen'] = pars[0]
    array['ycen'] = pars[1]
    array['g'] = [pars[2],pars[3]]
    array['T'] = pars[4]
    array['fracDev'] = pars[5]
    array['flux'] = pars[6]
    return array
    
def get_coordinates(cfile='Coordinates_gan.txt',lowres=False):
    dtype = [('id',np.int),('x1',np.float),('x2',np.float),('y1',np.float),('y2',np.float),('flux1',np.float),('flux2',np.float),('x_lowres',np.int),('y_lowres',np.int)]
    data = np.loadtxt(cfile,dtype=dtype)
    if lowres:
        data['x1'] = data['x1']/3.
        data['x2'] = data['x2']/3.
        data['y1'] = data['y1']/3.
        data['y2'] = data['y2']/3.        
    return data


if __name__ == '__main__':

    coords = get_coordinates(lowres=True)
    flux_calib =  5.07224239892
    flux1 = np.zeros(coords.size)
    flux1_err = np.zeros(coords.size)
    flux2 = np.zeros(coords.size)
    flux2_err = np.zeros(coords.size)
    flux_lores = np.zeros(coords.size)
    flux_lores_err = np.zeros(coords.size)
    flux_ref_lores = np.zeros(coords.size)
    flux_raw = np.zeros(coords.size)
    chi2_lowres = np.zeros(coords.size)
    chi2_hires = np.zeros(coords.size)
    for i,entry in enumerate(coords):
        if i%100 ==0 :
            print(f"{i+1} of {coords.size+1}")
        try:
            obs = make_observation(entry['id'])
        except:
            continue
        flux_raw[i] = np.sum(obs.image)/flux_calib
        #fit_ngmix_simple(entry,obs)
        pars1,errs1,pars2,errs2 = fit_mof(entry,obs,max_tries=10)
        pars_lores,errs_lores = fit_single(entry,obs,max_tries=10)
        chi2_hires[i] = pars1['chisq']
        chi2_lowres[i] = pars_lores['chisq']
        flux1[i] = pars1['flux']/flux_calib
        flux1_err[i] = errs1['flux']/flux_calib
        flux2[i] = pars2['flux']/flux_calib
        flux2_err[i] = errs2['flux']/flux_calib
        flux_lores[i] = pars_lores['flux']/flux_calib
        flux_lores_err[i] = errs_lores['flux']/flux_calib
        # Determine which source this is closer to.
        dist1 = np.sqrt( (pars1['xcen'] - coords['x1'][i])**2 + (pars1['ycen'] - coords['y1'][i])**2)
        dist2 = np.sqrt( (pars1['xcen'] - coords['x2'][i])**2 + (pars1['ycen'] - coords['y2'][i])**2)
        if dist1 < dist2:
            flux_ref_lores[i] = coords['flux1'][i]
        else:
            flux_ref_lores[i] = coords['flux2'][i]


    chi2_thresh = 2.
    good_chi2 = chi2_hires <= chi2_thresh
    bad_chi2 = chi2_hires >  chi2_thresh

    good_chi2_lo = chi2_lowres <= chi2_thresh
    bad_chi2_lo = chi2_lowres >  chi2_thresh
    
    
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
    ax1.errorbar(coords['flux1'][good_chi2],flux1[good_chi2],yerr=flux1_err[good_chi2],fmt=',',label = 'flux 2',color='blue',)
    ax1.errorbar(coords['flux2'][good_chi2],flux2[good_chi2],flux2_err[good_chi2],fmt=',',label='flux 1',color='blue')
    ax1.errorbar(coords['flux1'][bad_chi2],flux1[bad_chi2],flux1_err[bad_chi2],fmt=',',label = 'flux 2',color='red')
    ax1.errorbar(coords['flux2'][bad_chi2],flux2[bad_chi2],flux2_err[bad_chi2],fmt=',',label='flux 1',color='red')

    ax1.plot(coords['flux2'],coords['flux2'],'--',color='red',alpha=0.5)
    ax1.set_xlabel('input flux')
    ax1.set_ylabel('ngmix-mof deblended flux')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.errorbar(flux_ref_lores[good_chi2_lo],flux_lores[good_chi2_lo],yerr=flux_lores_err[good_chi2_lo],fmt=',',color='blue')
    ax2.errorbar(flux_ref_lores[bad_chi2_lo],flux_lores[bad_chi2_lo],yerr=flux_lores_err[bad_chi2_lo],fmt=',',color='red')    
    ax2.plot(flux_ref_lores,flux_ref_lores,'--',color='red',alpha=0.5)
    ax2.set_xlabel('input flux')
    ax2.set_ylabel('ngmix-mof deblended flux')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    plt.savefig('flux_comparison.png')
    
    np.savez('fitting_results.npz',truth=coords,flux1=flux1,flux1_err=flux1_err,flux2=flux2,flux2_err=flux2_err,\
             flux_lores=flux_lores,flux_lores_err=flux_lores_err,flux_ref_lores=flux_ref_lores,
             chi2_hires = chi2_hires,chi2_lowres=chi2_lowres)
    ipdb.set_trace()
    

    
    f = open('photometry_blend1.txt','w+')
    for i in range(len(flux1)):
        f.write(str(flux1[i])+'\t'+str(coords['flux1'][i])+'\n')
    f.close()
    
    f = open('photometry_blend2.txt','w+')
    for i in range(len(flux2)):
        f.write(str(flux2[i])+'\t'+str(coords['flux2'][i])+'\n')
    f.close()
    
    f = open('photometry_lowres.txt','w+')
    for i in range(len(flux_ref_lores)):
        f.write(str(flux_lores[i])+'\t'+str(flux_ref_lores[i])+'\n')
    f.close()
    
        
    #ipdb.set_trace()
