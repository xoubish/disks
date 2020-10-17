import numpy as np
import matplotlib.pyplot as plt
import fitsio
import ngmix
import mof
import os
import scipy.optimize
import ipdb

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
    priors = mof.priors.PriorBDFSepMulti(
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
    priors = mof.priors.PriorBDFSepMulti(
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
        fitter = mof.moflib.MOF(obs,model='bdf',nobj=2,prior=prior)
        fitter.go(guess=guess)
        # Did it work?
        result = fitter.get_result()
        if 'chi2per' in result.keys():
            pars = fitter.get_result()['pars']
            chi2 = result['chi2per']
            print(f"{chi2}")
            if chi2 < max_chi2:
                acceptable_fit = True
                print ("good enough")
            else:
                n_tries = n_tries+1
        else:
            n_tries = n_tries+1

    params1 = parameter_array(result['pars'][:npars_per_object])
    params1['chisq'] = chi2
    params2 = parameter_array(result['pars'][npars_per_object:])
    params2['chisq'] = chi2
            
    image = fitter.make_image()
    fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
    ax1.imshow(obs.image,origin='lower')
    ax1.plot(entry['y1']-0.3333,entry['x1']-0.3333,'r+')
    ax1.plot(entry['y2']-0.3333,entry['x2']-0.3333,'r+')    
    ax2.imshow(image,origin='lower')    
    ax2.plot(entry['y1']-0.3333,entry['x1']-0.3333,'r+')
    ax2.plot(entry['y2']-0.3333,entry['x2']-0.3333,'r+')    
    ax3.imshow(obs.image-image,origin='lower')
    ax3.plot(entry['y1']-0.3333,entry['x1']-0.3333,'r+')
    ax3.plot(entry['y2']-0.3333,entry['x2']-0.3333,'r+')
    chi2per = result['chi2per']
    fig.suptitle(f"chi2per: {result['chi2per']:.08}")
    fig.savefig(f"fit_hires-{entry['id']}.png")
    plt.close(fig)
    return params1,params2


def fit_single(entry,obs,max_tries = 5, max_chi2 = 25):
    prior = get_priors_single(entry)
    npars_per_object = 7    
    acceptable_fit = False
    n_tries = 0
    while (n_tries <= max_tries) & (not acceptable_fit):
        guess = prior.sample()
        fitter = mof.moflib.MOF(obs,model='bdf',nobj=1,prior=prior)
        fitter.go(guess=guess)
        # Did it work?
        result = fitter.get_result()
        pars = fitter.get_result()['pars']
        if 'chi2per' in result.keys():
            chi2 = result['chi2per']
            print(f"{chi2}")
            if chi2 < max_chi2:
                acceptable_fit = True
                print ("good enough")
            else:
                n_tries = n_tries+1
        else:
            n_tries = n_tries+1
    params = parameter_array(result['pars'][:npars_per_object])
    params['chisq'] = chi2
    image = fitter.make_image()
    fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
    ax1.imshow(obs.image,origin='lower')
    ax1.plot(entry['y1']-0.3333,entry['x1']-0.3333,'r+')
    ax1.plot(entry['y2']-0.3333,entry['x2']-0.3333,'r+')
    ax1.plot(entry['y_lowres']-0.3333,entry['x_lowres']-0.3333,'o')    
    
    ax2.imshow(image,origin='lower')    
    ax2.plot(entry['y1']-0.3333,entry['x1']-0.3333,'r+')
    ax2.plot(entry['y2']-0.3333,entry['x2']-0.3333,'r+')    
    ax2.plot(entry['y_lowres']-0.3333,entry['x_lowres']-0.3333,'o')    

    ax3.imshow(obs.image-image,origin='lower')
    ax3.plot(entry['y1']-0.3333,entry['x1']-0.3333,'r+')
    ax3.plot(entry['y2']-0.3333,entry['x2']-0.3333,'r+')
    ax3.plot(entry['y_lowres']-0.3333,entry['x_lowres']-0.3333,'o')    
    
    chi2per = result['chi2per']
    fig.suptitle(f"chi2per: {result['chi2per']:.08}")
    fig.savefig(f"fit_lowres-{entry['id']}.png")
    plt.close(fig)
    return params
    
def parameter_array(pars):
    array = {}
    array['xcen'] = pars[0]
    array['ycen'] = pars[1]
    array['g'] = [pars[2],pars[3]]
    array['T'] = pars[4]
    array['fracDev'] = pars[5]
    array['flux'] = pars[6]
    return array

'''
def fit_ngmix_simple(entry,obs,n_guess = 20):
    prior = get_priors(entry)
    guess = prior.sample()
    npars_per_object = 7
    model_type = 'bdf'
    method = "Nelder-Mead"

    def eval_model(pars):
        pars_gm1 = pars[:npars_per_object]
        gm1 = ngmix.gmix.GMixBDF(pars_gm1).convolve(obs.psf.get_gmix())
        pars_gm2 = pars[npars_per_object:]
        gm2 = ngmix.gmix.GMixBDF(pars_gm2).convolve(obs.psf.get_gmix())
        im1 = gm1.make_image(obs.image.shape,obs.jacobian)
        im2 = gm2.make_image(obs.image.shape,obs.jacobian)
        model = im1+im2
        return model
        
    
    def chisq(pars):
        try:
            model = eval_model(pars)
            chisq = np.sum((obs.image - model)**2*obs.weight)
            penalty = -2*prior.get_lnprob_scalar(pars)
            return chisq + penalty
        except:
            return np.inf


    chisq_guess = np.zeros(n_guess)
    guesses = np.zeros((n_guess,guess.size))
    for i in range(n_guess):
        guesses[:,i] = prior.sample()
        chisq_guess[i] = chi(guesses[:,i],chisq=True)
    best_guess = guesses[:,np.argmin(chisq_guess)] 
    result = scipy.optimize.minimize(chisq,guess,method=method)
    model = eval_model(result.x)
    pars1 = parameter_array(result.x[:npars_per_object])
    pars2 = parameter_array(result.x[npars_per_object:])
    print(pars1)
    print(pars2)
   

    
    fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
    vmin,vmax = -1,50
    cmap = plt.cm.inferno
    ax1.imshow(obs.image,origin='lower',cmap=cmap,vmin=vmin,vmax=vmax)
    ax1.plot(entry['x1'],entry['y1'],'r+')
    ax1.plot(entry['x2'],entry['y2'],'r+')    
    ax2.imshow(model,origin='lower',vmin=vmin,vmax=vmax,cmap=cmap)
    ax2.plot(entry['x1'],entry['y1'],'r+')
    ax2.plot(entry['x2'],entry['y2'],'r+')    
    ax3.imshow(obs.image-model,origin='lower',vmin=vmin,vmax=vmax,cmap=cmap)
    ax3.plot(entry['x1'],entry['y1'],'r+')
    ax3.plot(entry['x2'],entry['y2'],'r+')    
    
    plt.show()
    ipdb.set_trace()
'''    
    
    
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
    flux2 = np.zeros(coords.size)
    flux_lores = np.zeros(coords.size)
    flux_ref_lores = np.zeros(coords.size)
    flux_raw = np.zeros(coords.size)
    
    for i,entry in enumerate(coords):
        print(f"{i+1} of {coords.size+1}")
        obs = make_observation(entry['id'])
        flux_raw[i] = np.sum(obs.image)/flux_calib
        #fit_ngmix_simple(entry,obs)
        pars1,pars2 = fit_mof(entry,obs,max_tries=10)
        pars_lores = fit_single(entry,obs,max_tries=10)

        flux1[i] = pars1['flux']/flux_calib
        flux2[i] = pars2['flux']/flux_calib
        flux_lores[i] = pars_lores['flux']/flux_calib
        # Determine which source this is closer to.
        dist1 = np.sqrt( (pars1['xcen'] - coords['x1'][i])**2 + (pars1['ycen'] - coords['y1'][i])**2)
        dist2 = np.sqrt( (pars1['xcen'] - coords['x2'][i])**2 + (pars1['ycen'] - coords['y2'][i])**2)
        if dist1 < dist2:
            flux_ref_lores[i] = coords['flux1'][i]
        else:
            flux_ref_lores[i] = coords['flux2'][i]

        
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
    ax1.plot(coords['flux1'],flux1,'.',label = 'flux 2')
    ax1.plot(coords['flux2'],flux2,'.',label='flux 1')
    ax1.plot(coords['flux2'],coords['flux2'],'--',color='red',alpha=0.5)
    ax1.set_xlabel('input flux')
    ax1.set_ylabel('ngmix-mof deblended flux')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.plot(flux_ref_lores,flux_lores,'.')
    ax2.plot(flux_ref_lores,flux_ref_lores,'--',color='red',alpha=0.5)
    ax2.set_xlabel('input flux')
    ax2.set_ylabel('ngmix-mof deblended flux')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    plt.savefig('flux_comparison.png')
    ipdb.set_trace()
