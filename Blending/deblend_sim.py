import numpy as np
import galblend
import matplotlib.pyplot as plt


class Simulation(object):

    def __init__(self,number_of_images = 100, do_image_plots=False):
        self.number_of_images = number_of_images
        self.do_image_plots = do_image_plots
        self.scale_hires = 1.0/3.0
        scale.scale_lores = 1.0
        pass

    def make_simulated_data(self):

        self.hires_obs = []
        self.hires_data = []
        self.lores_obs = []
        self.lores_data = []
        self.single_obs1 = []
        self.single_obs2 = []
        self.single_data = []        
        for i in range(self.number_of_images):
            hi,i1,i2,lo,gan,psf_hires,psf_lores,data = galblend.galblend(gals=2,lim_hmag=24,plot_it=self.do_image_plots)
            this_hires_obs = self._make_observation(hi,psf_hires,image_scale=self.scale_hires,psf_scale=self.scale_hires)
            this_single_obs1 = self._make_observation(i1,psf_hires,image_scale=self.scale_hires,psf_scale=self.scale_hires)
            this_single_obs2 = self._make_observation(i2,psf_hires,image_scale=self.scale_hires,psf_scale=self.scale_hires)
            this_lores_obs = self._make_observation(lo,psf_lores,image_scale=self.scale_lores,psf_scale=self.scale_lores)

            self.hires_obs.append(this_hires_obs)
            xhi,yhi = data[0],data[1]
            self.hires_data.append([[xhi[0],yhi[0]],[xhi[1],yhi[1]]])
            self.lores_obs.append(this_lores_obs)
            self.lores_data.append(data[1])
            self.single_obs1.append(this_single_obs1)
            self.single_data.append(data[2])
            self.single_obs2.append(this_single_obs2)

        print(f"finished making Observations for {self.number_of_images} observations.")
            
    def fit_simulated_data(self):
        # Loop over the existing simulations.
        # Fit them all, and put the results into catalogs.
        

        
        pass
    
    def _make_observation(self, image,psf,image_scale=1.0, psf_scale=1.0, image_noise = .055,psf_noise = 1e-3, scale = 1.0):
        gal_image = get_image(objid)
        psf_image = get_psf(objid)
        weight_image = np.ones_like(gal_image) + 1./image_noise**2
        psf_weight_image = np.ones_like(psf_image) + 1./psf_noise**2

        jj_im = ngmix.jacobian.DiagonalJacobian(scale=image_scale,col=0,row=0)
        jj_psf = ngmix.jacobian.DiagonalJacobian(scale=psf_scale,col=(psf_image.shape[0]-1)/2.,row=(psf_image.shape[0]-1)/2.)
        psf_obs = ngmix.Observation(psf_image,weight=psf_weight_image,jacobian=jj_psf)
        psf_gmix = fit_psf(psf_obs)
        psf_obs.set_gmix(psf_gmix)
        gal_obs = ngmix.Observation(gal_image, weight=weight_image,jacobian=jj_im, psf=psf_obs)
    
        return gal_obs

    def _get_priors(self, x_pos = 0.0, y_pos = 0.0):
        '''
        Returns a priors object to handle deblending for one or more sources in a single image.

        x_pos: list of x positions for objects in image.
        y_pos: list of y-positions for objects in image.
        '''

        assert len(x_pos) == len(y_pos),"x and y positions must be the same length."
        
        # prior on ellipticity.  The details don't matter, as long
        # as it regularizes the fit.  This one is from Bernstein & Armstrong 2014
        
        g_sigma = 0.3
        g_prior = ngmix.priors.GPriorBA(g_sigma)
        # 2-d gaussian prior on the center
        # row and column center (relative to the center of the jacobian, which would be zero)
        # and the sigma of the gaussians
        
        # units in pixels
        # Center relative to Jacobian center.
        row_sigma, col_sigma = .3, .3
        cen_priors = []
        for ix,iy in zip(x_pos,y_pos):
            this_cen_prior = ngmix.priors.CenPrior(ix, iy, row_sigma, col_sigma)
            cen_priors.append(this_cen_prior)
        
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
            cen_priors,
            g_prior,
            T_prior,
            FD_prior,
            F_prior)
    
        return priors
    def _get_parameter_dict(self,pars):
        result = {}
        result['xcen'] = pars[0]
        result['ycen'] = pars[1]
        result['g'] = [pars[2],pars[3]]
        result['T'] = pars[4]
        result['fracDev'] = pars[5]
        result['flux'] = pars[6]
        return result

    def _get_errors_dict(self,pars):
        result = {}
        result['xcen_err'] = pars[0]
        result['ycen_err'] = pars[1]
        result['g_err'] = [pars[2],pars[3]]
        result['T_err'] = pars[4]
        result['fracDev_err'] = pars[5]
        result['flux_err'] = pars[6]
        return result
    
    

    def _fit_one_obs(self,obs,,xpos,ypos,fitpars = {'ftol': 1.0e-5,'xtol': 1.0e-5}):
        prior = self._get_priors(x_xpos=xpos, y_pos=ypos)
        npars_per_object = 7 # Only for BDF fits
        n_object = len(xpos)
        acceptable_fit = False
        n_tries = 0
        while (n_tries <= max_tries) & (not acceptable_fit):
            guess = prior.sample()
            fitter = mof.moflib.MOF(obs,model='bdf',nobj=n_object,prior=prior)
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

        results = []
        for i in range(n_object):
            this_params = self._get_parameter_dict(result['pars'][i*npars_per_object:((i+1)*npars_per_object)])
            this_errors = self._get_errors_dict(result['pars_err'][i*npars_per_object:((i+1)*npars_per_object)])
            this_params['chisq'] = chi2
            this_params.update(this_errors)
            results.append(this_params)

        return results

        
    
    def go(self):
        pass

    
