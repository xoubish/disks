import numpy as np
import galblend
import matplotlib.pyplot as plt
import ngmix
import mof
import ipdb


class ObjectData():
    def __init__(self,x_true=None,y_true=None, x_gan=None, y_gan = None, x_lores = None, y_lores=None,image_size = 64, resample_factor = 1./3.):


        assert len(x_true) == len(y_true), 'x and y (true) must be the same length'
        assert len(x_gan) == len(y_gan), 'x and y (GAN) must be the same length'
        assert len(x_lores) == len(y_lores), 'x and y (lores) must be the same length'
        
        self.x_true = x_true
        self.y_true = y_true
        self.x_gan = x_gan
        self.y_gan = y_gan

        # Here let's just choose the closest object to the center.
        xcen,ycen = image_size/2.,image_size/2.
        dist = np.sqrt((np.array(x_lores) - xcen)**2 + (np.array(y_lores)-ycen)**2)
        self.lores_match_ind = np.argmin(dist)
        
        self.x_lores = x_lores[self.lores_match_ind]*resample_factor
        self.y_lores = y_lores[self.lores_match_ind]*resample_factor

        self.nobj_true = len(self.x_true)
        self.nobj_gan = len(self.x_gan)
        self.nobj_lores = 1


class Simulation(object):

    def __init__(self,number_of_images = 100, do_image_plots=False,max_tries=5,max_chi2perdof = 2.,flux_calibration = 5.07224239892):
        self.number_of_images = number_of_images
        self.do_image_plots = do_image_plots
        self.scale_hires = 1.0/3.0
        self.scale_lores = 1.0
        self.max_tries = max_tries # for fitting.
        self.max_chi2 = max_chi2perdof
        self.flux_calibration = flux_calibration


    def make_simulated_data(self):


        self.object_data = []
        self.hires_obs = []
        self.lores_obs = []
        self.single_obs1 = []
        self.single_obs2 = []
        
        for i in range(self.number_of_images):
            hi,i1,i2,lo,gan,psf_hires,psf_lores,data = galblend.galblend(gals=2,lim_hmag=24,plot_it=self.do_image_plots)
            this_hires_obs = self._make_observation(hi,psf_hires,image_scale=self.scale_hires,psf_scale=self.scale_hires)
            this_single_obs1 = self._make_observation(i1,psf_hires,image_scale=self.scale_hires,psf_scale=self.scale_hires)
            this_single_obs2 = self._make_observation(i2,psf_hires,image_scale=self.scale_hires,psf_scale=self.scale_hires)
            this_lores_obs = self._make_observation(lo,psf_lores,image_scale=self.scale_lores,psf_scale=self.scale_lores)

            # Put the catalog positions into an appropriate structure.
            xhi,yhi = data[0],data[1]
            xgan,ygan = data[-1][-1][0], data[-1][-1][1]
            xlo,ylo = data[-1][-2]
            # Make sure that something was detected in every image.
            if (len(xlo) == 0) or (len(xgan) != 2) or (len(xhi) != 2):
                continue
            
            thisData = ObjectData(x_true=xhi,y_true=yhi, x_gan=xgan, y_gan = ygan, x_lores = xlo, y_lores=ylo)
            self.object_data.append(thisData)

            # Store the images in Observation objects.
            self.hires_obs.append(this_hires_obs)
            self.lores_obs.append(this_lores_obs)
            self.single_obs1.append(this_single_obs1)
            self.single_obs2.append(this_single_obs2)

        print(f"finished making Observations for {self.number_of_images} observations.")
            
    def fit_simulated_data(self):
        # Loop over the existing simulations.
        # Fit them all, and put the results into catalogs.
        catalog_dtype = [('input_flux1',np.float),('input_flux2',np.float),
                         ('input_flux1_err',np.float),('input_flux2_err',np.float),
                         ('hires_deblended_flux1',np.float),('hires_deblended_flux2',np.float),
                         ('hires_deblended_flux1_err',np.float),('hires_deblended_flux2_err',np.float),                         
                         ('lores_deblended_flux1',np.float),('lores_deblended_flux2',np.float),
                         ('lores_deblended_flux1_err',np.float),('lores_deblended_flux2_err',np.float),
                         ('lores_blended_flux',np.float),('lores_blended_flux_err',np.float)]
        self.catalog = np.zeros(len(self.object_data),dtype=catalog_dtype)
        
        for i,datum in enumerate(self.object_data):

            # First, fit each hires image separately.
            result1 = self._fit_one_obs(self.single_obs1[i],datum.x_true[0],datum.y_true[0])
            result2 = self._fit_one_obs(self.single_obs2[i],datum.x_true[1],datum.y_true[1])
            # Package the results into convenient catalog format.
            self.catalog[i]['input_flux1'] = result1[0]['flux']/self.flux_calibration
            self.catalog[i]['input_flux1_err'] = result1[0]['flux_err']/self.flux_calibration
            self.catalog[i]['input_flux2'] = result2[0]['flux']/self.flux_calibration
            self.catalog[i]['input_flux2_err'] = result2[0]['flux_err']/self.flux_calibration

            # Then, deblend on lores.
            deblend_result1,deblend_result2 = self._fit_one_obs( self.lores_obs[i],datum.x_gan,datum.y_gan )
            # Package the results into convenient catalog format.
            self.catalog[i]['lores_deblended_flux1'] = deblend_result1['flux']/self.flux_calibration
            self.catalog[i]['lores_deblended_flux1_err'] = deblend_result1['flux_err']/self.flux_calibration
            self.catalog[i]['lores_deblended_flux2'] = deblend_result2['flux']/self.flux_calibration
            self.catalog[i]['lores_deblended_flux2'] = deblend_result2['flux_err']/self.flux_calibration
            
            # Finally, don't deblend.
            blend_result = self._fit_one_obs(self.lores_obs[i], datum.x_lores,datum.y_lores)
            self.catalog[i]['lores_blended_flux'] = blend_result[0]['flux']/self.flux_calibration
            self.catalog[i]['lores_blended_flux_err'] = blend_result[0]['flux_err']/self.flux_calibration
            
                
    def _make_observation(self, gal_image,psf_image,image_scale=1.0, psf_scale=1.0, image_noise = .055,psf_noise = 1e-3, scale = 1.0):
        weight_image = np.ones_like(gal_image) + 1./image_noise**2
        psf_weight_image = np.ones_like(psf_image) + 1./psf_noise**2

        jj_im = ngmix.jacobian.DiagonalJacobian(scale=image_scale,col=0,row=0)
        jj_psf = ngmix.jacobian.DiagonalJacobian(scale=psf_scale,col=(psf_image.shape[0]-1)/2.,row=(psf_image.shape[0]-1)/2.)
        psf_obs = ngmix.Observation(psf_image,weight=psf_weight_image,jacobian=jj_psf)
        psf_gmix = self._fit_psf(psf_obs)
        psf_obs.set_gmix(psf_gmix)
        gal_obs = ngmix.Observation(gal_image, weight=weight_image,jacobian=jj_im, psf=psf_obs)
    
        return gal_obs

    def _get_priors(self, x_pos = 0.0, y_pos = 0.0):
        '''
        Returns a priors object to handle deblending for one or more sources in a single image.

        x_pos: list of x positions for objects in image.
        y_pos: list of y-positions for objects in image.
        '''

        #assert len(x_pos) == len(y_pos),"x and y positions must be the same length."
        
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

    
    def _fit_psf(self,psfObs):
        lm_pars = {'ftol': 1.0e-5,
                    'xtol': 1.0e-5}
        psf_boot = ngmix.bootstrap.PSFRunner(psfObs,'turb',1.0,lm_pars=lm_pars)
        psf_boot.go()
        gm_psf = psf_boot.fitter.get_gmix()
        return gm_psf
    

    def _fit_one_obs(self,obs,xpos,ypos,fitpars = {'ftol': 1.0e-5,'xtol': 1.0e-5},max_tries = 5):
        if type(xpos) is not type([]):
            xpos = [xpos]
            ypos = [ypos]
        prior = self._get_priors(x_pos=xpos, y_pos=ypos)
        npars_per_object = 7 # Only for BDF fits
        n_object = len(xpos)
        acceptable_fit = False
        n_tries = 0
        while (n_tries <= self.max_tries) & (not acceptable_fit):
            guess = prior.sample()
            fitter = mof.moflib.MOF(obs,model='bdf',nobj=n_object,prior=prior)
            fitter.go(guess=guess)
            # Did it work?
            result = fitter.get_result()
            if 'chi2per' in result.keys():
                pars = fitter.get_result()['pars']
                chi2 = result['chi2per']
                print(f"{chi2}")
                if chi2 < self.max_chi2:
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
        self.make_simulated_data()
        self.fit_simulated_data()

    


if __name__ == '__main__':
    sim = Simulation(number_of_images=100)
    sim.make_simulated_data()
    sim.fit_simulated_data()
    ipdb.set_trace()
