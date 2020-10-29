import numpy as np
from galblend import *
import matplotlib.pyplot as plt
import ngmix
import mof
import moflib
import priors as pedar
import ipdb

# For GP plot.
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class ObjectData():
    def __init__(self,x_true=None,y_true=None, x_gan=None, y_gan = None, x_lores = None, y_lores=None,image_size = 64, resample_factor = 1./3.):


        assert len(x_true) == len(y_true), 'x and y (true) must be the same length'
        assert len(x_gan) == len(y_gan), 'x and y (GAN) must be the same length'
        assert len(x_lores) == len(y_lores), 'x and y (lores) must be the same length'
        
        self.x_true = x_true
        self.y_true = y_true
        self.x_gan = [x*resample_factor for x in x_gan]
        self.y_gan = [y*resample_factor for y in y_gan]

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
        self.scale_hires = 1.0
        self.scale_lores = 1.0
        self.scale_psf = 1./3.
        self.max_tries = max_tries # for fitting.
        self.max_chi2per = max_chi2perdof
        self.flux_calibration = flux_calibration


    def make_simulated_data(self):

        self.object_data = []
        self.hires_obs = []
        self.lores_obs = []
        self.single_obs1 = []
        self.single_obs2 = []
        
        for i in range(self.number_of_images):
            hi,i1,i2,lo,gan,psf_hires,psf_lores,data = galblend(gals=2,lim_hmag=25,plot_it=self.do_image_plots)
            this_hires_obs = self._make_observation(hi,psf_hires,image_scale=self.scale_hires,psf_scale=self.scale_psf)
            this_single_obs1 = self._make_observation(i1,psf_lores,image_scale=self.scale_hires,psf_scale=self.scale_psf)
            this_single_obs2 = self._make_observation(i2,psf_lores,image_scale=self.scale_hires,psf_scale=self.scale_psf)
            this_lores_obs = self._make_observation(lo,psf_lores,image_scale=self.scale_lores,psf_scale=self.scale_psf)

            # Put the catalog positions into an appropriate structure.
            #xhi,yhi = data[0],data[1]
            #xgan,ygan = data[-1][-1][0], data[-1][-1][1]
            #xlo,ylo = data[-1][-2]
            
            yhi,xhi = str2coord(str(data[5][0]))
            ygan,xgan = str2coord(str(data[5][2]))
            ylo,xlo = str2coord(str(data[5][1]))
            
            # Make sure that something was detected in every image.
            if not((len(xhi) == 2) and(len(xlo) == 1) and (len(xgan) == 2)):
                continue
            
            thisData = ObjectData(x_true=xhi,y_true=yhi, x_gan=xgan, y_gan = ygan, x_lores = xlo, y_lores=ylo)
            self.object_data.append(thisData)

            # Store the images in Observation objects.
            self.hires_obs.append(this_hires_obs)
            self.lores_obs.append(this_lores_obs)
            self.single_obs1.append(this_single_obs1)
            self.single_obs2.append(this_single_obs2)

        print(f"finished making Observations for {self.number_of_images} observations.")
            
    def fit_simulated_data(self,render_fits=False,plot_dir='./rendered_fits'):
        # Loop over the existing simulations.
        # Fit them all, and put the results into catalogs.
        catalog_dtype = [('input_flux1',np.float),('input_flux2',np.float),
                         ('input_chi2per1',np.float),('input_chi2per2',np.float),
                         ('input_flux1_err',np.float),('input_flux2_err',np.float),
                         ('hires_deblended_flux1',np.float),('hires_deblended_flux2',np.float),
                         ('hires_deblended_flux1_err',np.float),('hires_deblended_flux2_err',np.float),
                         ('lores_deblended_flux1',np.float),('lores_deblended_flux2',np.float),
                         ('lores_deblended_chi2per',np.float),
                         ('lores_deblended_flux1_err',np.float),('lores_deblended_flux2_err',np.float),
                         ('lores_blended_flux',np.float),('lores_blended_chi2per',np.float),('lores_blended_flux_err',np.float)]
        self.catalog = np.zeros(len(self.object_data),dtype=catalog_dtype)
        
        for i,datum in enumerate(self.object_data):

            # First, fit each hires image separately.
            result1 = self._fit_one_obs(self.single_obs1[i],datum.x_true[0],datum.y_true[0],
                                        render_fit=render_fits,plot_filename=f'{plot_dir}/mof-hires-1-{i:04}.png')
            result2 = self._fit_one_obs(self.single_obs2[i],datum.x_true[1],datum.y_true[1],
                                        render_fit=render_fits,plot_filename=f'{plot_dir}/mof-hires-2-{i:04}.png')
            # Package the results into convenient catalog format.
            self.catalog[i]['input_flux1'] = result1[0]['flux']/self.flux_calibration
            self.catalog[i]['input_chi2per1'] = result1[0]['chi2per']
            self.catalog[i]['input_flux1_err'] = result1[0]['flux_err']/self.flux_calibration
            self.catalog[i]['input_flux2'] = result2[0]['flux']/self.flux_calibration
            self.catalog[i]['input_chi2per2'] = result2[0]['chi2per']            
            self.catalog[i]['input_flux2_err'] = result2[0]['flux_err']/self.flux_calibration


            # Next, deblend on hires.
            deblend_result1,deblend_result2 = self._fit_one_obs( self.hires_obs[i],datum.x_gan,
                                                                datum.y_gan,render_fit=render_fits,plot_filename=f'{plot_dir}/mof-hi-deblended-{i:04}.png')
            # Package the results into convenient catalog format.
            self.catalog[i]['hires_deblended_flux1'] = deblend_result1['flux']/self.flux_calibration
            self.catalog[i]['hires_deblended_flux1_err'] = deblend_result1['flux_err']/self.flux_calibration
            self.catalog[i]['hires_deblended_flux2'] = deblend_result2['flux']/self.flux_calibration
            self.catalog[i]['hires_deblended_flux2_err'] = deblend_result2['flux_err']/self.flux_calibration
            
            
            # Then, deblend on lores.
            deblend_result1,deblend_result2 = self._fit_one_obs( self.lores_obs[i],datum.x_gan,
                                                                datum.y_gan,render_fit=render_fits,plot_filename=f'{plot_dir}/mof-deblended-{i:04}.png')
            # Package the results into convenient catalog format.
            self.catalog[i]['lores_deblended_flux1'] = deblend_result1['flux']/self.flux_calibration
            self.catalog[i]['lores_deblended_chi2per'] = deblend_result1['chi2per']            
            self.catalog[i]['lores_deblended_flux1_err'] = deblend_result1['flux_err']/self.flux_calibration
            self.catalog[i]['lores_deblended_flux2'] = deblend_result2['flux']/self.flux_calibration
            self.catalog[i]['lores_deblended_flux2_err'] = deblend_result2['flux_err']/self.flux_calibration
            
            # Finally, don't deblend.
            blend_result = self._fit_one_obs(self.lores_obs[i], datum.x_lores,datum.y_lores,
                                             render_fit=render_fits,plot_filename=f'{plot_dir}/mof-blended-{i:04}.png')
            self.catalog[i]['lores_blended_flux'] = blend_result[0]['flux']/self.flux_calibration
            self.catalog[i]['lores_blended_chi2per'] = blend_result[0]['chi2per']
            self.catalog[i]['lores_blended_flux_err'] = blend_result[0]['flux_err']/self.flux_calibration

    def _estimate_noise(self,image):
        noise = np.std([image[0,:],image[-1,:],image[:,0],image[:,-1]])
        return noise

    def _estimate_background(self,image):
        bckgd = np.mean([image[0,:],image[-1,:],image[:,0],image[:,-1]])
        return bckgd

    def _make_observation(self, gal_image,psf_image,image_scale=1.0, psf_scale=1.0, image_noise = None,psf_noise = 1e-3, scale = 1.0, background_subtract = True):
        if image_noise is None:
            image_noise = self._estimate_noise(gal_image)
        if background_subtract:
            bckgd = self._estimate_background(gal_image)
            gal_image = gal_image - bckgd
        weight_image = np.ones_like(gal_image) + 1./image_noise**2
        psf_weight_image = np.ones_like(psf_image) + 1./psf_noise**2

        jj_im = ngmix.jacobian.DiagonalJacobian(scale=image_scale,col=0,row=0)
        jj_psf = ngmix.jacobian.DiagonalJacobian(scale=psf_scale,col=0,row=0)#(psf_image.shape[0]-1)/2.,row=(psf_image.shape[0]-1)/2.)
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
        priors = pedar.PriorBDFSepMulti(
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
    
    def _fit_one_obs(self,obs,xpos,ypos,fitpars = {'ftol': 1.0e-5,'xtol': 1.0e-5},max_tries = 5,render_fit=False,plot_filename='model_fit.png'):
        if type(xpos) is not type([]):
            xpos = [xpos]
            ypos = [ypos]
        prior = self._get_priors(x_pos=ypos, y_pos=xpos)
        npars_per_object = 7 # Only for BDF fits
        n_object = len(xpos)
        acceptable_fit = False
        n_tries = 0
        while (n_tries <= self.max_tries) & (not acceptable_fit):
            guess = prior.sample()
            fitter = moflib.MOF(obs,model='bdf',nobj=n_object,prior=prior)
            fitter.go(guess=guess)
            # Did it work?
            result = fitter.get_result()
            if 'chi2per' in result.keys():
                if render_fit:
                    image = fitter.make_image()
                    fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
                    ax1.imshow(obs.image,origin='lower')
                    ax1.plot(ypos,xpos,'r+')
                    ax2.imshow(image,origin='lower')
                    ax2.plot(ypos,xpos,'r+')
                    ax3.imshow(obs.image-image,origin='lower')
                    ax3.plot(ypos,xpos,'r+')
                    chi2per = result['chi2per']
                    fig.suptitle(f"chi2per: {result['chi2per']:.08}")
                    fig.savefig(plot_filename)
                    plt.close(fig)
                pars = fitter.get_result()['pars']
                chi2per = result['chi2per']
                #print(f"{chi2per}")
                if chi2per < self.max_chi2per:
                    acceptable_fit = True
                    #print ("good enough")
                else:
                    n_tries = n_tries+1
            else:
                n_tries = n_tries+1

        results = []
        for i in range(n_object):
            this_params = self._get_parameter_dict(result['pars'][i*npars_per_object:((i+1)*npars_per_object)])
            this_errors = self._get_errors_dict(result['pars_err'][i*npars_per_object:((i+1)*npars_per_object)])
            this_params['chi2per'] = chi2per
            this_params.update(this_errors)
            results.append(this_params)

        return results

        
    
    def go(self):
        self.make_simulated_data()
        self.fit_simulated_data()


    def get_gp(self,x,y,yerr,min_calib_error = 0.05, sys_add_error = 1.):

        X  =  np.atleast_2d(x).T
        Y  = y
        dy = np.sqrt(yerr**2 + (x*min_calib_error)**2 + sys_add_error**2)
        kernel =   C(10.0, (1e-4, 1e4)) * RBF(1000, (1e1, 1e7))
        gp = GaussianProcessRegressor(kernel=kernel, alpha= dy ** 2,n_restarts_optimizer=10)
        gp.fit(X,Y)
        return gp
        

    def make_plots(self,filename='blending_results.png',chi2_thresh = 2.5):
        # Plot various generated catalog quantities against one another.
        fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))

        good_input = (self.catalog['input_chi2per1'] <= chi2_thresh) & (self.catalog['input_chi2per2'] <= chi2_thresh)
        good_deblended = self.catalog['lores_deblended_chi2per'] <= chi2_thresh
        good_blended = self.catalog['lores_blended_chi2per'] <= chi2_thresh

        keep = good_input & good_deblended

        x_train = np.hstack([self.catalog['input_flux1'][keep],self.catalog['input_flux2'][keep]])
        y_train = np.hstack([self.catalog['lores_deblended_flux1'][keep],self.catalog['lores_deblended_flux2'][keep]])
        y_err_train = np.hstack([self.catalog['lores_deblended_flux1_err'][keep],self.catalog['lores_deblended_flux2_err'][keep]])
        
        gp_db = self.get_gp(x_train,y_train/x_train-1,y_err_train/x_train)
        Xinterp = np.atleast_2d(np.logspace(-2,2,100)).T
        Yinterp,Yinterp_err = gp_db.predict(Xinterp,return_std=True)
        errscl = np.sqrt(np.sum(keep))

        ax1.errorbar(self.catalog['input_flux1'][keep],self.catalog['lores_deblended_flux1'][keep]/self.catalog['input_flux1'][keep]-1.,xerr=self.catalog['input_flux1_err'][keep],yerr=self.catalog['lores_deblended_flux1_err'][keep]/self.catalog['input_flux1'][keep],fmt='.',label='flux1,good',color='blue',elinewidth=0.2,markersize=0.5,)
        ax1.errorbar(self.catalog['input_flux1'][~keep],self.catalog['lores_deblended_flux1'][~keep]/self.catalog['input_flux1'][~keep]-1.,xerr=self.catalog['input_flux1_err'][~keep],yerr=self.catalog['lores_deblended_flux1_err'][~keep]/self.catalog['input_flux1'][~keep],fmt='.',label='flux1,bad',color='red',elinewidth=0.2,markersize=0.5,)
        ax1.errorbar(self.catalog['input_flux2'][keep],self.catalog['lores_deblended_flux2'][keep]/self.catalog['input_flux2'][keep]-1.,xerr=self.catalog['input_flux2_err'][keep],yerr=self.catalog['lores_deblended_flux2_err'][keep]/self.catalog['input_flux2'][keep],fmt='.',label='flux2,good',color='blue',elinewidth=0.2,markersize=0.5,)
        ax1.errorbar(self.catalog['input_flux2'][~keep],self.catalog['lores_deblended_flux2'][~keep]/self.catalog['input_flux2'][~keep]-1.,xerr=self.catalog['input_flux2_err'][~keep],yerr=self.catalog['lores_deblended_flux2_err'][~keep]/self.catalog['input_flux2'][~keep],fmt='.',label='flux2,bad',color='red',elinewidth=0.2,markersize=0.5,)
        # Add the GP fit.
        ax1.fill_between(Xinterp.ravel(),np.zeros(Xinterp.size)+0.1,np.zeros(Xinterp.size)-0.1,color='grey',alpha=0.33,zorder=1)
        ax1.fill_between(Xinterp.ravel(),(Yinterp-Yinterp_err),(Yinterp+Yinterp_err),color='orange')
        ax1.fill_between(Xinterp.ravel(),(Yinterp-Yinterp_err*errscl),(Yinterp+Yinterp_err*errscl),color='orange',alpha=0.25)
        ax1.set_xscale('log')
        ax1.set_yscale('symlog')
        ax1.set_xlim(0.1,100)
        keep = good_input & good_blended


        x_train = np.hstack([self.catalog['input_flux1'][keep],self.catalog['input_flux2'][keep]])
        y_train = np.hstack([self.catalog['lores_deblended_flux1'][keep],self.catalog['lores_deblended_flux2'][keep]])
        y_err_train = np.hstack([self.catalog['lores_deblended_flux1_err'][keep],self.catalog['lores_deblended_flux2_err'][keep]])
        
        gp_bl = self.get_gp(x_train,y_train/x_train-1,y_err_train/x_train)
        Xinterp = np.atleast_2d(np.logspace(-2,2,100)).T
        Yinterp,Yinterp_err = gp_bl.predict(Xinterp,return_std=True)
        errscl = np.sqrt(np.sum(keep))

        
        xerr = np.sqrt(self.catalog['input_flux1_err']**2+self.catalog['input_flux2_err']**2)
        ax2.errorbar(self.catalog['input_flux1'][keep]+self.catalog['input_flux2'][keep],self.catalog['lores_blended_flux'][keep]/(self.catalog['input_flux1'][keep]+self.catalog['input_flux2'][keep])-1.,xerr=xerr[keep],yerr=self.catalog['lores_blended_flux_err'][keep]/(self.catalog['input_flux1'][keep]+self.catalog['input_flux2'][keep]),fmt='.',label='good',color='blue',elinewidth=0.2,markersize=0.5)
        ax2.errorbar(self.catalog['input_flux1'][~keep]+self.catalog['input_flux2'][~keep],self.catalog['lores_blended_flux'][~keep]/(self.catalog['input_flux1'][~keep]+self.catalog['input_flux2'][~keep])-1.,xerr=xerr[~keep],yerr=self.catalog['lores_blended_flux_err'][~keep]/(self.catalog['input_flux1'][~keep]+self.catalog['input_flux2'][~keep]),fmt='.',label='bad',color='blue',elinewidth=0.2,markersize=0.5)
        ax2.set_xscale('log')
        ax2.set_yscale('symlog')
        ax2.set_xlim(0.1,100)
        ax2.axhline(0,color='grey',linestyle='--',alpha=0.5)
        ax2.fill_between(Xinterp.ravel(),np.zeros(Xinterp.size)+0.1,np.zeros(Xinterp.size)-0.1,color='grey',alpha=0.33,zorder=1)
        ax2.fill_between(Xinterp.ravel(),(Yinterp-Yinterp_err),(Yinterp+Yinterp_err),color='orange')
        ax2.fill_between(Xinterp.ravel(),(Yinterp-Yinterp_err*errscl),(Yinterp+Yinterp_err*errscl),color='orange',alpha=0.25)
        
        plt.tight_layout()
        fig.savefig(filename)
        #ipdb.set_trace()

    def write_cat(self,filename='blending_results.txt'):
        f=open(filename,'w+')
        f.write('# 0-Input_flux1 1-input_flux2 2-input_chi2per1 3-input_chi2per2 4-input_flux1_err 5-input_flux2_err'+
                '6-hires_deblended_flux1  7-hires_deblended_flux2  8-hires_deblended_flux1_err  9-hires_deblended_flux2_err'+
                '10-lores_deblended_flux1  11-lores_deblended_flux2   12-lores_deblended_chi2per   13-lores_deblended_flux1_err'+
                '14-lores_deblended_flux2_err  15-lores_blended_flux   16-lores_blended_chi2per   17-lores_blended_flux_err'+'\n')
        for i in self.catalog:
            for j in i:
                f.write(str(j)+'\t')
            f.write('\n')
        f.close()

if __name__ == '__main__':
    sim = Simulation(number_of_images=200)
    sim.make_simulated_data()
    sim.fit_simulated_data(render_fits=False)
    sim.make_plots()
    sim.write_cat()
    #ipdb.set_trace()
