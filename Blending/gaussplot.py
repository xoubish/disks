import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
import ipdb


min_calib_error = 0.1
data = np.load('fitting_results.npz')
keep = (data['flux1'] > 0) & (data['chi2_hires'] < 2.)
keepL = (data['flux_ref_lores']>0) & (data['chi2_lowres'] < 2.)

truth = data['truth'][keep]

sigma_calib1 = min_calib_error*np.abs(data['flux1'][keep])
sigma_calib2 = min_calib_error*np.abs(data['flux2'][keep] )                              
sigma_calib_lores =  min_calib_error*np.abs(data['flux_lores'][keepL] )                              

sigma_meas1 = np.sqrt(data['flux1_err'][keep]**2 + sigma_calib1**2)
sigma_meas2 = np.sqrt(data['flux2_err'][keep]**2 + sigma_calib2**2)
sigma_meas_lores = np.sqrt(data['flux_lores_err'][keepL]**2 + sigma_calib_lores**2)

X  =  np.atleast_2d(np.hstack([truth['flux1'],truth['flux2']])).T
XL =  np.atleast_2d(data['flux_ref_lores'][keepL]).T
Y  = np.hstack([data['flux1'][keep] - truth['flux1'],data['flux2'][keep] - truth['flux2']])
YL = data['flux1'][keepL]+data['flux2'][keepL]#data['flux_lores'][keepL]
dy = np.hstack([sigma_meas1,sigma_meas2])
dyL = sigma_meas_lores

kernel =   C(10.0, (1e-3, 1e3)) * RBF(10000, (1e1, 1e7))
gp = GaussianProcessRegressor(kernel=kernel, alpha= dy ** 2,n_restarts_optimizer=10)
gpL = GaussianProcessRegressor(kernel=kernel, alpha= dyL ** 2,n_restarts_optimizer=10)
gp.fit(X,Y)
Xinterp = np.atleast_2d(np.logspace(-2,2,100)).T
Yinterp,Yinterp_err = gp.predict(Xinterp,return_std=True)

gpL.fit(XL,YL)
YLinterp,YLinterp_err = gpL.predict(Xinterp,return_std=True)


# Now plot:
# -- the data, with errors
fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
ax1.errorbar(X.ravel(),Y/X.ravel(),dy/X.ravel(),elinewidth=0.2,markersize=0.5,fmt='.',zorder=2)
ax1.plot(X.ravel(),Y/X.ravel(),'.',markersize=0.5,color='red',zorder=3)
ax1.fill_between(Xinterp.ravel(),np.zeros(Xinterp.size)+0.1,np.zeros(Xinterp.size)-0.1,color='grey',alpha=0.33,zorder=1)
# -- the model fit
#plt.plot(Xinterp.ravel(),Yinterp+Xinterp.ravel())
ax1.fill_between(Xinterp.ravel(),(Yinterp-Yinterp_err)/Xinterp.ravel(),(Yinterp+Yinterp_err)/Xinterp.ravel(),color='orange')
ax1.axhline(0,color='black',alpha=0.5,linestyle='--')
ax1.semilogx()
ax1.set_yscale('symlog')
ax1.set_xlim(0.1,100)
ax1.set_ylabel('fractional error')
ax1.set_xlabel('input flux')
ax1.set_ylim(-50,50)

ax2.errorbar(XL.ravel(),YL/XL.ravel(),dyL/XL.ravel(),elinewidth=0.2,markersize=0.5,fmt='.',zorder=2)
ax2.plot(XL.ravel(),YL/XL.ravel(),'.',markersize=0.5,color='red',zorder=3)
# -- the model fit
#plt.plot(Xinterp.ravel(),Yinterp+Xinterp.ravel())
ax2.fill_between(Xinterp.ravel(),(YLinterp-YLinterp_err)/Xinterp.ravel(),(YLinterp+YLinterp_err)/Xinterp.ravel(),color='orange')
ax2.fill_between(Xinterp.ravel(),np.zeros(Xinterp.size)+0.1,np.zeros(Xinterp.size)-0.1,color='grey',alpha=0.33,zorder=1)
ax2.axhline(0,color='grey',alpha=0.5,linestyle='--')
ax2.semilogx()
ax2.set_yscale('symlog')
ax2.set_ylabel('fractional error')
ax2.set_xlabel('input flux')
ax2.set_ylim(-50,50)
ax2.set_xlim(0.1,100)

fig.savefig('flux-difference-gp-diagnostics.png')
plt.show()
ipdb.set_trace()

