# Disks

The idea is to synthesize realistic 3D disk galaxies, indistingushable from IFU data lets say by MUSE. 

### simple simulated disks
Using galsims one can create simple disk images with a light distribution, and add inclination, shear, etc. Also, observational effects can be incorporated such as convolution with a PSF, adding background sky, etc. Simple example shown in "galfit_disk.ipynb". With some modifications to it, I generate simple disks with velocity information as third dimension using given there exists a rotation curve. This can be found in "3Ddisk.ipynb". However, with these examples the synthesized images look still very different from real observations, no clumpiness for instance.

### what about clumps
Here, the idea of NN comes to play. GANs are relatively new networks which do surprisingly good in generating real looking images. I aim to start by making 2D disk images similar to those in CANDELS (my training set) and move on by adding in kinematics (will get MUSE abell). 

In this directory there is also a notebook example of fitting 3D disk data to find physical parameters of a disk, called "galpak_test.ipynb". This can be useful after realistic disks are generated to measure parameters ...


### Fist things first

Ok before getting excited about realistic disk images, I should learn GANs, and which versions to use. I start GAN with MNIST examples from web. I will add the learning notebooks in this directory as well. Starting with a simple GAN implementation in the tensorflow (no keras). found in "MNIST_tf.ipynb"
