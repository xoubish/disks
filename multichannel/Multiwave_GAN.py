from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from photutils import create_matching_kernel
import torch.nn.functional as F
import torchvision


import astropy.io.fits as pyfits
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from skimage.transform import downscale_local_mean
from scipy.ndimage import zoom

from galaxy_hdf5loader import galaxydata

nc=7

hi_psfs = ['psf_b.fits','psf_v.fits', 'psf_i.fits','psf_i.fits', 'psf_z.fits', 'psf_j.fits', 'psf_h.fits']
lo_psfs = ['PSF_subaru_i.fits','PSF_subaru_i.fits','PSF_subaru_i.fits','PSF_subaru_i.fits',
           'PSF_subaru_i.fits','PSF_subaru_i.fits','PSF_subaru_i.fits']


kernel = np.zeros((41,41,1,7))
for i in range(len(hi_psfs)):
    psf = pyfits.getdata('../psfs/'+hi_psfs[i])
    psf = downscale_local_mean(psf,(3,3))
    psf = psf[7:-8,7:-8]

    psf_hsc = pyfits.getdata('../psfs/'+lo_psfs[i])
    psf_hsc = psf_hsc[1:42,1:42]    
    kern = create_matching_kernel(psf,psf_hsc)
    psfh = np.repeat(kern[:,:, np.newaxis], 1, axis=2)
    kernel[:,:,:,i] = psfh
  
kernel = torch.Tensor(kernel)
kernel = kernel.permute(2,3,0,1)
kernel =  kernel.float()
kernel = kernel.cuda()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.8, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=3, help='number of GPUs to use')
parser.add_argument('--netS', default='', help="path to netS (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='outputs', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")



dataset = galaxydata('Sample_train.hdf5')
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Shoobygen(nn.Module):

    def __init__(self,ngpu):
        super(Shoobygen, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            nn.Conv2d(nc, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            
            nn.ConvTranspose2d( ngf*4, ngf * 8, 3, 3, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 7, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            
            nn.ConvTranspose2d(ngf*4, nc, 4, 1, 0, bias=False),
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
netS.apply(weights_init)
print(netS)



class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr= 0.00001, betas=(opt.beta1, 0.999))
optimizerS = optim.Adam(netS.parameters(), lr= opt.lr, betas=(opt.beta1, 0.999))

writer = SummaryWriter()

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        real_cpu = real_cpu.float()
        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        
        kernel = kernel.to(device)
        img2 = torch.tensor(np.zeros((batch_size,nc,22,22)))
        img = torch.tensor(np.zeros((batch_size,nc,22,22)))

        for ch in range(real_cpu.shape[1]):
            imagetoconvolv = real_cpu[:,ch,:,:].reshape(-1,1,64,64)
            kerneltoconvolv = kernel[:,ch,:,:].reshape(-1,1,41,41) 
            a = F.conv2d(imagetoconvolv, kerneltoconvolv,padding = 21) ## convolve with kernel
            img2[:,ch,:,:] = (F.upsample(a,scale_factor=1/3,mode='bilinear')).reshape(-1,22,22) ### fix pixel scale
          
            t,ashghal = torch.median(img2[:,ch,2:-2,:],dim=1)
            img2[:,ch,-2:,:]=  torch.reshape(t,(batch_size,1,22)) 
            t,ashghal = torch.median(img2[:,ch,:,2:-2],dim=2)
            img2[:,ch,:,-2:]= torch.reshape(t,(batch_size,22,1)) 
            t,ashghal = torch.median(img2[:,ch,2:-2,:],dim=1)
            img2[:,ch,:2,:]=  torch.reshape(t,(batch_size,1,22)) 
            t,ashghal = torch.median(img2[:,ch,:,2:-2],dim=2)
            img2[:,ch,:,:2]= torch.reshape(t,(batch_size,22,1)) 
            img[:,ch,:,:] = img2[:,ch,:,:]+0.25*torch.rand_like(img2[:,ch,:,:]) ### add noise            
        
        img = img.view(-1,nc,22,22)
        img = img[:,:,:,:].float().cuda()
 
        fake = netS(img)
        label.fill_(fake_label)
        fd = fake.detach()
        output = netD(fd.float())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netS.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerS.step()
        
        writer.add_scalar('Generator/Error', errG.item(), epoch)
        writer.add_scalar('Discriminator/Error', errD.item(), epoch)
        writer.add_scalar('Discriminator/mean_out_real', D_x, epoch)
        writer.add_scalar('Discriminator/mean_out_fake', D_G_z1, epoch)


        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            real_cp = real_cpu[:,2,:,:].reshape(opt.batchSize,1,opt.imageSize,opt.imageSize)        
            vutils.save_image(real_cp,'%s/real_samples.png' % opt.outf, normalize=True)
            fake = netS(img)
            fak = fake[:,2,:,:].reshape(opt.batchSize,1,opt.imageSize,opt.imageSize)
            vutils.save_image(fak.detach(),'%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),normalize=True)
            grid = torchvision.utils.make_grid(fak.detach())
            writer.add_image('images',grid,i)

            
    # do checkpointing
    torch.save(netS.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    writer.close()
