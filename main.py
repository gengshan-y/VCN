from __future__ import print_function
import cv2
cv2.setNumThreads(0)
import sys
import pdb
import argparse
import collections
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from utils.flowlib import flow_to_image
from models import *
from utils import logger
torch.backends.cudnn.benchmark=True
from utils.multiscaleloss import realEPE


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=256,
                    help='maxium disparity, out of range pixels will be masked out. Only affect the coarsest cost volume size')
parser.add_argument('--fac', type=float ,default=1,
                    help='controls the shape of search grid. Only affect the coarsest cost volume size')
parser.add_argument('--logname', default='logname',
                    help='name of the log file')
parser.add_argument('--database', default='/',
                    help='path to the database')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='path of the pre-trained model')
parser.add_argument('--model', default='VCN',
                    help='VCN or VCN_small')
parser.add_argument('--savemodel', default='./',
                    help='path to save the model')
parser.add_argument('--retrain', default='false',
                    help='whether to reset moving mean / other hyperparameters')
parser.add_argument('--stage', default='chairs',
                    help='one of {chairs, things, 2015train, 2015trainval, sinteltrain, sinteltrainval}')
parser.add_argument('--ngpus', type=int, default=2,
                    help='number of gpus to use.')
args = parser.parse_args()

if args.model == 'VCN':
    from models.VCN import VCN
elif args.model == 'VCN_small':
    from models.VCN_small import VCN

# fix random seed
torch.manual_seed(1)
def _init_fn(worker_id):
    np.random.seed()
    random.seed()
torch.manual_seed(8)  # do it again
torch.cuda.manual_seed(1)

## set hyperparameters for training
ngpus = args.ngpus
batch_size = 4*ngpus
if args.stage == 'chairs' or args.stage == 'things':
    lr_schedule = 'slong_ours'
else:
    lr_schedule = 'rob_ours'
#baselr = 1e-4
baselr = 1e-3
worker_mul = int(2)
#worker_mul = int(0)
if args.stage == 'chairs' or args.stage == 'things':
    datashape = [320,448]
elif '2015' in args.stage:
    datashape = [256,768]
elif 'sintel' in args.stage:
    datashape = [320,576]
else: 
    print('error')
    exit(0)

## dataloader
from dataloader import robloader as dr

if args.stage == 'chairs' or 'sintel' in args.stage:
    # flying chairs
    from dataloader import chairslist as lc
    iml0, iml1, flowl0 = lc.dataloader('%s/FlyingChairs_release/data/'%args.database)
    with open('order.txt','r') as f:
        order = [int(i) for i in f.readline().split(' ')]
    with open('FlyingChairs_train_val.txt', 'r') as f:
        split = [int(i) for i in f.readlines()]
    iml0 = [iml0[i] for i in order     if split[i]==1]
    iml1 = [iml1[i] for i in order     if split[i]==1]
    flowl0 = [flowl0[i] for i in order if split[i]==1]
    loader_chairs = dr.myImageFloder(iml0,iml1,flowl0, shape = datashape)

if args.stage == 'things' or 'sintel' in args.stage:
    # flything things
    from dataloader import thingslist as lt
    iml0, iml1, flowl0 = lt.dataloader('%s/FlyingThings3D_subset/train/'%args.database)
    loader_things = dr.myImageFloder(iml0,iml1,flowl0,shape = datashape,scale=1, order=1)

# fine-tuning datasets
if args.stage == '2015train':
    from dataloader import kitti15list_train as lk15
else:
    from dataloader import kitti15list as lk15
if args.stage == 'sinteltrain':
    from dataloader import sintellist_train as ls
else:
    from dataloader import sintellist as ls
from dataloader import kitti12list as lk12
from dataloader import hd1klist as lh

if 'sintel' in args.stage:
    iml0, iml1, flowl0 = lk15.dataloader('%s/kitti_scene/training/'%args.database)
    loader_kitti15 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, noise=0)  # SINTEL
    iml0, iml1, flowl0 = lh.dataloader('%s/rob_flow/training/'%args.database)
    loader_hd1k = dr.myImageFloder(iml0,iml1,flowl0,shape=datashape, scale=0.5,order=0, noise=0)
    iml0, iml1, flowl0 = ls.dataloader('%s/rob_flow/training/'%args.database)
    loader_sintel = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=1, noise=0)
if '2015' in args.stage:
    iml0, iml1, flowl0 = lk12.dataloader('%s/data_stereo_flow/training/'%args.database)
    loader_kitti12 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, prob=0.5)
    iml0, iml1, flowl0 = lk15.dataloader('%s/kitti_scene/training/'%args.database)
    loader_kitti15 = dr.myImageFloder(iml0,iml1,flowl0, shape=datashape, scale=1, order=0, prob=0.5)  # KITTI

if args.stage=='chairs':
    data_inuse = torch.utils.data.ConcatDataset([loader_chairs]*100) 
elif args.stage=='things':
    data_inuse = torch.utils.data.ConcatDataset([loader_things]*100) 
elif '2015' in args.stage:
    data_inuse = torch.utils.data.ConcatDataset([loader_kitti15]*50+[loader_kitti12]*50)
    for i in data_inuse.datasets:
        i.black = True
        i.cover = True
elif 'sintel' in args.stage:
    data_inuse = torch.utils.data.ConcatDataset([loader_kitti15]*200+[loader_hd1k]*40 + [loader_sintel]*150 + [loader_chairs]*2 + [loader_things])
    for i in data_inuse.datasets:
        i.black = True
        i.cover = True
else:
    print('error')
    exit(0)



## Stereo data
#from dataloader import stereo_kittilist15 as lks15
#from dataloader import stereo_kittilist12 as lks12
#from dataloader import MiddleburyList as lmbs
#from dataloader import listflowfile as lsfs
#
#def disparity_loader_sf(path):
#    from utils.util_flow import readPFM as rp
#    out =  rp(path)[0]
#    shape = (out.shape[0], out.shape[1], 1)
#    out = np.concatenate((-out[:,:,np.newaxis],np.zeros(shape),np.ones(shape)),-1)
#    return out
#
#def disparity_loader_mb(path):
#    from utils.util_flow import readPFM as rp
#    out =  rp(path)[0]
#    mask = np.asarray(out!=np.inf,float)[:,:,np.newaxis]
#    out[out==np.inf]=0
#
#    shape = (out.shape[0], out.shape[1], 1)
##    out = np.concatenate((-out[:,:,np.newaxis],np.zeros(shape),np.ones(shape)),-1)
#    out = np.concatenate((-out[:,:,np.newaxis],np.zeros(shape),mask),-1)
#    return out
#
#def disparity_loader(path):
##    from utils.util_flow import readPFM as rp
##    out =  rp(path)[0]
#    from PIL import Image 
#    out = Image.open(path)
#    out = np.ascontiguousarray(out,dtype=np.float32)/256
#    mask = np.asarray(out>0,float)[:,:,np.newaxis]
#
#    shape = (out.shape[0], out.shape[1], 1)
##    out = np.concatenate((-out[:,:,np.newaxis],np.zeros(shape),np.ones(shape)),-1)
#    out = np.concatenate((-out[:,:,np.newaxis],np.zeros(shape),mask),-1)
#    return out
##iml0, iml1, flowl0, _, _, _ = lks15.dataloader('%s/kitti_scene/training/'%args.database, typ='trainval')
##loader_stereo_15 = dr.myImageFloder(iml0,iml1,flowl0,shape = datashape,scale=1, order=0, prob=0.5,dploader=disparity_loader)
##iml0, iml1, flowl0, _, _, _ = lks12.dataloader('%s/data_stereo_flow/training/'%args.database)
##loader_stereo_12 = dr.myImageFloder(iml0,iml1,flowl0,shape = datashape,scale=1, order=0, prob=0.5,dploader=disparity_loader)
##iml0, iml1, flowl0, _, _, _ = lmbs.dataloader('%s/mb-ex-training/'%args.database, res='F')
##loader_stereo_mb = dr.myImageFloder(iml0,iml1,flowl0,shape = datashape,scale=0.5, order=1, prob=0.5,dploader=disparity_loader_mb)
##iml0, iml1, flowl0, _, _, _,_,_ = lsfs.dataloader('%s/sceneflow/'%args.database)
##loader_stereo_sf = dr.myImageFloder(iml0,iml1,flowl0,shape = datashape,scale=1, order=1, dploader=disparity_loader_sf)

#data_inuse = torch.utils.data.ConcatDataset([loader_stereo_15]*75+[loader_stereo_12]*75+[loader_stereo_mb]*600+[loader_stereo_sf])
#data_inuse = torch.utils.data.ConcatDataset([loader_stereo_15]*50+[loader_stereo_12]*50+[loader_stereo_mb]*600+[loader_chairs])
#data_inuse = torch.utils.data.ConcatDataset([loader_chairs]*2 + [loader_things] +[loader_stereo_15]*300+[loader_stereo_12]*300) # stereo transfer
#data_inuse = torch.utils.data.ConcatDataset([loader_stereo_15]*20+[loader_stereo_12]*20) # stereo transfer
#data_inuse = torch.utils.data.ConcatDataset([loader_kitti15]*20+[loader_kitti12]*20+[loader_stereo_15]*20+[loader_stereo_12]*20)
print('%d batches per epoch'%(len(data_inuse)//batch_size))

#TODO
model = VCN([batch_size//ngpus]+data_inuse.datasets[0].shape[::-1], md=[int(4*(args.maxdisp/256)), 4,4,4,4], fac=args.fac)
model = nn.DataParallel(model)
model.cuda()

total_iters = 0
mean_L=[[0.33,0.33,0.33]]
mean_R=[[0.33,0.33,0.33]]
if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}

    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
    if args.retrain == 'true':
        print('re-training')
    else:
        with open('./iter_counts-%d.txt'%int(args.logname.split('-')[-1]), 'r') as f:
            total_iters = int(f.readline())
        print('resuming from %d'%total_iters)
        mean_L=pretrained_dict['mean_L']
        mean_R=pretrained_dict['mean_R']


print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), amsgrad=False)

def train(imgL,imgR,flowl0):
        model.train()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        flowl0 = Variable(torch.FloatTensor(flowl0))

        imgL, imgR, flowl0 = imgL.cuda(), imgR.cuda(), flowl0.cuda()
        mask = (flowl0[:,:,:,2] == 1) & (flowl0[:,:,:,0].abs() < args.maxdisp) & (flowl0[:,:,:,1].abs() < (args.maxdisp//args.fac))
        mask.detach_(); 

        # rearrange inputs
        groups = []
        for i in range(ngpus):
            groups.append(imgL[i*batch_size//ngpus:(i+1)*batch_size//ngpus])
            groups.append(imgR[i*batch_size//ngpus:(i+1)*batch_size//ngpus])

        # forward-backward
        optimizer.zero_grad()
        output = model(torch.cat(groups,0), [flowl0,mask])
        loss = output[-2].mean()
        loss.backward()
        optimizer.step()

        vis = {}
        vis['output2'] = output[0].detach().cpu().numpy()
        vis['output3'] = output[1].detach().cpu().numpy()
        vis['output4'] = output[2].detach().cpu().numpy()
        vis['output5'] = output[3].detach().cpu().numpy()
        vis['output6'] = output[4].detach().cpu().numpy()
        vis['oor'] = output[6][0].detach().cpu().numpy()
        vis['gt'] = flowl0[:,:,:,:].detach().cpu().numpy()
        if mask.sum():
            vis['AEPE'] = realEPE(output[0].detach(), flowl0.permute(0,3,1,2).detach(),mask,sparse=False)
        vis['mask'] = mask
        return loss.data,vis

def adjust_learning_rate(optimizer, total_iters):
    if lr_schedule == 'slong':
        if total_iters < 200000:
            lr = baselr
        elif total_iters < 300000:
            lr = baselr/2.
        elif total_iters < 400000:
            lr = baselr/4.
        elif total_iters < 500000:
            lr = baselr/8.
        elif total_iters < 600000:
            lr = baselr/16.
    if lr_schedule == 'slong_ours':
        if total_iters < 70000:
            lr = baselr
        elif total_iters < 130000:
            lr = baselr/2.
        elif total_iters < 190000:
            lr = baselr/4.
        elif total_iters < 240000:
            lr = baselr/8.
        elif total_iters < 290000:
            lr = baselr/16.
    if lr_schedule == 'slong_pwc':
        if total_iters < 400000:
            lr = baselr
        elif total_iters < 600000:
            lr = baselr/2.
        elif total_iters < 800000:
            lr = baselr/4.
        elif total_iters < 1000000:
            lr = baselr/8.
        elif total_iters < 1200000:
            lr = baselr/16.
    if lr_schedule == 'sfine_pwc':
        if total_iters < 1400000:
            lr = baselr/10.
        elif total_iters < 1500000:
            lr = baselr/20.
        elif total_iters < 1600000:
            lr = baselr/40.
        elif total_iters < 1700000:
            lr = baselr/80.
    if lr_schedule == 'sfine':
        if total_iters < 700000:
            lr = baselr/10.
        elif total_iters < 750000:
            lr = baselr/20.
        elif total_iters < 800000:
            lr = baselr/40.
        elif total_iters < 850000:
            lr = baselr/80.
    if lr_schedule == 'rob_ours':
        if total_iters < 30000:
            lr = baselr
        elif total_iters < 40000:
            lr = baselr / 2.
        elif total_iters < 50000:
            lr = baselr / 4.
        elif total_iters < 60000:
            lr = baselr / 8.
        elif total_iters < 70000:
            lr = baselr / 16.
        elif total_iters < 100000:
            lr = baselr
        elif total_iters < 110000:
            lr = baselr / 2.
        elif total_iters < 120000:
            lr = baselr / 4.
        elif total_iters < 130000:
            lr = baselr / 8.
        elif total_iters < 140000:
            lr = baselr / 16.
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# get global counts                
with open('./iter_counts-%d.txt'%int(args.logname.split('-')[-1]), 'w') as f:
    f.write('%d'%total_iters)

def main():
    TrainImgLoader = torch.utils.data.DataLoader(
         data_inuse, 
         batch_size= batch_size, shuffle= True, num_workers=worker_mul*batch_size, drop_last=True, worker_init_fn=_init_fn, pin_memory=True)
    log = logger.Logger(args.savemodel, name=args.logname)
    start_full_time = time.time()
    global total_iters

    for epoch in range(1, args.epochs+1):
        total_train_loss = 0
        total_train_aepe = 0

        # training loop
        for batch_idx, (imgL_crop, imgR_crop, flowl0) in enumerate(TrainImgLoader):
            if batch_idx % 100 == 0:
                adjust_learning_rate(optimizer,total_iters)

            if total_iters < 1000:
                # subtract mean
                mean_L.append( np.asarray(imgL_crop.mean(0).mean(1).mean(1)) )
                mean_R.append( np.asarray(imgR_crop.mean(0).mean(1).mean(1)) )
            imgL_crop -= torch.from_numpy(np.asarray(mean_L).mean(0)[np.newaxis,:,np.newaxis, np.newaxis]).float()
            imgR_crop -= torch.from_numpy(np.asarray(mean_R).mean(0)[np.newaxis,:,np.newaxis, np.newaxis]).float()

            start_time = time.time() 
            loss,vis = train(imgL_crop,imgR_crop, flowl0)
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
            total_train_aepe += vis['AEPE']

            if total_iters %10 == 0:
                log.scalar_summary('train/loss_batch',loss, total_iters)
                log.scalar_summary('train/aepe_batch',vis['AEPE'], total_iters)
            if total_iters %100 == 0:
                log.image_summary('train/left',imgL_crop[0:1],total_iters)
                log.image_summary('train/right',imgR_crop[0:1],total_iters)
                log.histo_summary('train/pred_hist',vis['output2'], total_iters)
                if len(np.asarray(vis['gt']))>0:
                    log.histo_summary('train/gt_hist',np.asarray(vis['gt']), total_iters)
                gu = vis['gt'][0,:,:,0]; gv = vis['gt'][0,:,:,1]
                gu = gu*np.asarray(vis['mask'][0].float().cpu());  gv = gv*np.asarray(vis['mask'][0].float().cpu())
                mask = vis['mask'][0].float().cpu()
                log.image_summary('train/gt0', flow_to_image(np.concatenate((gu[:,:,np.newaxis],gv[:,:,np.newaxis],mask[:,:,np.newaxis]),-1))[np.newaxis],total_iters)
                log.image_summary('train/output2',flow_to_image(vis['output2'][0].transpose((1,2,0)))[np.newaxis],total_iters)
                log.image_summary('train/output3',flow_to_image(vis['output3'][0].transpose((1,2,0)))[np.newaxis],total_iters)
                log.image_summary('train/output4',flow_to_image(vis['output4'][0].transpose((1,2,0)))[np.newaxis],total_iters)
                log.image_summary('train/output5',flow_to_image(vis['output5'][0].transpose((1,2,0)))[np.newaxis],total_iters)
                log.image_summary('train/output6',flow_to_image(vis['output6'][0].transpose((1,2,0)))[np.newaxis],total_iters)
                log.image_summary('train/oor',vis['oor'][np.newaxis],total_iters)
                torch.cuda.empty_cache()
            total_iters += 1
            # get global counts                
            with open('./iter_counts-%d.txt'%int(args.logname.split('-')[-1]), 'w') as f:
                f.write('%d'%total_iters)

            if (total_iters + 1)%2000==0:
                #SAVE
                savefilename = args.savemodel+'/'+args.logname+'/finetune_'+str(total_iters)+'.tar'
                save_dict = model.state_dict()
                save_dict = collections.OrderedDict({k:v for k,v in save_dict.items() if ('flow_reg' not in k or 'conv1' in k) and ('grid' not in k)})
                torch.save({
                    'iters': total_iters,
                    'state_dict': save_dict,
                    'train_loss': total_train_loss/len(TrainImgLoader),
                    'mean_L': mean_L,
                    'mean_R': mean_R,
                }, savefilename)

        log.scalar_summary('train/loss',total_train_loss/len(TrainImgLoader), epoch)
        log.scalar_summary('train/aepe',total_train_aepe/len(TrainImgLoader), epoch)
      

        
    print('full finetune time = %.2f HR' %((time.time() - start_full_time)/3600))
    print(max_epo)


if __name__ == '__main__':
    main()
