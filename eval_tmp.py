import os
from matplotlib import pyplot as plt
import numpy as np
import sys
sys.path.insert(0,'utils')
from utils.flowlib import flow_to_image, read_flow, compute_color, visualize_flow
from utils.io import mkdir_p
import pdb
import glob

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', default='/data/ptmodel/',
                    help='database')
parser.add_argument('--vis', default='no',
                    help='database')
parser.add_argument('--dataset', default='2015',
                    help='database')
args = parser.parse_args()

aepe_s = []
fall_s64 = []
fall_s32 = []
fall_s16 = []
fall_s8 = []
fall_s = []
oor_tp = []
oor_fp = []

# dataloader
if args.dataset == '2015':
    #from dataloader import kitti15list as DA
    #from dataloader import kitti15list_val_lidar as DA
    from dataloader import kitti15list_val as DA
    datapath = '/ssd/kitti_scene/training/'
elif args.dataset == '2015test':
    from dataloader import kitti15list as DA
    datapath = '/ssd/kitti_scene/testing/'
elif args.dataset == 'kitticlip':
    from dataloader import kitticliplist as DA
    #datapath = '/ssd/rob_flow/test/image_2/Kitti2015_000140_'
    datapath = '/data/gengshay/KITTI_png/2011_09_30/2011_09_30_drive_0028_sync/image_02/data/'
elif args.dataset == 'tumclip':
    from dataloader import kitticliplist as DA
    datapath = '/data/gengshay/TUM/rgbd_dataset_freiburg1_plant/rgb/'
elif args.dataset == '2012':
    from dataloader import kitti12list as DA
    datapath = '/ssd/data_stereo_flow/training/'
elif args.dataset == '2012test':
    from dataloader import kitti12list as DA
    datapath = '/ssd/data_stereo_flow/testing/'
elif args.dataset == 'mb':
    from dataloader import mblist as DA
    datapath = '/ssd/rob_flow/training/'
elif args.dataset == 'sintel':
    #from dataloader import sintellist as DA
    from dataloader import sintellist_val as DA
    #from dataloader import sintellist_clean as DA
    datapath = '/ssd/rob_flow/training/'
elif args.dataset == 'hd1k':
    from dataloader import hd1klist as DA
    datapath = '/ssd/rob_flow/training/'
elif args.dataset == 'mbtest':
    from dataloader import mblist as DA
    datapath = '/ssd/rob_flow/test/'
elif args.dataset == 'sinteltest':
    from dataloader import sintellist as DA
    datapath = '/ssd/rob_flow/test/'
elif args.dataset == 'chairs':
    from dataloader import chairslist as DA
    datapath = '/ssd/FlyingChairs_release/data/'
test_left_img, test_right_img ,flow_paths= DA.dataloader(datapath)

#pdb.set_trace()
#with open('/data/gengshay/PWC-Net/Caffe/sintel_test1.txt','w') as f:
#    for i in test_left_img:
#        f.write(i+'\n')
#
#with open('/data/gengshay/PWC-Net/Caffe/sintel_test2.txt','w') as f:
#    for i in test_right_img:
#        f.write(i+'\n')
#
#with open('/data/gengshay/PWC-Net/Caffe/sintel_testout.txt','w') as f:
#    for i in test_left_img:
#        f.write('/data/ptmodel/pwcnet-1/sintel/%s.flo'%(i.split('/')[-1].split('.')[0])+'\n')
#exit()
if args.dataset == 'chairs':
    with open('FlyingChairs_train_val.txt', 'r') as f:
        split = [int(i) for i in f.readlines()]
    test_left_img = [test_left_img[i]   for i,flag in enumerate(split) if flag==2]
    test_right_img = [test_right_img[i] for i,flag in enumerate(split) if flag==2]
    flow_paths = [flow_paths[i]         for i,flag in enumerate(split) if flag==2]

#pdb.set_trace()
#test_left_img = [i for i in test_left_img if 'clean' in i]
#test_right_img = [i for i in test_right_img if 'clean' in i]
#flow_paths = [i for i in flow_paths if 'clean' in i]
    
#for i,gtflow_path in enumerate(sorted(flow_paths)):
for i,gtflow_path in enumerate(flow_paths):
    #if not 'Sintel_clean_cave_4_10' in gtflow_path:
    #    continue
    #if i%10!=1:
    #    continue
    num = gtflow_path.split('/')[-1].strip().replace('flow.flo','img1.png')
    if not 'test' in args.dataset and not 'clip' in args.dataset:
        gtflow = read_flow(gtflow_path)
    num = num.replace('jpg','png')
    flow = read_flow('%s/%s/%s'%(args.path,args.dataset,num))
    if args.vis == 'yes':
        #flowimg = flow_to_image(flow)
        flowimg = flow_to_image(flow)*np.linalg.norm(flow[:,:,:2],2,2)[:,:,np.newaxis]/100./255.
        #gtflowimg = compute_color(gtflow[:,:,0]/20, gtflow[:,:,1]/20)/255.
        #flowimg = compute_color(flow[:,:,0]/20, flow[:,:,1]/20)/255.
        mkdir_p('%s/%s/flowimg'%(args.path,args.dataset))
        plt.imsave('%s/%s/flowimg/%s'%(args.path,args.dataset,num), flowimg)
        if 'test' in args.dataset or 'clip' in args.dataset:
            continue
        gtflowimg = flow_to_image(gtflow)
        mkdir_p('%s/%s/gtimg'%(args.path,args.dataset))
        plt.imsave('%s/%s/gtimg/%s'%(args.path,args.dataset,num), gtflowimg)

    mask = gtflow[:,:,2]==1

    ## occlusion
    #H,W,_ = gtflow.shape
    #xx = np.tile(np.asarray(range(0, W))[np.newaxis:],(H,1))
    #yy = np.tile(np.asarray(range(0, H))[:,np.newaxis],(1,W))
    #occmask = np.logical_or( np.logical_or(xx + gtflow[:,:,0] <0, xx + gtflow[:,:,0]>W-1),
    #                         np.logical_or(yy + gtflow[:,:,1] <0, yy + gtflow[:,:,1]>H-1))
    #mask = np.logical_and(mask,~occmask)
    


   # if args.dataset == 'mb':
   #     ##TODO
   #     mask = np.logical_and(np.logical_and(np.abs(gtflow[:,:,0]) < 16,np.abs(gtflow[:,:,1]) < 16), mask)
    gtflow = gtflow[:,:,:2]
    flow = flow[:,:,:2]

    epe = np.sqrt(np.power(gtflow - flow,2).sum(-1))[mask]
    gt_mag = np.sqrt(np.power(gtflow,2).sum(-1))[mask] 

    
    #aepe_s.append( epe.mean() )
    #fall_s.append( np.sum(np.logical_and(epe > 3, epe/gt_mag > 0.05)) / float(epe.size) )

    clippx = [0,1000]
    inrangepx = np.logical_and((np.abs(gtflow)>=clippx[0]).sum(-1), (np.abs(gtflow)<clippx[1]).prod(-1))[mask]
    if os.path.isfile('%s/%s/%s'%(args.path,args.dataset,num.replace('png','npy'))):
        isoor = np.load('%s/%s/%s'%(args.path,args.dataset,num.replace('png','npy')))
        gtoortp = mask*((np.abs(gtflow)>clippx).sum(-1)>0)
        gtoorfp = mask*((np.abs(gtflow)>clippx).sum(-1)==0)
        oor_tp.append(isoor[gtoortp])
        oor_fp.append(isoor[gtoorfp])
    if args.vis == 'yes' and 'test' not in args.dataset:
        epeimg = np.sqrt(np.power(gtflow - flow,2).sum(-1))*(mask*(np.logical_and((np.abs(gtflow)>=clippx[0]).sum(-1), (np.abs(gtflow)<clippx[1]).prod(-1))).astype(float))
        mkdir_p('%s/%s/epeimg'%(args.path,args.dataset))
        plt.imsave('%s/%s/epeimg/%s'%(args.path,args.dataset,num), epeimg, vmax=32)

    aepe_s.append( epe[inrangepx] )
    fall_s64.append( (epe > 64)[inrangepx])
    fall_s32.append( (epe > 32)[inrangepx])
    fall_s16.append( (epe > 16)[inrangepx])
    fall_s8.append(  (epe > 8)[inrangepx])
    fall_s.append(   np.logical_and(epe > 3, epe/gt_mag > 0.05)[inrangepx])
   # aepe_s.append( epe )
    #fall_s.append( epe[gt_mag<32] > 8)
   # fall_s.append( np.logical_and(epe > 3, epe/gt_mag > 0.05))
#    print(gtflow_path)
#for i in [np.mean(i) for i in aepe_s]:
#    print('%f'%i)
#for i in [np.mean(i) for i in fall_s]:
#    print('%f'%i)
#print('\t%.1f/%.1f/%.1f/%.1f/%.1f/%.3f'%(
#                np.mean( 100*np.concatenate(fall_s64,0)),
#                np.mean( 100*np.concatenate(fall_s32,0)),
#                np.mean( 100*np.concatenate(fall_s16,0)),
#                np.mean( 100*np.concatenate(fall_s8,0)),
#                np.mean( 100*np.concatenate(fall_s,0)),
#                np.mean( np.concatenate(aepe_s,0))) )
print('\t%.1f/%.3f'%(
                np.mean( 100*np.concatenate(fall_s,0)),
                np.mean( np.concatenate(aepe_s,0))) )
#print('\t%.1f/%.1f'%(100*np.mean( np.concatenate(oor_tp,0) ), 100*np.mean( np.concatenate(oor_fp,0) )) )
