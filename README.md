# VCN: Volumetric correspondence networks for optical flow

<img src="figs/architecture.png" width="800">

**Requirements**
- python 3.6
- pytorch 1.0.0-1.3.0
- [pytorch correlation module](https://github.com/gengshan-y/Pytorch-Correlation-extension) (optional) This gives a noticible inference time speed up. Please replace self.corr() with self.corrf() in models/VCN.py if pytorch correlation module is not installed.
- [weights files](https://drive.google.com/drive/folders/1mgadg50ti1QdwfAf6aR2v1pCx-ITsYfE?usp=sharing)

## Pre-trained models
#### KITTI
**This correspondens to the entry on the [leaderboard](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) (Fl-all=6.30%).**
##### Try on a single image

To run + visualize on KITTI-15 test set,
```
modelname=kitti-ft-trainval
i=149999
CUDA_VISIBLE_DEVICES=0 python submission.py --dataset 2015test --datapath dataset/kitti_scene/testing/   --outdir ./weights/$modelname/ --loadmodel ./weights/$modelname/finetune_$i.tar  --testres 1
```
then running point-vec.ipynb will give you flow visualizations with color and vectors as follows.

<img src="figs/kitti-test-42.png" width="300">
<img src="figs/kitti-test-42-vec.png" width="300">

##### Evaluate on KITTI-val
*To see the details of the train-val split, please scroll down to "note on train-val" and run dataloader/kitti15list_val.py, dataloader/kitti15list_train.py, dataloader/sitnellist_train.py, and dataloader/sintellist_val.py.*

To evaluate on the 40 validation images of KITTI-15 (0,5,...195), (also assuming the data is at /ssd/kitti_scene)
```
modelname=kitti-ft-trainval
i=149999
CUDA_VISIBLE_DEVICES=0 python submission.py --dataset 2015 --datapath /ssd/kitti_scene/training/   --outdir ./weights/$modelname/ --loadmodel ./weights/$modelname/finetune_$i.tar  --testres 1
python eval_tmp.py --path ./weights/$modelname/ --vis no --dataset 2015
```

To evaluate + visualize on KITTI-15 validation set,
```
python eval_tmp.py --path ./weights/$modelname/ --vis yes --dataset 2015
```
Evaluation error on 40 validation images : Fl-err = 3.9, EPE = 1.144

#### Sintel
**This correspondens to the entry on the [leaderboard](http://sintel.is.tue.mpg.de/quant?metric_id=0&selected_pass=0) (EPE-all-final = 4.404, EPE-all-clean = 2.808).**
##### Evaluate on Sintel-val

To evaluate on Sintel validation set, 
```
modelname=sintel-ft-trainval
i=67999
CUDA_VISIBLE_DEVICES=0 python submission.py --dataset sintel --datapath /ssd/rob_flow/training/   --outdir ./weights/$modelname/ --loadmodel ./weights/$modelname/finetune_$i.tar  --testres 1
python eval_tmp.py --path ./weights/$modelname/ --vis no --dataset sintel
```
Evaluation error on sintel validation images: Fl-err = 7.9, EPE = 2.351



## Measure FLOPS
```
python flops.py
```
gives

PWCNet:     flops(G)/params(M):101.6/9.37

VCN:        flops(G)/params(M):101.7/6.23

## Train the model
We follow the same stage-wise training procedure as prior work: Chairs->Things->KITTI or Chairs->Things->Sintel, but uses much lesser iterations.
If you plan to train the model and reproduce the numbers, please check out our [supplementary material](https://papers.nips.cc/paper/8367-volumetric-correspondence-networks-for-optical-flow) for the differences in hyper-parameters with FlowNet2 and PWCNet.

#### Pretrain on flying chairs and flying things
Make sure you have downloaded [flying chairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html) 
and [flying things **subset**](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html),
and placed them under the same folder, say /ssd/.

To first train on flying chairs for 140k iterations, run (assuming you have two gpus)
```
CUDA_VISIBLE_DEVICES=0,1 python main.py --maxdisp 256 --fac 1 --database /ssd/ --logname chairs-0 --savemodel /data/ptmodel/  --epochs 1000 --stage chairs --ngpus 2
```
Then we want to fine-tune on flying things for 80k iterations, resume from your pre-trained model or use our pretrained model
```
CUDA_VISIBLE_DEVICES=0,1 python main.py --maxdisp 256 --fac 1 --database /ssd/ --logname things-0 --savemodel /data/ptmodel/  --epochs 1000 --stage things --ngpus 2 --loadmodel ./weights/charis/finetune_141999.tar --retrain false
```
Note that to resume the number of iterations, put the iteration to start from in iter_counts-(your suffix).txt. In this example, I'll put 141999 in iter_counts-0.txt.
Be aware that the program reads/writes to iter_counts-(suffix).txt at training time, so you may want to use different suffix when multiple training programs are running at the same time.

#### Finetune on KITTI / Sintel
Please first download the kitti 2012/2015 flow dataset if you want to fine-tune on kitti. 
Download [rob_devkit](http://www.cvlibs.net:3000/ageiger/rob_devkit/src/flow/flow) if you want to fine-tune on sintel.

To fine-tune on KITTI, run
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --maxdisp 512 --fac 2 --database /ssd/ --logname kitti-trainval-0 --savemodel /data/ptmodel/  --epochs 1000 --stage 2015trainval --ngpus 4 --loadmodel ./weights/things/finetune_211999.tar --retrain true
```
To fine-tune on Sintel, run
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --maxdisp 448 --fac 1.4 --database /ssd/ --logname sintel-trainval-0 --savemodel /data/ptmodel/  --epochs 1000 --stage sinteltrainval --ngpus 4 --loadmodel ./weights/things/finetune_211999.tar --retrain true
```

#### Note on train-val
- To tune hyper-parameters, we use a train-val split for kitti and sintel, which is not covered by the 
above procedure. 
- For kitti we use every 5th image in the training set (0,5,10,...195) for validation, and the rest for training; while for Sintel, we manually select several sequences for validation.
- If you plan to use our split, put "--stage 2015train" or "--stage sinteltrain" for training.
- The numbers in Tab.3 of the paper is on the whole train-val set (all the data with ground-truth).
- You might find run.sh helpful to run evaluation on KITTI/Sintel.

## Acknowledgement
Thanks [ClementPinard](https://github.com/ClementPinard), [Lyken17](https://github.com/Lyken17), [NVlabs](https://github.com/NVlabs) and many others for open-sourcing their code.
- Pytorch op counter thop is modified from [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)
- Correlation module is modified from [Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension)

## Citation
```
@inproceedings{yang2019vcn,
  title={Volumetric Correspondence Networks for Optical Flow},
  author={Yang, Gengshan and Ramanan, Deva},
  booktitle={NeurIPS},
  year={2019}
}
```
