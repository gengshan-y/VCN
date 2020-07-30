## point $datapath to the folder of your images
datapath=./dataset/IIW/
modelname=things
i=239999
CUDA_VISIBLE_DEVICES=1 python submission.py --dataset kitticlip --datapath $datapath/   --outdir ./weights/$modelname/ --loadmodel ./weights/$modelname/finetune_$i.tar  --maxdisp 256 --fac 1
