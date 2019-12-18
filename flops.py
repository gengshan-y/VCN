import torch
from thop import profile
import pdb
from models.PWCNet import PWCDCNet
from models.VCN import VCN
#from models.VCN_small import VCN

vcn = VCN([1, 1280,384])
pwcnet = PWCDCNet([1, 1280,384])

f_pwc, p_pwc = profile(pwcnet, input_size=(2, 3, 384,1280), device='cuda')
f_vcn, p_vcn = profile(vcn, input_size=(2, 3, 384,1280), device='cuda')
print('PWCNet: \tflops(G)/params(M):%.1f/%.2f'%(f_pwc/1e9,p_pwc/1e6))
print('VCN:  \t\tflops(G)/params(M):%.1f/%.2f'%(f_vcn/1e9,p_vcn/1e6))


#from models.conv4d import butterfly4D, sepConv4dBlock, sepConv4d
#
#model = torch.nn.Conv2d(30,30, (3,3), stride=1, padding=(1, 1), bias=False)
#f, p = profile(model, input_size=(9*9,30,64,128), device='cuda')
#
##model = torch.nn.Conv3d(30,30, (1,3,3), stride=1, padding=(0, 1, 1), bias=False)
##f, p = profile(model, input_size=(1,30,9*9,64,128), device='cuda')
#
#
###model = butterfly4D(64, 12,withbn=True,full=True)
###model = sepConv4dBlock(16,16,with_bn=True, stride=(1,1,1),full=True)
##model = sepConv4d(20, 20, (1,1,1), with_bn=True, full=True)
##f, p = profile(model, input_size=(1,20,9,9,64,128), device='cuda')
#
##model = torch.nn.Conv2d(256,256,(3,3),padding=(1,1))
##f, p = profile(model, input_size=(1,256,64,128), device='cuda')
#
#print('flops(G)/params(M):%.1f/%.2f'%(f/1e9,p/1e6))
