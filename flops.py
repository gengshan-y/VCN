import torch
from thop import profile
import pdb
from models.PWCNet import PWCDCNet
from models.VCN import VCN

vcn = VCN([1, 1280,384])
pwcnet = PWCDCNet([1, 1280,384])
inp = torch.rand(2,3,384,1280)

f_pwc, p_pwc = profile(pwcnet, input_size=(2, 3, 384,1280), device='cuda')
f_vcn, p_vcn = profile(vcn, input_size=(2, 3, 384,1280), device='cuda')
print('PWCNet: \tflops(G)/params(M):%.1f/%.2f'%(f_pwc/1e9,p_pwc/1e6))
print('VCN:  \t\tflops(G)/params(M):%.1f/%.2f'%(f_vcn/1e9,p_vcn/1e6))



