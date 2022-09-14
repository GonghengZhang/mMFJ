import numpy as np
import ccfj
from mMFJ_util import ShowFJSpectrum,LoadCCF,SpaceMean,K_filtering
import os
import matplotlib.pyplot as plt


CCDir = './data/'
savePath =  './result/'
# reference disp file, if you don't have this file, please set the variable to None
Lovedispfile = './data/Love_disp.txt';Rayleighdispfile = './data/Rayleigh_disp.txt'
# the frequency bands shown in result
max_freq = 25;maxPeriod = 1
# the phase velocity range in the result
minV = 51;maxV = 400;numV = 450
# use modified FJ or not 
MFJ = True

# load CCF data
f,r,ZZ = LoadCCF(os.path.join(CCDir,'ZZ.npz'),max_freq)
f,r,ZR = LoadCCF(os.path.join(CCDir,'ZR.npz'),max_freq)
f,r,RZ = LoadCCF(os.path.join(CCDir,'RZ.npz'),max_freq)
f,r,RR = LoadCCF(os.path.join(CCDir,'RR.npz'),max_freq)
f,r,TT = LoadCCF(os.path.join(CCDir,'TT.npz'),max_freq)

# using K_filtering method
ZZ = K_filtering(ZZ,f,r,minV,maxV)
ZR = K_filtering(ZR,f,r,minV,maxV)
RZ = K_filtering(RZ,f,r,minV,maxV)
RR = K_filtering(RR,f,r,minV,maxV)
TT = K_filtering(TT,f,r,minV,maxV)

# using space mean method
ZZ = SpaceMean(ZZ,r,5)
ZR = SpaceMean(ZR,r,5)
RZ = SpaceMean(RZ,r,5)
RR = SpaceMean(RR,r,5)
TT = SpaceMean(TT,r,5)

plt.contourf(f,r,ZZ,cmap='jet')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Inter-station(m)')
plt.title('ZZ CCFs')
plt.show()

velocity = np.linspace(minV,maxV,numV) 
#   do the MFJ transform for ZZ component   
ZZ_FC = ccfj.fj_noise_v2(ZZ,[],r,velocity,f,tag = 'ZZ',itype = 1,func=MFJ)
ShowFJSpectrum(ZZ_FC,f,velocity,savepath = savePath,subname = 'ER0', maxPeriod = maxPeriod, theoreticalfile = Rayleighdispfile)

#   do the MFJ transform for RZ component  
RZ_FC =ccfj.fj_noise_v2(RZ,[],r,velocity,f,tag = 'RZ',itype = 0,func=MFJ)
#   do the MFJ transform for ZR component
ZR_FC =-ccfj.fj_noise_v2(ZR,[],r,velocity,f,tag = 'ZR',itype = 0,func=MFJ)
ShowFJSpectrum(np.abs(RZ_FC + ZR_FC),f,velocity,savepath = savePath,subname = 'ER1', maxPeriod = maxPeriod,theoreticalfile = Rayleighdispfile)     

#   do the MFJ transform for RR-TT component for Rayleigh wave
RT_r_FC = ccfj.fj_noise_v2(RR,TT,r,velocity,f,tag = 'RT_r',itype = 0,func=MFJ)
ShowFJSpectrum(RT_r_FC,f,velocity,savepath = savePath,subname = 'ER2', maxPeriod = maxPeriod, theoreticalfile = Rayleighdispfile)     

ShowFJSpectrum(ZZ_FC + RT_r_FC + np.abs(RZ_FC + ZR_FC),f,velocity,savepath = savePath,subname = 'ER', maxPeriod = maxPeriod, theoreticalfile = Rayleighdispfile)      
    
#   do the MFJ transform for RR-TT component for Love wave
RT_l_FC = ccfj.fj_noise_v2(RR,TT,r,velocity,f,tag = 'RT_l',itype = 0,func=MFJ)
ShowFJSpectrum(RT_l_FC,f,velocity,savepath = savePath,subname = 'El', maxPeriod = maxPeriod, theoreticalfile = Lovedispfile)

