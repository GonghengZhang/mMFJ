import matplotlib.pyplot as plt
import numpy as np
from obspy.signal import filter

def moving_average(data, window_size):
    if window_size == 0:
        return data
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')

def SpaceMean(data,r,r_window, method = 'fast'):
    if r_window == 0:
        return data
    row = np.size(data,0);col = np.size(data,1)
    SMdata = np.zeros_like(data)
    
    if method == 'low':
        r_window = r_window*1000 # r window size convert to meters
        for ii in range(row):
            index = np.where((r <= r[ii] + 0.5*r_window) & (r >= r[ii] - 0.5*r_window))[0]
            for jj in range(col):
                SMdata[ii,jj] = np.mean(data[index,jj]) 
                
    if method == 'fast':     
        for jj in range(col):
            SMdata[:,jj] = moving_average(data[:,jj],r_window)      
    
    return SMdata
    
def K_filtering(data,f,r,vmin,vmax):
    num = np.size(data,0)
    df = f[1] - f[0]
    for ii in range(num):
        data[ii,:] = filter.bandpass(data[ii,:],r[ii]/vmax,r[ii]/vmin,1/df,corners = 4,zerophase=True)
        data[ii,:] = data[ii,:]/np.max(np.abs(data[ii,:]))
        data[ii,:] = data[ii,:] - np.mean(data[ii,:])
    
    return data

def SmoothFCdata(fc, f_size = 10, c_size = 10):
    num = np.size(fc,0) # freq num
    size = np.size(fc,1) # velocity num
    
    for ii in range(size):
        fc[:,ii] = moving_average(fc[:,ii],f_size)      
    for ii in range(num):
        fc[ii,:] = moving_average(fc[ii,:],c_size)
        fc[ii,:] = fc[ii,:]/np.max(np.abs(fc[ii,:]))
        
    fc = np.maximum(fc, 0)

    return fc

def ShowFJSpectrum(fjData,freq,velocity,savepath = None,subname = 'test',maxPeriod = 100,saveData = False, theoreticalfile = None):
    fjData = SmoothFCdata(fjData.T,f_size = 1, c_size = 1);fjData = fjData.T
    if saveData:
        saveName = savepath + subname + '_fc.txt';np.savetxt(saveName,fjData)
        saveName = savepath + 'c.txt';np.savetxt(saveName,velocity)
        saveName = savepath + 'f.txt';np.savetxt(saveName,freq)

    fig = plt.figure()
    plt.contourf(freq, velocity/1000, fjData, 50, vmin = 0,vmax=1.0, cmap = 'jet')
    if theoreticalfile != None:
        theodisp = np.loadtxt(theoreticalfile)
        if theodisp[0,1]> 1000:
            theodisp[:,1] = theodisp[:,1]/1000
        plt.plot(theodisp[:,0],theodisp[:,1],'w.',markersize=2)
    plt.xlim([np.min(freq),np.max(freq)]);plt.ylim([np.min(velocity/1000),np.max(velocity/1000)])
    plt.xlabel('Frequency (Hz)');plt.ylabel('Phase Velocity (km/s)')
    if savepath == None:
        plt.show()
    else:
        saveName = savepath + 'fc_' + subname + '.png'
        fig.savefig(saveName,format = 'png', dpi = 300)
    plt.close()
        
    fig = plt.figure()
    T = [1.0/x for x in list(freq)]; T.reverse()
    fjData = np.fliplr(fjData)
    plt.contourf(T, velocity/1000, fjData, 50, vmin = 0,vmax=1, cmap = 'jet')
    if theoreticalfile != None:
        theodisp = np.loadtxt(theoreticalfile)
        if theodisp[0,1]> 1000:
            theodisp[:,1] = theodisp[:,1]/1000
        plt.plot(1/theodisp[:,0],theodisp[:,1],'w.',markersize=2)
    plt.xlim((np.min(T),maxPeriod))
    plt.ylim([np.min(velocity/1000),np.max(velocity/1000)])

    plt.xlabel('Period (s)')
    plt.ylabel('Phase Velocity (km/s)')
    if savepath == None:
        plt.show()
    else:
        saveName = savepath + 'tc_' + subname + '.png'
        fig.savefig(saveName,format = 'png', dpi = 300)
        
    plt.close()

def LoadCCF(gatherfile,maxf):
    gatherInfo = np.load(gatherfile)
    f = gatherInfo['f'];r = gatherInfo['r']
    f = f - f[0];index = np.where((f<=maxf) &(f >0))[0];f = f[index]
    ncfs = gatherInfo['ncfs'][:,index]
    indx = np.argsort(r);r = r[indx];ncfs = np.real(ncfs[indx,:])
    return f,r,ncfs