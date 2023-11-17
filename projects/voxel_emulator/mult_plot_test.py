import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pycorr import TwoPointCorrelationFunction


cosmo = [0,1,2,3,4]
data = '/pscratch/sd/t/tsfraser/new_sunbird/sunbird/data/clustering/abacus/voidprior/voxel_voids/'
for c in cosmo:
    print('DO YOU WORK YOU DUMB SACK OF SHIT?')
    old_fname = f'/pscratch/sd/t/tsfraser/new_sunbird/sunbird/data/clustering/abacus/voidprior/voxel_voids/voxel_multipoles_Rs10_RvMed_c{int(c):03}_cut20.npy'
    new_fname = f'/pscratch/sd/t/tsfraser/new_sunbird/sunbird/data/clustering/abacus/voidprior/voxel_voids/voxel_multipoles_AP_Rs10_c{int(c):03}.npy'

    print(np.load(old_fname,allow_pickle=True)[()])
    old_res = np.load(old_fname,allow_pickle=True)[()]
    s,mult_old = old_res['s'],old_res['multipoles']

    new_res = np.load(new_fname,allow_pickle=True)[()]
    s,mult_new = new_res['s'],new_res['multipoles']

    print(mult_new.shape)
    plt.plot(s,mult_old[0,0,1,:],linestyle='solid',label=f'c{c:03}, old result')
    plt.plot(s,mult_new[0,0,1,:],linestyle='dashed',label=f'c{c:03}, new result')
    

plt.legend()
plt.show()
   


