import numpy as np
import matplotlib.pyplot as plt
from pycorr import TwoPointCorrelationFunction
from pathlib import Path

f,ax = plt.subplots(1,2,figsize=(9.8,4.8),sharex =True)
#ff,axi = plt.subplots(1,2,figsize=(9.8,4.8),sharex=True)
small_dir = '/pscratch/sd/t/tsfraser/rev_test/multipoles/AbacusSummit_small_c000_ph3000/z0.500/'
los = ['x','y','z']
void_dir = '/pscratch/sd/t/tsfraser/rev_test/voids/AbacusSummit_small_c000_ph3000/z0.500/'
prefix =['NOCUT_UP','NOCUT_DN','NOCUT_BASE']
pres2 = ['_UP_','_DN_','_BASE_']
leglabel = ['ISOTROPIC SHRINK', 'ISOTROPIC EXP', 'ORIGINAL']
rescales = [0.6,1.4,1]
res_rev = [1.4,0.6,1]
#VOID_NOCUT_UP_DISC_voxel_multipoles_Rs10_Rv10_c000_ph3000_hod0_losz_cut20.npy
#ax[0].set_title('Monopoles')
#ax[1].set_title('Quadrupoles')
hists = []
for j in range(3):
    mono_arr = []
    quad_arr = []
    mono2 = []
    quad2 =[]
    vr = []
    vr2=[]
    for i in range(3):
        fffname = f'pscratch/sd/t/tsfraser/rev_test/multipoles/AbacusSummit_base_c002_ph000/z0.500/VOID_PAIRAP_DISC_voxel_multipoles_Rs10_Rv10_c002_ph000_hod0_los{los[i]}_cut20.npy'
        ffname = f'VOID_{prefix[j]}_DISC_voxel_voids_Rs10_RvMed_c000_ph3000_hod0_los{los[i]}_cut20.npy'
        fname = f'VOID_{prefix[j]}_DISC_voxel_multipoles_Rs10_Rv10_c000_ph3000_hod0_los{los[i]}_cut20.npy'
        #/pscratch/sd/t/tsfraser/rev_test/voids/AbacusSummit_small_c000_ph3000/z0.500/VOID_RESCALE_BASE_CELL_SMOOTH_DISC_voxel_voids_Rs10_RvMed_c000_ph3000_hod0_losz_cut20.npy
        ffname2 = f'VOID_RESCALE{pres2[j]}CELL_SMOOTH_DISC_voxel_voids_Rs10_RvMed_c000_ph3000_hod0_los{los[i]}_cut20.npy'
        fname2= f'VOID_RESCALE{pres2[j]}CELL_SMOOTH_DISC_voxel_multipoles_Rs10_Rv10_c000_ph3000_hod0_los{los[i]}_cut20.npy'
        void_r = np.load(Path(void_dir,ffname),allow_pickle=True)[()]["radii"]
        void_r2 = np.load(Path(void_dir,ffname2),allow_pickle=True)[()]["radii"]

        void_p = np.load(Path(void_dir,ffname),allow_pickle=True)[()]["positions"].max()
        print(rescales[j],void_p,'greatest coord','and greatest r:', void_r.max())
        #axi[i].hist(void_r,bins=np.linspace(5,75,
        histo,bin_edges = np.histogram(void_r,bins=np.linspace(5,75,30), density=True)
        his2,bin_eges2 = np.histogram(void_r2,bins=np.linspace(5,75,30),density=False)
        #axi[i].plot(bin_edges[:-1]+(bin_edges[0]+bin_edges[1])*0.5,histo)
        #axi[i].set_yscale("log")       
#print(void_r
                       #)
        vr.append(void_r)
        vr2.append(void_r2)
        #print(vr)
        res = TwoPointCorrelationFunction.load(Path(small_dir,fname))
        res2 =TwoPointCorrelationFunction.load(Path(small_dir,fname2))
        s,multipoles = res(ells=(0, 2, 4), return_sep=True)
        s2,multipoles2 = res2(ells=(0,2,4),return_sep=True)
        print(np.shape(multipoles))

        #ax[i,0].plot(s,s**2*multipoles[0])
        #ax[i,1].plot(s,s**2*multipoles[1])
        mono_arr.append(multipoles[0])
        quad_arr.append(multipoles[1])
        mono2.append(multipoles2[0])
        quad2.append(multipoles2[1])
        #ax[i,0].set_ylabel(f'LOS: {los[i]}')

    
    avg_mono = np.mean(np.asarray(mono_arr),axis=0)
    avg_quad = np.mean(np.asarray(quad_arr),axis=0)

    avg_mono2 = np.mean(np.asarray(mono2),axis=0)
    avg_quad2 = np.mean(np.asarray(quad2),axis=0)
    cs = ['#D81B60','#1E88E5','#004D40','#FFC107']    
    #ax[0].plot(s,avg_mono2,label=prefix[j],lw=2)
    #ax[1].plot(s,avg_quad2,lw=2)
    #ax[0].set_xlabel('Separation [Mpc/h]')
    #ax[1].set_xlabel('Separation [Mpc/h]')
    #ax[0].legend()
    #plt.show()
    print(np.concatenate(vr,axis=0))
    histi,bedges = np.histogram((np.concatenate(vr,axis=0).flatten()),bins =np.linspace(5,75,30)/rescales[j],density=False)
    histi2,bedges2 = np.histogram((np.concatenate(vr2,axis=0).flatten()),bins=np.linspace(5,75,30)/rescales[j],density=False)  
    hists.append(histi2)
    ax[0].plot((bedges[:-1]+0.5*(-bedges[0]+bedges[1]))*rescales[j],histi,lw=3,label=leglabel[j],color=cs[j])
    #axi[0].legend()
    ax[0].set_yscale('log')
    ls = ['dotted','dashed','solid']
    marker=['*','^','o']
    hatch =['O','**','//']
    
    ax[1].stairs(histi2,(bedges2[0:]+0.5*(-bedges2[0]+bedges2[1]))*rescales[j],lw=1,hatch=hatch[j],label=leglabel[j],alpha=0.6,color=cs[j])
    ax[0].legend(loc='best')
    ax[1].legend()
    ax[0].set_title('Fixed cellSize and smoothing scale')
    ax[1].set_title('Rescaling cellSize and smoothing scale')
    ax[1].set_yscale('log')
plt.legend()
plt.show()

print('=====================')
print('=== TESTING PERCENT VARIATION ===')
print(hists[0])
print((hists[0]-hists[1]))
print((hists[0]-hists[2]))
#plt.clf()
ffff,aaaa = plt.subplots(2,figsize=(10,5))
plt.title('Comparing histograms')
aaaa[0].set_ylabel('(V1-V0)/V1')
aaaa[1].set_ylabel('(V1-V2)/V1')
aaaa[0].set_xlabel(' Radius [Mpc]')
aaaa[1].set_xlabel(' Radius [Mpc]')
aaaa[0].plot((bedges2[:-1]+0.5*(bedges2[1]-bedges2[0])), (hists[2]-hists[1])/hists[2],'r-')
aaaa[1].plot((bedges2[:-1]+0.5*(bedges2[1]-bedges2[0])), (hists[2]-hists[0])/hists[2],'k-')
plt.show()

original_dir='/pscratch/sd/t/tsfraser/new_sunbird/sunbird/data/clustering/abacus/voidprior/voxel_voids/'
large_dir = '/pscratch/sd/t/tsfraser/rev_test/multipoles/AbacusSummit_base_c002_ph000/z0.500/'

original_fname = 'voxel_multipoles_Rs10_RvMed_c002_cut20.npy'

original_data = np.load(Path(original_dir,original_fname),allow_pickle=True)[()]#['multipoles']

monos = original_data['multipoles'][0,:,0,:]
quads = original_data['multipoles'][0,:,1,:]



f,axs = plt.subplots(1,2,figsize=(9,5),sharex=True)
axs[0].set_title('Monopoles')
axs[1].set_title('Quadrupoles')
pres = ['QSCALE','PAIRAP','NOAP']
cs = ['#D81B60','#1E88E5','#004D40','#FFC107']   
for j in range(3):
    mono_arr = []
    quad_arr = []

    for i in range(3):
        #fffname = f'pscratch/sd/t/tsfraser/rev_test/multipoles/AbacusSummit_base_c002_ph000/z0.500/VOID_PAIRAP_DISC_voxel_multipoles_Rs10_Rv10_c002_ph000_hod0_los{los[i]}_cut20.npy'

        fname = f'VOID_{pres[j]}_DISC_voxel_multipoles_Rs10_Rv10_c002_ph000_hod0_los{los[i]}_cut20.npy'#'VOID_{prefix[j]}_DISC_voxel_multipoles_Rs10_Rv10_c000_ph3000_hod0_los{los[i]}_cut20.npy'
        res = TwoPointCorrelationFunction.load(Path(large_dir,fname))
        s,multipoles = res(ells=(0, 2, 4), return_sep=True)
        print(np.shape(multipoles))
        print(multipoles[0])
        #axs[i,0].plot(s,s**2*multipoles[0])
        #axs[i,1].plot(s,s**2*multipoles[1])
        mono_arr.append(multipoles[0])
        quad_arr.append(multipoles[1])

        #axs[i,0].set_ylabel(f'LOS: {los[i]}')
        #if j==1:
        #   axs[0].plot(original_data['s'],original_data['s']**2*monos[i,:],label='training data',lw=2)
        #   axs[1].plot(original_data['s'],original_data['s']**2*quads[i,:],lw=2)
    
    avg_mono = np.mean(np.asarray(mono_arr),axis=0)
    avg_quad = np.mean(np.asarray(quad_arr),axis=0)
    #axs[3,0].plot(original_data
    axs[0].plot(s,avg_mono,label=pres[j],color=cs[j])
    axs[1].plot(s,avg_quad,color=cs[j])
    #axs[3,0].legend()
axs[0].plot(original_data['s'],np.mean(monos,axis=0),lw=2,label='PAIR AP from TRAINING',color=cs[-1])
axs[1].plot(original_data['s'],np.mean(quads,axis=0),lw=2,color=cs[-1])
axs[0].legend()
axs[0].set_xlabel('Separation [Mpc/h]')
axs[1].set_xlabel('Separation [Mpc/h]')

plt.show()
#plt.clf()


f,axx = plt.subplots(3,3,figsize=(9,9))

pos_dir = '/pscratch/sd/t/tsfraser/rev_test/voids/AbacusSummit_small_c000_ph3000/z0.500/'
pre_fixes = ['BASESCALE','UPSCALE','DOWNSCALE']

for i in range(3):
    for j in range(3):
        #'VOID_NOAP_DISC_voxel_voids_Rs{smoothing_radius}_RvMed_c{cosmo:03}_ph{phase:03}_hod{hod}_los{los}_cut20.npy'

        #/pscratch/sd/t/tsfraser/rev_test/multipoles/AbacusSummit_small_c000_ph3000/z0.500/VOID_BASESCALE_DISC_voxel_multipoles_Rs10_Rv10_c000_ph3000_hod0_losx_cut20.npy
        #/pscratch/sd/t/tsfraser/rev_test/multipoles/AbacusSummit_small_c000_ph3000/z0.500/VOID_BASESCALE_DISC_voxel_multipoles_Rs10_Rv10_c000_ph3000_hod0_losx_cut20.npy
        #/pscratch/sd/t/tsfraser/rev_test/multipoles/AbacusSummit_small_c000_ph3000/z0.500/VOID_UPSCALE_DISC_voxel_multipoles_Rs10_Rv10_c000_ph3000_hod0_losx_cut20.npy
        fname = f'VOID_{pre_fixes[i]}_DISC_voxel_voids_Rs10_RvMed_c000_ph3000_hod0_los{los[j]}_cut20.npy'
        void_positions = np.load(Path(pos_dir,fname),allow_pickle=True)[()]["positions"]
        void_pos = void_positions
        cut = 100
        mask = void_pos[:,j] < 100
        void_pos_cut = void_pos[mask]

        axx[i,j].hist(void_pos_cut[:,j],bins=np.linspace(0,100,50))

plt.show()
