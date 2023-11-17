import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path
from sunbird.data.data_readers import NseriesCutsky, CMASS
from sunbird.covariance import CovarianceMatrix
from sunbird.summaries import Bundle
from getdist import plots, MCSamples
from cosmoprimo.fiducial import AbacusSummit
from sunbird.cosmology.growth_rate import Growth
from sunbird.summaries import VoxelVoids
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sunbird.data.data_readers import Abacus

#names = ['omega_b', 'omega_cdm', 'sigma8_m','n_s','logM1','logM_cut','alpha','logsigma','kappa']
            # 'nrun', 'N_ur', 'w0_fld', 'wa_fld',
            #'logM1', 'logM_cut', 'alpha', 'logsigma', 'kappa'#, 'B_cen', 'B_sat'
       # ]
labels = {
    "omega_b": r"\omega_{\rm b}",
    "omega_cdm": r"\omega_{\rm cdm}",
    "sigma8_m": r"\sigma_8",
    "n_s": r"n_s",
    "nrun": r"\alpha_s",
    "N_ur": r"N_{\rm ur}",
    "w0_fld": r"w_0",
    "wa_fld": r"w_a",
    "logM1": "logM_1",
    "logM_cut": r"logM_{\rm cut}",
    "alpha": r"\alpha",
    "alpha_s": r"\alpha_{\rm vel, s}",
    "alpha_c": r"\alpha_{\rm vel, c}",
    "logsigma": r"\log \sigma",
    "kappa": r"\kappa",
    "B_cen": r"B_{\rm cen}",
    "B_sat": r"B_{\rm sat}",
    "fsigma8": r"f \sigma_8",
    "Omega_m": r"\Omega_{\rm m}",
    "H0": r"H_0",
}
#labels = #[r'\omega_{\rm b}',r'\omega_{\rm cdm}',r'\sigma_8','n_s',r'logM_1',r'logM_{\rm cut}',r'\alpha',r'\log \sigma',r'\kappa']
            # r'\alpha_s',
            # r'N_{\rm ur}',
            # r'w_0',
            # r'w_a',
            #'logM_1', r'logM_{\rm cut}', r'\alpha'#, r'\alpha_{\rm vel, s}',
          




emulator =VoxelVoids(path_to_models='/pscratch/sd/t/tsfraser/new_sunbird/sunbird/trained_models/best/')
s = emulator.coordinates['s']

select_filters ={ 'multipoles': [0,2]}
slice_filters = {'s' : [0.7,120.00]}

cosmo_params = {}
hod_params = {}

parameters = Abacus(dataset='voidprior', statistics=['voxel_voids'], select_filters=select_filters, slice_filters = slice_filters).get_parameters_for_observation(cosmology=4,hod_idx=80)

datavector = Abacus(dataset='voidprior',
                statistics=['voxel_voids'],
                select_filters=select_filters,
                slice_filters=slice_filters,
            ).get_observation(cosmology=4,hod_idx=80)
prediction,_ = emulator(
             param_dict= parameters,
             select_filters = select_filters,
             slice_filters = slice_filters)


pred = prediction.numpy()


emulator2 =VoxelVoids(path_to_models='/pscratch/sd/t/tsfraser/old_models/')
s = emulator2.coordinates['s']

select_filters ={ 'multipoles': [0,2]}
slice_filters = {'s' : [0.7,120.00]}

cosmo_params = {}
hod_params = {}

parameters = Abacus(dataset='voidprior', statistics=['voxel_voids'], select_filters=select_filters, slice_filters = slice_filters).get_parameters_for_observation(cosmology=4,hod_idx=80)

datavector2 = Abacus(dataset='voidprior',
                statistics=['voxel_voids'],
                select_filters=select_filters,
                slice_filters=slice_filters,
            ).get_observation(cosmology=4,hod_idx=80)

datavector2 = np.load('/pscratch/sd/t/tsfraser/new_sunbird/sunbird/data/clustering/abacus/voidprior/voxel_voids/voxel_multipoles_Rs10_RvMed_c004_cut20.npy',allow_pickle=True)[()]['multipoles'][80,0,1,:]

prediction2,_ = emulator2(
             param_dict= parameters,
             select_filters = select_filters,
             slice_filters = slice_filters)


pred2 = prediction2.numpy()



plt.plot(s,datavector[30:],label='new training data')
plt.plot(s,datavector2[:],label='old training data')
plt.plot(s,pred[30:],label='c004 prediction from current emulator model')
plt.plot(s,pred2[30:],label='c004 prediction from old emulator')
plt.legend()
plt.show()

print(pred) 
