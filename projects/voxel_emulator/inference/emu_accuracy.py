from sunbird.summaries import VoxelVoids
from sunbird.data.data_readers import Abacus
from sunbird.covariance import CovarianceMatrix
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
emulator = VoxelVoids()
s = emulator.coordinates['s']
ell = 0
select_filters = {
    'multipoles': [ell]
    }
slice_filters = {
    's': [0.7, 120]
    }
spow = 0
fig, ax = plt.subplots()
colors = ['C0', 'C1', 'C2', 'C3', 'C4']
data_class = Abacus(
        dataset='voidprior',
        statistics=['voxel_voids'],
        select_filters=select_filters,
        slice_filters=slice_filters)
for i, cosmo in enumerate([0, 1, 3, 4, 13]):
    observation = data_class.get_observation(cosmology=cosmo, hod_idx=80)
    cov = CovarianceMatrix(
          dataset='voidprior',
          covariance_data_class='AbacusSmall',
          statistics=['voxel_voids'],
          select_filters=select_filters,
          slice_filters=slice_filters,
          path_to_models='/pscratch/sd/t/tsfraser/pscratch/new_sunbird/sunbird/trained_models/best/'
          )      
    cov_data = cov.get_covariance_data(volume_scaling=64)
      # cov_emu = cov.get_covariance_emulator()
      # cov_sim = cov.get_covariance_simulation()
    error_data = np.sqrt(np.diag(cov_data))
    parameters = data_class.get_parameters_for_observation(cosmology=cosmo, hod_idx=80)
    prediction, error = emulator(
          param_dict=parameters,
          select_filters=select_filters,
          slice_filters=slice_filters,
          )
    prediction = prediction.numpy()
    
    ax.errorbar(s, s**spow*observation, s**spow*error_data, label=f' c{cosmo:03} Abacus', color=colors[i], ls='', marker='o', ms=4.0)
    ax.plot(s, s**spow*prediction, label=f' c{cosmo:03} Emulator', color=colors[i])
    ax.set_xlim(0, 80)
#if spow == 2:
 #   ax.set_ylabel(rf'$s^2 \ xi_{ell}(s)\,[h^{{-2}}{{\rm Mpc}}^2]$', fontsize=13)
#else:
    ax.set_ylabel(rf'$\xi_{ell}(s)$', fontsize=13)
      # ax[0].set_xlim(0, 80)
    ax.set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$')
    ax.legend()   
    plt.tight_layout()
plt.savefig(f'/pscratch/sd/t/tsfraser/predictions/abacus/box/voidprior/xi{ell}_emuprediction.pdf',bbox_inches='tight')
plt.show()
   # plt.subplots_adjust(hspace=0.0)
   # # plt.savefig(f'fig/parameter_dependence_density_split_cross_{param_name}_nos2.pdf', bbox_inches='tight')
   # plt.show()
