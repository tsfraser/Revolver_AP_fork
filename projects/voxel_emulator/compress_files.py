import numpy as np
from pycorr import TwoPointCorrelationFunction
from pathlib import Path
import matplotlib.pyplot as plt
import os

hods = list(range(0, 100))
cosmos = [0, 1, 2, 3, 4] + [13] + list(range(100, 127))  + list(range(130, 182))

for cosmo in cosmos:
    multipoles_hod = []
    for hod in hods:
        multipoles_los = []
        for los in ['x', 'y', 'z']:
            #print(f'{int(cosmo):03}')
            data_dir = f"/pscratch/sd/t/tsfraser/voxel_emulator/voxel_multipoles/HOD/voidprior/AbacusSummit_base_c{int(cosmo):03}_ph000/z0.500/"
            data_fn = Path(data_dir,f'voxel_multipoles_AP_Rs10_c{int(cosmo):03}_ph000_hod{hod}_los{los}.npy')
            result = TwoPointCorrelationFunction.load(data_fn)
            result.select((0, 120))
            result = result[::4, :]
            s, multipoles = result(ells=(0, 2, 4), return_sep=True)
            multipoles_los.append(multipoles)
        multipoles_hod.append(multipoles_los)

    multipoles_hod = np.asarray(multipoles_hod)
    print(cosmo, np.shape(multipoles_hod))

    output_dir = f'/pscratch/sd/t/tsfraser/new_sunbird/sunbird/data/clustering/abacus/voidprior/voxel_voids/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_fn = Path(output_dir) / f'voxel_multipoles_AP_Rs10_c{cosmo:03}.npy'
    cout = {'s': s, 'multipoles': multipoles_hod}
    print(output_fn)
    np.save(output_fn, cout)

#hod = 0
#cosmo = 0
#count = 0
#phases = list(range(3000, 5000))
#multipoles_phases = []
#for phase in phases:
 #   data_dir = f'/pscratch/sd/t/tsfraser/voxel_emulator/voxel_multipoles/HOD/voidprior/small/AbacusSummit_small_c{cosmo:03}_ph{phase:03}/z0.500/'
  #  if not os.path.exists(data_dir):
  #      continue
  #  multipoles_los = []
  #  for los in ['x', 'y', 'z']:
  #      data_fn = Path(data_dir) / f'voxel_multipoles_AP_Rs10_c{cosmo:03}_ph{phase:03}_hod{hod}_los{los}.npy'
  #      result = TwoPointCorrelationFunction.load(data_fn)
  #      result.select((0, 120))
  #      result = result[::4, :]
  #      s, multipoles = result(ells=(0, 2, 4), return_sep=True)
  #      if multipoles[0, 6] < -0.75:
  #          count +=1 
  #          plt.plot(s,multipoles[0],'k-')
  #          print('failed',phase,multipoles[0,4],count)
  #          break
  #      multipoles_los.append(multipoles)
  #  else:
  #      multipoles_phases.append(multipoles_los)
#mu#ltipoles_phases = np.array(multipoles_phases)
#plt.show()
#output_dir = f'/pscratch/sd/t/tsfraser/voxel_emulator/voxel_multipoles/HOD/voidprior/small/compressed/z0.500/'
#Path(output_dir).mkdir(parents=True, exist_ok=True)
#output_fn = Path(output_dir) / f'voxel_multipoles_AP_Rs10_c{cosmo:03}_hod{hod}.npy'3cout = {'s': s, 'multipoles': multipoles_phases}
#np.save(output_fn, cout)


# phases = list(range(1, 41))
# multipoles_phases = []
# for phase in phases:
#     data_dir = f'/pscratch/sd/e/epaillas/voxel_emulator/voxel_multipoles/Nseries/'
#     data_fn = Path(data_dir) / f'voxel_multipoles_Nseries_zmin0.45_zmax0.6_Rs10_default_ph{phase:04}.npy'
#     result = TwoPointCorrelationFunction.load(data_fn)
#     result.select((0, 120))
#     result = result[::4, :]
#     s, multipoles = result(ells=(0, 2, 4), return_sep=True)
#     multipoles_phases.append(multipoles)
# multipoles_phases.insert(0, np.asarray(multipoles_phases).mean(axis=0))
# multipoles_phases = np.array(multipoles_phases)

# output_dir = f'/pscratch/sd/e/epaillas/voxel_emulator/voxel_multipoles/Nseries/compressed/'
# Path(output_dir).mkdir(parents=True, exist_ok=True)
# output_fn = Path(output_dir) / f'voxel_multipoles_Nseries_zmin0.45_zmax0.6_Rs10_default.npy'
# cout = {'s': s, 'multipoles': multipoles_phases}
# np.save(output_fn, cout)

#phases = list(range(1, 551))
#multipoles_phases = []
#for phase in phases:
#    data_dir = f'/pscratch/sd/e/epaillas/voxel_emulator/voxel_multipoles/Patchy/'
#    data_fn = Path(data_dir) / f'voxel_multipoles_Patchy_NGC_zmin0.45_zmax0.6_Rs10_default_FKP_ph{phase:04}.npy'
#    result = TwoPointCorrelationFunction.load(data_fn)
#    result.select((0, 120))
#    result = result[::4, :]
#    s, multipoles = result(ells=(0, 2, 4), return_sep=True)
#    multipoles_phases.append(multipoles)
#multipoles_phases.insert(0, np.asarray(multipoles_phases).mean(axis=0))
#multipoles_phases = np.array(multipoles_phases)

#output_dir = f'/pscratch/sd/e/epaillas/sunbird/data/clustering/patchy/voidprior/voxel_voids/'
#Path(output_dir).mkdir(parents=True, exist_ok=True)
#output_fn = Path(output_dir) / f'voxel_voids_multipoles_Rs10_NGC_default_FKP_landyszalay.npy'
#cout = {'s': s, 'multipoles': multipoles_phases}
#np.save(output_fn, cout)

