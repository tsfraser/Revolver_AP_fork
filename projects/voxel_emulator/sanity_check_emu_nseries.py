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
            #r'\alpha_{\rm vel, c}', 
            #r'\log \sigma', r'\kappa'#, r'B_{\rm cen}', r'B_{\rm sat}'
        #]


def read_dynesty_chain(filename, add_fsigma8=False, redshift=0.5):
    data = pd.read_csv(filename)
    param_names = list(data.columns[4:])
    if add_fsigma8:
        growth = Growth(
            emulate=True,
        )
        data["fsigma8"] = growth.get_fsigma8(
            omega_b=data["omega_b"].to_numpy(),
            omega_cdm=data["omega_cdm"].to_numpy(),
            sigma8=data["sigma8_m"].to_numpy(),
            n_s=data["n_s"].to_numpy(),
            N_ur=data["N_ur"].to_numpy(),
            w0_fld=data["w0_fld"].to_numpy(),
            wa_fld=data["wa_fld"].to_numpy(),
            z=redshift,
        )
        param_names.append("fsigma8")
    data = data.to_numpy()
    chain = data[:, 4:]
    weights = np.exp(data[:, 1] - data[-1, 2])
    return param_names, chain, weights

def read_hmc_chain(filename, add_fsigma8=False, redshift=0.5):
    data = pd.read_csv(filename)
    param_names = list(data.columns)
    if add_fsigma8:
        growth = Growth(
            emulate=True,
        )
        data["fsigma8"] = growth.get_fsigma8(
            omega_b=data["omega_b"].to_numpy(),
            omega_cdm=data["omega_cdm"].to_numpy(),
            sigma8=data["sigma8_m"].to_numpy(),
            n_s=data["n_s"].to_numpy(),
            N_ur=data["N_ur"].to_numpy(),
            w0_fld=data["w0_fld"].to_numpy(),
            wa_fld=data["wa_fld"].to_numpy(),
            z=redshift,
        )
        param_names.append("fsigma8")
    data = data.to_numpy()
    return param_names, data, None 


def get_MCSamples(filename, add_fsigma8=False, redshift=0.5, hmc=False,):
    priors = {
        "omega_b": [0.0207, 0.0243],
        "omega_cdm": [0.1032, 0.140],
        "sigma8_m": [0.678, 0.938],
        "n_s": [0.9012, 1.025],
        "nrun": [-0.038, 0.038],
        "N_ur": [1.188, 2.889],
        "w0_fld": [-1.22, -0.726],
        "wa_fld": [-0.628, 0.621],
        "logM1": [13.2, 14.4],
        "logM_cut": [12.4, 13.3],
        "alpha": [0.7, 1.5],
        "alpha_s": [0.7, 1.3],
        "alpha_c": [0.0, 0.5],
        "logsigma": [-3.0, 0.0],
        "kappa": [0.0, 1.5],
        "B_cen": [-0.5, 0.5],
        "B_sat": [-1.0, 1.0],
    }
    if hmc:
        names, chain, weights = read_hmc_chain(
            filename,
            add_fsigma8=add_fsigma8,
            redshift=redshift,
        )
    else:
        names, chain, weights = read_dynesty_chain(
            filename,
            add_fsigma8=add_fsigma8,
            redshift=redshift,
        )
    print(names,labels,'what are the types????')
    samples = MCSamples(
        samples=chain,
        weights=weights,
        labels=[labels[n] for n in names],
        names=names,
        ranges=priors,
    )
    print(samples.getTable(limit=1).tableTex())
    return samples,names



#slice_filters = {
        #    's': [smin, smax] }
        #select_filters = {
            #'multipoles': ell}


#parameters = {names[i]: samples.mean(names[i]) for i in range(len(names))}
#parameters['nrun'] = 0.0
#parameters['N_ur'] = 2.046
#parameters['w0_fld'] = -1.0
#parameters['wa_fld'] = 0.0

chain_fn = ['/pscratch/sd/t/tsfraser/sunbird/chains/voxel_voids/abacus_c0_voxel_voids_mae_patchycov_vol64_smin0.70_smax120.00_m02_BOXCUT_BBN/results.csv',
            '/pscratch/sd/t/tsfraser/sunbird/chains/voxel_voids/abacus_c0_voxel_voids_mae_patchycov_vol64_smin0.70_smax120.00_m02_NEW_RUN_4PARAM/results.csv']



for statistic in ['voxel_voids']:
    for ell in [0,2]:
        for fn in chain_fn:
            slice_filters = {
                's': [0.7,120.00] }
            select_filters = {
            'multipoles': ell}
            print(statistic, ell)
        #chain_fn = '/pscratch/sd/t/tsfraser/sunbird/chains/voxel_voids/abacus_c0_voxel_voids_mae_patchycov_vol64_smin0.70_smax120.00_m02_BOXCUT_BBN/'


###_cutsky_ph0_voxel_voids_mae_patchycov_vol84_smin0.70_smax120.00_m02_vscale84/results.csv

            samples, names =  get_MCSamples(fn, hmc=False, add_fsigma8=False)
        #print(cnames)
        #print(names)
        #
        #print(len(names))
        #print(names[0])
            indices = [1,2,3,4,5,6,8,9,10,13]
        #print(samples.mean(names[1]))
        #parameters = data_class.get_parameters_for_observation(cosmology=cosmo, hod_idx=80)
            chain_parameters = {names[i]: samples.mean(names[i]) for i in indices}#range(1,len(names))}
        #chain_parameters['nrun'] = 0.0
        #chain_parameters['N_ur'] = 2.046
        #chain_parameters['w0_fld'] = -1.0
        #chain_parameters['wa_fld'] = 0.0
            datavector = Abacus(dataset='voidprior',
                statistics=[statistic],
                select_filters=select_filters,
                slice_filters=slice_filters,
            ).get_observation(phase=0)
#cmass = CMASS(dataset='voidprior',
       #         statistics=[statistic],
       #         select_filters=select_filters,
       #         slice_filters=slice_filters,
       # ).get_observation()
            cov = CovarianceMatrix(dataset='voidprior',
                covariance_data_class='AbacusSmall',
                statistics=[statistic],
                select_filters=select_filters,
                slice_filters=slice_filters,
                path_to_models='/pscratch/sd/t/tsfraser/new_sunbird/sunbird/trained_models/best/')
            emulator = VoxelVoids()
            s = emulator.coordinates['s']
#Bundle(dataset='voidprior',
                #summaries=[statistic],
                 #path_to_models='/pscratch/sd/t/tsfraser/new_sunbird/sunbird/trained_models/best/')
            parameters = Abacus(dataset='voidprior', statistics=[statistic], select_filters=select_filters, slice_filters = slice_filters).get_parameters_for_observation(phase=0)
        #THESE SHOULD BE THE TRUE COSMOLOGY OF THE NSERIES
        #print('NseriesCutsky:',parameters)
        #NSERIES_TRUE
        #print()
            #Omega_m = 0.286
       # pred_fsigma8 = growth.get_fsigma8(
       # omega_b = np.array(truth['omega_b']).reshape(1,1),
       # omega_cdm = np.array(truth['omega_cdm']).reshape(1,1),
       # sigma8 = np.array(truth['sigma8_m']).reshape(1,1),
       # n_s = np.array(truth['n_s']).reshape(1,1),
       # N_ur = np.array(truth['N_ur']).reshape(1,1),
       # w0_fld = np.array(truth['w0_fld']).reshape(1,1),
       # wa_fld = np.array(truth['wa_fld']).reshape(1,1),
       # z=redshift,
       # )
            truth = {
            "omega_b": 0.02303,
            "omega_cdm": 0.1171,
            "sigma8_m": 0.82,
            "n_s": 0.96,
            "nrun": 0.0,
            "N_ur": 2.046,
            "w0_fld": -1.0,
            "wa_fld": 0.0,
              "logM_cut": 13.7299,
            "logM1": 13.0644,
            "alpha": 1.3446,
            "logsigma": -0.37,
            "kappa": 1.0446,
            "alpha_c": 0.0,
            "alpha_s": 1.0,
            "B_cen": 0.0,
            "B_sat": 0.0,
            }
        #truth['fsigma8'] = pred_fsigma8[0][0]
        #data_class.get_parameters_for_observation(cosmology=cosmo, hod_idx=80)
            parameters=truth
        #print('updated truth',truth)
            model, error_model = emulator(
                param_dict=parameters,
                select_filters=select_filters,
                slice_filters=slice_filters)
        #THIS IS THE EMULATED MONOPOLE+QUAD FOR THE TRUE COSMOLOGY OF NSERIES
            cov_data = cov.get_covariance_data(volume_scaling=64)
            error_data=np.sqrt(np.diag(cov_data))

            model = model.numpy()
            emulator=VoxelVoids(path_to_models='/pscratch/sd/t/tsfraser/new_sunbird/sunbird/trained_models/best/')
            #best_fit_params= chain_parameters
        #print(chain_parameters)
            #model_bestfit,error_bestfit = emulator(param_dict=best_fit_params,select_filters=select_filters,slice_filters=slice_filters)
            #plt.errorbar(s,datavector,error_data, label ='Nseries with true cosmo',lw=3)
            plt.plot(s,model,label=fn,lw=3)
            #plt.plot(s,model_bestfit,label='best fit cosmo + %s'%(fn),lw=3,linestyle='dashed')
            plt.legend()
        plt.show()
