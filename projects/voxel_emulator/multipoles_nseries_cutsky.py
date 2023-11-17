import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path
from sunbird.data.data_readers import NseriesCutsky, CMASS
from sunbird.covariance import CovarianceMatrix
from sunbird.summaries import Bundle
from getdist import plots, MCSamples
import argparse
#plt.style.use(['science.mplstyle', 'bright.mplstyle'])
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
def read_dynesty_chain(filename,hmc=True):
    data = np.genfromtxt(filename, skip_header=1, delimiter=",")
    chain = data[:, 4:]
    weights = np.exp(data[:, 1] - data[-1, 2])
    return chain, weights
args = argparse.ArgumentParser()
args.add_argument('--statistic', type=str, nargs="*", default=['voxel_voids'])
args.add_argument('--ell', type=int, nargs="*", default=[0])
args.add_argument('--phase', type=int, default=0)
args = args.parse_args()
for statistic in args.statistic:
    for ell in args.ell:
        print(statistic, ell)
        root_dir = Path('/pscratch/sd/t/tsfraser/sunbird/chains/voxel_voids/')
        # chain_handle = f'nseries_cutsky_ph{args.phase}_density_split_cross_density_split_auto_tpcf_mae_patchycov_smin0.70_smax150.00_m02_q0134_base_Nur_bbn'
        chain_handle = 'hmc_nseries_cutsky_ph0_voxel_voids_mae_patchycov_vol1_smin0.70_smax120.00_m02_vscale1'#,
       # 'hmc_nseries_cutsky_ph0_voxel_voids_mae_patchycov_vol40_smin0.70_smax120.00_m02_vscale40',
        #'hmc_nseries_cutsky_ph0_voxel_voids_mae_patchycov_vol84_smin0.70_smax120.00_m02_vscale84' #f'nseries_cutsky_ph0_density_split_cross_density_split_auto_tpcf_mae_patchycov_vol1_smin0.70_smax150.00_m02_q0134_base_bbn'
        names = [
            'omega_b', 'omega_cdm', 'sigma8_m','n_s','logM1','logM_cut','alpha','logsigma','kappa'
            # 'nrun', 'N_ur', 'w0_fld', 'wa_fld',
            #'logM1', 'logM_cut', 'alpha', 'logsigma', 'kappa'#, 'B_cen', 'B_sat'
        ]
        labels = [
            r'\omega_{\rm b}',
            r'\omega_{\rm cdm}',
            r'\sigma_8',
            r'n_s',
            # r'\alpha_s',
            # r'N_{\rm ur}',
            # r'w_0',
            # r'w_a',
            'logM_1', r'logM_{\rm cut}', r'\alpha'#, r'\alpha_{\rm vel, s}',
            #r'\alpha_{\rm vel, c}', 
            r'\log \sigma', r'\kappa'#, r'B_{\rm cen}', r'B_{\rm sat}'
        ]
        chain_fn = root_dir / chain_handle / 'results.csv'
        data = np.genfromtxt(chain_fn, skip_header=1, delimiter=",")
        chain = data[:, 4:]
        weights = np.exp(data[:, 1] - data[-1, 2])
        samples = MCSamples(samples=chain, weights=weights, labels=labels, names=names)
        parameters = {names[i]: samples.mean(names[i]) for i in range(len(names))}
        parameters['nrun'] = 0.0
        parameters['N_ur'] = 2.046
        parameters['w0_fld'] = -1.0
        parameters['wa_fld'] = 0.0
        fig, ax = plt.subplots(2, 1, figsize=(4.5, 4.5), sharex=True, gridspec_kw={'height_ratios': [4, 1]})
        colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
        #quantiles = [0, 1, 3, 4] if 'density_split' in statistic else [5]
        #for ds in quantiles:
        s = np.load('/pscratch/sd/t/tsfraser/new_sunbird/sunbird/data/s.npy')
        smin, smax = 0.7, 120
        s = s[(s > smin) & (s < smax)]
        slice_filters = {
            's': [smin, smax] }
        select_filters = {
            'multipoles': ell}
        datavector = NseriesCutsky(
                statistics=[statistic],
                select_filters=select_filters,
                slice_filters=slice_filters,
        ).get_observation(phase=args.phase)
        cmass = CMASS(
                statistics=[statistic],
                select_filters=select_filters,
                slice_filters=slice_filters,
        ).get_observation()
        cov = CovarianceMatrix(
                covariance_data_class='Patchy',
                statistics=[statistic],
                select_filters=select_filters,
                slice_filters=slice_filters,
                path_to_models='/pscratch/t/tsfraser/new_sunbird/sunbird/trained_models/best/'
        )
        emulator = Bundle(
                summaries=[statistic],
                path_to_models='/pscratch/t/tsfraser/new_sunbird/sunbird/trained_models/best/',
        )
        model, error_model = emulator(
                param_dict=parameters,
                select_filters=select_filters,
                slice_filters=slice_filters,
        )
        cov_data = cov.get_covariance_data()
        cov_emu = cov.get_covariance_emulator()
        cov_sim = cov.get_covariance_simulation()
        error_data = np.sqrt(np.diag(cov_data))
        error_emu = np.sqrt(np.diag(cov_emu))
        error_sim = np.sqrt(np.diag(cov_sim))
        error_model = np.sqrt(error_sim**2 + error_emu**2)
        error_tot = np.sqrt(error_data**2 + error_emu**2 + error_sim**2)
        ax[0].errorbar(s, s**2*datavector, s**2*error_data, marker='o',
                           ms=4.0, ls='', color=colors[ds], elinewidth=1.0,
                           capsize=0.0, markeredgewidth=1.0,
                           markerfacecolor=lighten_color(colors[ds], 0.5),
                           markeredgecolor=colors[ds], label='Nseries')
        ax[0].errorbar(s, s**2*cmass, s**2*error_data, ls='', color='k', marker='s', markeredgewidth=1.0,
                           markerfacecolor=lighten_color('k', 0.5), elinewidth=1.0, capsize=0.0,
                           markeredgecolor='k', label='CMASS', ms=4.0)
        nu = 0.01
        model = model / (1 + nu * model)
        ax[0].plot(s, s**2*model, ls='-', color=colors[ds])
        ax[0].fill_between(s, s**2*(model - error_model), s**2*(model + error_model), color=colors[ds], alpha=0.3)
        ax[1].plot(s, (datavector - model)/error_tot, ls='-', color=colors[ds])
        ax[0].axes.get_xaxis().set_visible(False)
        ax[1].fill_between([-1, 160], -1, 1, color='grey', alpha=0.2)
        ax[1].fill_between([-1, 160], -2, 2, color='grey', alpha=0.15)
        ax[1].set_ylim(-3, 3)
        ax[1].set_xlim(-0, 150)
        ax[1].set_xlabel(r'$s\,[h^{-1}{\rm Mpc}]$')
        ax[0].set_ylabel(rf'$s^2\xi_{ell}(s)\,[h^{{-2}}{{\rm Mpc}}^2]$')
        ax[1].set_ylabel(rf'$\Delta\xi_{ell}(s)/\sigma$')
        if ell == 0:
            multipole = 'monopole'
        elif ell == 2:
            multipole = 'quadrupole'
        if statistic == 'voxel_voids':
            title = f'Void-galaxy 2PCF {multipole}'
        elif statistic == 'density_split_auto':
            title = f'Density-split ACF {multipole}'
        elif statistic == 'density_split_cross':
            title = f'Density-split CCF {multipole}'
        ax[0].set_title(title, fontsize=15)
        # ax[0].legend()
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        plt.savefig(f'multipoles_nseries_cutsky_{statistic}_ell{ell}.pdf')
        plt.show()
