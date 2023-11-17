from getdist import plots, MCSamples
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
import inference_plot_utils as inference_plots

args = argparse.ArgumentParser()
args.add_argument(
    "--chain_dir",
    type=str,
    default="/pscratch/sd/t/tsfraser/sunbird/chains/voxel_voids/",
)
args.add_argument(
    "--loss",
    type=str,
    default='learned_gaussian',
)
args = args.parse_args()

cosmologies = [1]#, 1, 3, 4]

# best hod for each cosmology
best_hod = {1: 80}#, 1:80, 3:80, 4: 80}

chain_dir = Path(args.chain_dir)
chain_labels = ['c001 new data with new emu', 'c001 old chain with old data','c001 old emu and old data'] #[f"c{cosmo:03}" for cosmo in cosmologies]

params_toplot = [
    "omega_cdm",
    "sigma8_m",#,"n_s",
    # "logM_cut",
    # "logM1"
]
true_params = [
    inference_plots.get_true_params(cosmo, best_hod[cosmo]) for cosmo in cosmologies
]

samples_list = []
for cosmo in cosmologies:
    hod = best_hod[cosmo]
    chain_fn = [f'/pscratch/sd/t/tsfraser/sunbird/chains/voxel_voids/abacus_c{cosmo}_voxel_voids_mae_patchycov_vol64_smin0.70_smax120.00_m02_NEW_RUN_6PARAM',
       f'/pscratch/sd/t/tsfraser/sunbird/chains/voxel_voids/abacus_c{cosmo}_voxel_voids_mae_patchycov_vol64_smin0.70_smax120.00_m02_BOXCUT_BBN_FULL',
       f'/pscratch/sd/t/tsfraser/sunbird/chains/voxel_voids/abacus_c{cosmo}_voxel_voids_mae_patchycov_vol64_smin0.70_smax120.00_m02_OLD_DATA_OLD_EMU']
 
    samples_list.append(
        inference_plots.get_MCSamples(
            Path(chain_fn[0], "results.csv"),
        ))
    samples_list.append(
        inference_plots.get_MCSamples(
            Path(chain_fn[1], "results.csv"),
        ))

    samples_list.append(
        inference_plots.get_MCSamples(
            Path(chain_fn[2], "results.csv"),
        )

    )

colors = ["#4165c0", "#e770a2", "#5ac3be", "dimgray"]#,"#f79a1e"]
g = inference_plots.plot_corner(
    samples_list,
    params_toplot,
    chain_labels,
    colors=colors,
    true_params=None,
    markers=true_params,
    markers_colors=colors[::-1],
    inches_per_param=15.5 / 7,
)
# g.fig.suptitle('Voxel voids')
# plt.savefig("figures/pdf/F7_cosmo_c0_c1_c3_c4.pdf", bbox_inches="tight")
# plt.savefig("figures/png/F7_cosmo_c0_c1_c3_c4.png", bbox_inches="tight")
# plt.savefig('cosmo_inference_abacus.pdf')
plt.tight_layout()
plt.savefig('cosmo_inference_abacus.png', dpi=300, bbox_inches='tight')
plt.show()
