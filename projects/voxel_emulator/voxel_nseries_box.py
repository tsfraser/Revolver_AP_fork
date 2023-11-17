import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
import time
import yaml
import numpy as np
import argparse
from astropy.table import Table
from astropy.io import fits
from astropy.io.fits import Header
from revolver import VoxelVoids,setup_logging
from pathlib import Path
from pycorr import TwoPointCorrelationFunction, setup_logging
from cosmoprimo.fiducial import AbacusSummit
from cosmoprimo.cosmology import Cosmology
from cosmoprimo.utils import DistanceToRedshift
from pyrecon import utils
import fitsio
import logging
import time
import warnings
def get_rsd_positions(filename):
    """Read positions and velocities from input fits
    catalogue and return real and redshift-space
    positions."""
    data = np.genfromtxt(filename)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    vx = data[:, 3]
    vy = data[:, 4]
    vz = data[:, 5]
    x_rsd = x + vx / (hubble * az)
    y_rsd = y + vy / (hubble * az)
    z_rsd = z + vz / (hubble * az)
    x_rsd = x_rsd % boxsize
    y_rsd = y_rsd % boxsize
    z_rsd = z_rsd % boxsize
    return x, y, z, x_rsd, y_rsd, z_rsd

def get_data_positions(filename, distance, zmin=0.45, zmax=0.6,
    weight_type=None,):
    """Read Patchy mocks."""
    data = np.genfromtxt(filename)
    mask = (data[:, 2] > zmin) & (data[:, 2] < zmax)
    ra = data[mask, 0]
    dec = data[mask, 1]
    redshift = data[mask, 2]
    weights = np.ones(len(ra))
    if 'default' in weight_type:
        weights *= data[mask, 3]
    if 'completeness' in weight_type:
        weights *= data[mask, 4]
    dist = distance(redshift)
    positions = utils.sky_to_cartesian(dist=dist, ra=ra, dec=dec)
    return positions, weights

def get_distorted_positions(positions,qperp,qpara,los='z'):
    positions_ap = np.copy(positions)
    factor_x = q_para if los =='x' else q_perp
    factor_y = q_para if los =='y' else q_perp
    factor_z = q_para if los =='z' else q_perp
    positions_ap[:,0] /= factor_x
    positions_ap[:,1] /= factor_y
    positions_ap[:,2] /= factor_z
    return positions_ap


def get_voids_positions(data_positions, boxsize, cellsize, boxcenter=None, handle=None,
    wrap=True, boxpad=1.0, smoothing_radius=10, return_radii=False, nthreads=1):
    boxcenter = boxsize /2  if boxcenter is None else boxcenter
    voxel = VoxelVoids(
        handle = handle,
        data_positions=data_positions,
        boxsize=boxsize,
        boxcenter=boxcenter,
        wrap=wrap,
        boxpad=boxpad,
        cellsize=cellsize,
    )
 #   print(smoothing_radius,'smooth rad')
 #   print(boxcenter,'box center')
 #   print(data_positions,'positions......')
#    print('BOX FUCKING SIZE:', boxsize)
    voxel.set_density_contrast(smoothing_radius=smoothing_radius,nthreads=nthreads)
    voxel.find_voids()
    voids_positions, voids_radii = voxel.postprocess_voids()
    if return_radii:
        return voids_positions, voids_radii
    return voids_positions



def get_distorted_positions(positions, q_perp, q_para, los='z'):
    """Given a set of comoving galaxy positions in cartesian
    coordinates, return the positions distorted by the 
    Alcock-Pacynski effect"""
    positions_ap = np.copy(positions)
    factor_x = q_para if los == 'x' else q_perp
    factor_y = q_para if los == 'y' else q_perp
    factor_z = q_para if los == 'z' else q_perp
    positions_ap[:, 0] /= factor_x
    positions_ap[:, 1] /= factor_y
    positions_ap[:, 2] /= factor_z
    return positions_ap

def get_distorted_box(boxsize, q_perp, q_para, los='z'):
    """Distort the dimensions of a cubic box with the
    Alcock-Pacynski effect"""
    factor_x = q_para if los == 'x' else q_perp
    factor_y = q_para if los == 'y' else q_perp
    factor_z = q_para if los == 'z' else q_perp
    boxsize_ap = [boxsize/factor_x, boxsize/factor_y, boxsize/factor_z]
    return boxsize_ap

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    logger = logging.getLogger('voxel_nseries_cubic')
    setup_logging(level='INFO')
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_phase", type=int, default=1)
    parser.add_argument("--n_phase", type=int, default=1)
    parser.add_argument("--nthreads", type=int, default=256)
    #parser.add_argument("--boxsize", type=int, default=2600)
    parser.add_argument("--save_voids", action='store_true')


    args = parser.parse_args()
    n_phase = args.n_phase
    start_phase  = args.start_phase
    boxsize= 2600#  args.boxsize
    cellsize= 5.0
    smoothing_radius = 10
    redshift = 0.5    
    overwrite = True
    redges = np.arange(0,151,1)
    muedges = np.linspace(-1, 1, 241)
    edges = (redges, muedges)

    # baseline AbacusSummit cosmology as our fiducial
    fid_cosmo = AbacusSummit(0)
    #redshift = args.redshift

    # cosmology of the mock as the truth
    mock_cosmo = Cosmology(Omega_m=0.286, Omega_b=0.0470,
                           h=0.7, n_s=0.96, sigma8=0.82,
                           engine='class')
    az = 1 / (1 + redshift)
    hubble = 100 * mock_cosmo.efunc(redshift)

    # calculate distortion parameters
    q_perp = mock_cosmo.comoving_angular_distance(redshift) / fid_cosmo.comoving_angular_distance(redshift)
    q_para = fid_cosmo.efunc(redshift) / mock_cosmo.efunc(redshift)
    q = q_perp**(2/3) * q_para**(1/3)

    logger.info(f'q_perp = {q_perp:.3f}, q_para = {q_para:.3f}, q = {q:.3f}')

    for phase in range(args.start_phase, args.start_phase + args.n_phase):
        start_time = time.time()
        data_dir = Path(f'/pscratch/sd/e/epaillas/ds_boss/nseries/')
        data_fn = data_dir / f'BoxN{phase}.mock'
        logger.info(f'Reading data: {data_fn}')
        x, y, z, x_rsd, y_rsd, z_rsd = get_rsd_positions(data_fn)
        for los in ['x', 'y', 'z']:
            handle =f'/pscratch/sd/t/tsfraser/tmp/RvMed_ph{phase:04}'
            xpos = x_rsd if los == 'x' else x
            ypos = y_rsd if los == 'y' else y
            zpos = z_rsd if los == 'z' else z
            data_positions = np.c_[xpos, ypos, zpos]
            data_positions_ap = get_distorted_positions(positions=data_positions, los=los,
                                                        q_perp=q_perp, q_para=q_para)
            boxsize_ap = np.array(get_distorted_box(boxsize=boxsize, q_perp=q_perp, q_para=q_para,
                                                    los=los))
            #boxcenter_ap = boxsize_ap / 2
            start_time = time.time()
            voids_positions, voids_radii =  get_voids_positions(handle = handle,data_positions = data_positions,boxsize=boxsize,wrap =True,boxpad = 1.0,cellsize=cellsize,smoothing_radius = smoothing_radius,return_radii= True,nthreads = args.nthreads)

            cut = 20 #np.median(voids_radii)
            mask  =  voids_radii>=cut
            voids_positions = voids_positions[mask]
            voids_radii = voids_radii[mask]
          
            logger.info(f'Void finding took {time.time() - start_time:.2f} seconds')
            if args.save_voids:
                output_dir = Path('/pscratch/sd/t/tsfraser/voxel_emulator/voxel_voids/nseries_cubic/')
                Path.mkdir(output_dir, parents=True, exist_ok=True)
                output_fn = output_dir / \
                     f'voxel_voids_Rs{smoothing_radius}_ph{phase:04}_los{los}_Rv20.npy'
                #    f'Rs{args.smoothing_radius}_ph{phase:04}.npy'
                #logger.info(f'Saving density to {output_fn}')
                cout = {'positions': voids_positions, 'radii': voids_radii}
                print(voids_positions.min(),voids_position.max())
                
                np.save(output_fn, cout)

            #rescale for clustering
            voids_positions_ap = get_distorted_positions(positions=voids_positions, los=los, q_perp=q_perp,q_para=q_para)

            logger.info(f'Computing cross-correlation function for void galaxy correlation function')
            start_time = time.time()
            result = TwoPointCorrelationFunction(
                            'smu', edges=edges, data_positions1=data_positions_ap,
                            data_positions2=voids_positions_ap,estimator='auto', boxsize=boxsize_ap,
                            nthreads=8, compute_sepsavg=False, position_type='pos', los=los, gpu =False)
            logger.info(f'CCF took {time.time() - start_time:.2f} seconds')
            output_dir = Path(f'/pscratch/sd/t/tsfraser/voxel_emulator/voxel_voids_multipoles/nseries_cubic/')
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_fn  = Path(output_dir,
                              f'voxel_multipoles_Rs{smoothing_radius}_ph{phase:04}_los{los}_Rv20.npy'
            )
            logger.info(f'Saving to disk: {output_fn}')
            result.save(output_fn)
            #np.save(output_fn, cout)
