import fitsio
from pathlib import Path
import numpy as np
from revolver import VoxelVoids, setup_logging
from pycorr import TwoPointCorrelationFunction
from cosmoprimo.fiducial import AbacusSummit
from abacusnbody.hod.abacus_hod import AbacusHOD
from astropy.table import Table
from astropy.io import fits
from astropy.io.fits import Header
import argparse
import yaml
import os
import matplotlib.pyplot as plt
import logging


def get_rsd_positions(hod_dict):
    """Read positions and velocities from input fits
    catalogue and return real and redshift-space
    positions."""
    data = hod_dict['LRG']
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']
    x = data['x'] + boxsize / 2
    y = data['y'] + boxsize / 2
    z = data['z'] + boxsize / 2
    #print(np.min(x),np.min(y),np.min(z),'RSD POSITION MINS, NON RSD COORDS')
    x_rsd = x + vx / (hubble * az)
    y_rsd = y + vy / (hubble * az)
    z_rsd = z + vz / (hubble * az)
    x_rsd = x_rsd % boxsize
    y_rsd = y_rsd % boxsize
    z_rsd = z_rsd % boxsize
    return x, y, z, x_rsd, y_rsd, z_rsd

def get_distorted_positions(positions, q_perp, q_para,cellSize=5.0, los='z'):
    """Given a set of comoving galaxy positions in cartesian
    coordinates, return the positions distorted by the 
    Alcock-Pacynski effect"""
    positions_ap = np.copy(positions)
    factor_x = q_para if los == 'x' else q_perp
    factor_y = q_para if los == 'y' else q_perp
    factor_z = q_para if los == 'z' else q_perp
    #print(500/cellSize,'number of cells')
    #print((500/cellSize)/factor_x)
    #print('New AP', (500/cellSize * 1/factor_x / (500/cellSize)),'original ', 1/factor_x)
    #print(np.min(positions_ap),'WHAT AP SEES BEFORE SCALING')
    positions_ap[:, 0] = positions_ap[:,0]/factor_x
    positions_ap[:, 1] = positions_ap[:,1]/factor_y
    positions_ap[:, 2] = positions_ap[:,2]/factor_z
    return positions_ap


def discretize_AP(cellsize,boxsize,q_perp,q_para,los='z'):

    factor_x = q_para if los =='x' else q_perp
    factor_y = q_para if los == 'y' else q_perp
    factor_z = q_para if los == 'z' else q_perp

    current_Nperp  = boxsize/cellsize
    current_Npara  = boxsize/cellsize
    #current_Nz  = boxsize/cellsize
    
    current_q_para = q_para
    current_q_perp = q_perp 
    #current_factor_z = factor_z


    new_Npara = int(current_Npara/current_q_para+0.5)
    new_Nperp = int(current_Nperp/current_q_perp+0.5)
    #new_Nz = int(current_Nz/current_factor_z+0.5)

    new_q_para = current_Npara/new_Npara
    new_q_perp = current_Nperp/new_Nperp
    #new_factor_z = new_Nz/current_Nz
    print("Now: ", new_Npara, " bins along LOS: ",los)
    print("Now: ", new_Nperp, " bins perp to LOS: ", los)
    #new_N_arr = np.array([new_Nx,new_Ny,new_Nz])
    #q_arr = np.array([new_factor_x,new_factor_y,new_factor_z]) 

    return new_q_perp,new_q_para# returned as q_para,qperp

def get_distorted_box(boxsize, q_perp, q_para, los='z'):
    """Distort the dimensions of a cubic box with the
    Alcock-Pacynski effect"""
    factor_x = q_para if los == 'x' else q_perp
    factor_y = q_para if los == 'y' else q_perp
    factor_z = q_para if los == 'z' else q_perp
    boxsize_ap = np.array([boxsize/factor_x, boxsize/factor_y, boxsize/factor_z])
    return boxsize_ap

def output_mock(mock_dict, newBall, fn, tracer):
    """Save HOD catalogue to disk."""
    Ncent = mock_dict[tracer]['Ncent']
    mock_dict[tracer].pop('Ncent', None)
    cen = np.zeros(len(mock_dict[tracer]['x']))
    cen[:Ncent] += 1
    mock_dict[tracer]['cen'] = cen
    table = Table(mock_dict[tracer])
    header = Header({'Ncent': Ncent, 'Gal_type': tracer, **newBall.tracers[tracer]})
    myfits = fits.BinTableHDU(data = table, header = header)
    myfits.writeto(fn, overwrite=True)

def get_hod(p, param_mapping, param_tracer, data_params, Ball, nthread):
    # read the parameters 
    for key in param_mapping.keys():
        mapping_idx = param_mapping[key]
        tracer_type = param_tracer[key]
        if key == 'sigma' and tracer_type == 'LRG':
            Ball.tracers[tracer_type][key] = 10**p[mapping_idx]
        else:
            Ball.tracers[tracer_type][key] = p[mapping_idx]
        # Ball.tracers[tracer_type][key] = p[mapping_idx]
    Ball.tracers['LRG']['ic'] = 1 # a lot of this is a placeholder for something more suited for multi-tracer
    ngal_dict = Ball.compute_ngal(Nthread = nthread)[0]
    N_lrg = ngal_dict['LRG']
    Ball.tracers['LRG']['ic'] = min(1, data_params['tracer_density_mean']['LRG']*Ball.params['Lbox']**3/N_lrg)
    mock_dict = Ball.run_hod(Ball.tracers, Ball.want_rsd, Nthread = nthread)
    return mock_dict

def setup_hod(config):
    #print(f"Processing {config['sim_params']['sim_name']}")
    sim_params = config['sim_params']
    HOD_params = config['HOD_params']
    data_params = config['data_params']
    fit_params = config['fit_params']    
    # create a new abacushod object and load the subsamples
    newBall = AbacusHOD(sim_params, HOD_params)
    newBall.params['Lbox'] = boxsize
    # parameters to fit
    param_mapping = {}
    param_tracer = {}
    for key in fit_params.keys():
        mapping_idx = fit_params[key][0]
        tracer_type = fit_params[key][-1]
        param_mapping[key] = mapping_idx
        param_tracer[key] = tracer_type
    return newBall, param_mapping, param_tracer, data_params


def get_data_positions(filename):
    data = fitsio.read(filename)
    x = data['X']
    y = data['Y']
    z = data['Z']
    #print(np.min(x),np.min(y),np.min(z))
    data_positions = np.c_[x, y, z]
    return data_positions

def get_voids_positions(data_positions, boxsize, cellsize, boxcenter=None, handle=None,
    wrap=True, boxpad=1.0, smoothing_radius=10, return_radii=False, nthreads=1):
    boxcenter = boxsize / 2 if boxcenter is None else boxcenter
    voxel = VoxelVoids(
        handle=handle,
        data_positions=data_positions,
        boxsize=boxsize,
        boxcenter=boxcenter,
        wrap=wrap,
        boxpad=boxpad,
        cellsize=cellsize,
    )
    voxel.set_density_contrast(smoothing_radius=smoothing_radius, nthreads=nthreads)
    voxel.find_voids()
    voids_positions, voids_radii = voxel.postprocess_voids()
    if return_radii:
        return voids_positions, voids_radii
    return voids_positions

if __name__ == '__main__':
    logger = logging.getLogger('voxel_hod')#_covariance')
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_hod", type=int, default=0)
    parser.add_argument("--n_hod", type=int, default=1)
    parser.add_argument("--start_cosmo", type=int, default=0)
    parser.add_argument("--n_cosmo", type=int, default=1)
    parser.add_argument("--start_phase", type=int, default=3000)
    parser.add_argument("--n_phase", type=int, default=1)
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--save_voids", action='store_true')
    parser.add_argument("--save_mocks", action='store_true')

    args = parser.parse_args()
    start_hod = args.start_hod
    n_hod = args.n_hod
    start_cosmo = args.start_cosmo
    n_cosmo = args.n_cosmo
    start_phase = args.start_phase
    n_phase = args.n_phase

    setup_logging(level='INFO')
    boxsize = 2000 #np.array([500,500,500])
    cellsize = 5.0
    smoothing_radius = 10
    redshift = 0.5
    redges = np.arange(0, 151, 5)
    muedges = np.linspace(-1, 1, 241)
    edges = (redges, muedges)

    # HOD configuration
    dataset = 'voidprior'
    config_dir = './'
    config_fn = Path(config_dir, f'hod_config_{dataset}.yaml') #_covariance.yaml') #_covariance.yaml')
#Path(config_dir, f'hod_config_{dataset}_covariance.yaml')
    config = yaml.safe_load(open(config_fn))

    # baseline AbacusSummit cosmology as our fiducial
    fid_cosmo = AbacusSummit(0)

    for cosmo in range(start_cosmo, start_cosmo + n_cosmo):
        # cosmology of the mock as the truth
        mock_cosmo = AbacusSummit(cosmo)
        az = 1 / (1 + redshift)
        hubble = 100 * mock_cosmo.efunc(redshift)

        # calculate distortion parameters
        q_perp = mock_cosmo.comoving_angular_distance(redshift) / fid_cosmo.comoving_angular_distance(redshift)
        q_para = fid_cosmo.efunc(redshift) / mock_cosmo.efunc(redshift)
        #q_para = 1.#1. #.2#HARDCODE
        #q_perp = 1. #1. #.2
        q = q_perp**(2/3) * q_para**(1/3)
        logger.info(f'==========================================')
        logger.info(f'q_perp = {q_perp:.3f}')
        logger.info(f'q_para = {q_para:.3f}')
        logger.info(f'q = {q:.3f}')
        logger.info(f'==========================================')
          
        # Discretized AP
        hods_dir = Path(f'./hod_parameters/{dataset}/')
        hods_fn = hods_dir / f'hod_parameters_{dataset}_c{cosmo:03}.csv'
        hod_params = np.genfromtxt(hods_fn, skip_header=1, delimiter=',')

        for phase in range(start_phase, start_phase + n_phase):
            sim_fn = f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}'
            config['sim_params']['sim_name'] = sim_fn
            #print(sim_fn)
            #print(setup_hod(config))
            #try:
             #   print("Trying....")
            newBall, param_mapping, param_tracer, data_params = setup_hod(config)
            #except:
                #logger.info(f'Skipping c{cosmo:03} as the simulation files are not present')
                #continue

            for hod in range(start_hod, start_hod + n_hod):
                hod_dict = get_hod(hod_params[hod], param_mapping, param_tracer,
                              data_params, newBall, args.nthreads)

                if args.save_mocks:
                    output_dir = Path(f'/pscratch/sd/t/tsfraser/rev_test/{dataset}/',
                        f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/')
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    output_fn = Path(
                        output_dir,
                       f'V_Qcut_AbacusSummit_base_c{cosmo:03}_ph{phase:03}_hod{hod}.npy'
                    )
                    output_mock(hod_dict, newBall, output_fn, 'LRG',)

                x, y, z, x_rsd, y_rsd, z_rsd = get_rsd_positions(hod_dict)

                multipoles_los = []
                for los in ['x', 'y', 'z']:
                    #q_para = 1. #.2#HARDCODE
                    #q_perp = 1. #.2
                    q = q_perp**(2/3) * q_para**(1/3)

                    xpos = x_rsd if los == 'x' else x
                    ypos = y_rsd if los == 'y' else y
                    zpos = z_rsd if los == 'z' else z
                    #print(np.min(xpos),np.min(ypos),np.min(zpos))
                    data_positions = np.c_[xpos, ypos, zpos]
                    # ADD Q_perp = 1.02?
                    logger.info(f'==========================================')
                    logger.info(f'q_perp = {q_perp:.3f}')
                    logger.info(f'q_para = {q_para:.3f}')
                    logger.info(f'q = {q:.3f}')
                    logger.info(f'==========================================')
                    logger.info(f'===== D I S C R E T I Z I N G   A.P. =====')
                    logger.info(f'==========================================')
                    new_q_perp,new_q_para = discretize_AP(cellsize,boxsize,q_perp,q_para,los)
                    new_q = new_q_perp**(2./3.) * new_q_para**(1./3.)
                    logger.info(f'DISCRETIZED FOR LOS: {los}')
                    logger.info(f'==========================================')
                    logger.info(f'q_perp = {new_q_perp:.3f}')
                    logger.info(f'q_para = {new_q_para:.3f}')
                    logger.info(f'q = {new_q:.3f}')
                    logger.info(f'==========================================')
                    boxsize_ap = get_distorted_box(boxsize=boxsize, q_perp=new_q_perp, q_para=new_q_para,
                                                   los=los)
                    #print(boxsize_ap,'BOXSIZE')
                    #q_perp_new,q_para_new = discretize_AP(cellsize,boxsize_ap,q_perp,q_para,los)
                    data_positions_ap = get_distorted_positions(positions=data_positions, los=los,
                                                                q_perp=new_q_perp, q_para=new_q_para)
                    # Run the Voxel void finder
                    handle = f'rev_test'#'/pscratch/sd/t/tsfraser/tmp/voxel_c{cosmo:03}_ph{phase:03}_hod{hod}_los{los}_cut20'
                    #print(boxsize_ap,'stretched box')
                    #print(data_positions_ap.max(),'stretch')
                    voids_positions, voids_radii = get_voids_positions(
                        handle=handle,
                        data_positions=data_positions_ap,
                        boxsize= boxsize_ap,#np.array([boxsize,boxsize,boxsize]),#boxsize_ap,
                        wrap=True,
                        boxpad=1.0,
                        cellsize=cellsize,
                        smoothing_radius=smoothing_radius,
                        return_radii=True,
                        nthreads=args.nthreads
                    )
                    #print(np.median(voids_radii),'med radius')
                    cut = 20#/new_q #np.median(voids_radii)
                    mask = voids_radii>=cut
                    voids_positions = voids_positions[mask]
                    voids_radii= voids_radii[mask] #/new_q
                    #print(voids_positions.min())
                    #print(voids_positions.max())
                
                    #plt.plot(voids_positions[:,0],voids_positions[:,1],'ko')
                    
                    #q_perp = mock_cosmo.comoving_angular_distance(redshift) / fid_cosmo.comoving_angular_distance(redshift)
                    #q_para = fid_cosmo.efunc(redshift) / mock_cosmo.efunc(redshift)
                    #new_q_perp,new_q_para = discretize_AP(cellsize,boxsize,q_perp,q_para,los)
                    #print("=======")
                    #print("q_para: ",new_q_para, "q_perp: ", new_q_perp)
                    #print("==== QS AFTER VOID FINDING ====")
                    # rescale void positions with AP parameters to compute clustering
                    #v##oids_positions_ap = get_distorted_positions(positions=voids_positions, los=los,
                    #                                            q_perp=new_q_perp, q_para=new_q_para)
                   # print(voids_positions_ap.max(),'stretch positions....')
                    # Compute void-galaxy correlation function
                    boxsize_ap = get_distorted_box(boxsize=boxsize, q_perp=new_q_perp, q_para=new_q_para,
                                                   los=los)
                    #q_perp_new,q_para_new = discretize_AP(cellsize,boxsize_ap,q_perp,q_para,los)
                    data_positions_ap = get_distorted_positions(positions=data_positions, los=los,
                                                                q_perp=new_q_perp, q_para=new_q_para)

                    if args.save_voids:
                        cout = {
                            'positions': voids_positions,
                            'radii': voids_radii
                        }
                        output_dir = Path(f'/pscratch/sd/t/tsfraser/rev_test/voids/',#/pscratch/sd/t/tsfraser/voxel_emulator/voxel_voids/HOD/{dataset}/small/',
                            f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/')
                        Path(output_dir).mkdir(parents=True, exist_ok=True)
                        output_fn = Path(
                            output_dir,
                            f'VOID_FINALTEST_DISC_voxel_voids_Rs{smoothing_radius}_RvMed_c{cosmo:03}_ph{phase:03}_hod{hod}_los{los}_cut20.npy'
                        )
                        print(output_fn)
                        np.save(output_fn, cout)
                    #q_perp = mock_cosmo.comoving_angular_distance(redshift) / fid_cosmo.comoving_angular_distance(redshift)
                    #q_para = fid_cosmo.efunc(redshift) / mock_cosmo.efunc(redshift)
                    #new_q_perp,new_q_para = discretize_AP(cellsize,boxsize,q_perp,q_para,los)

                    # rescale void positions with AP parameters to compute clustering
                    #voids_positions_ap = get_distorted_positions(positions=voids_positions, los=los,
                    #                                            q_perp=new_q_perp, q_para=new_q_para)
                    #print(voids_positions_ap.max(),'stretch positions....')
                    # Compute void-galaxy correlation function
                    #boxsize_ap = get_distorted_box(boxsize=boxsize, q_perp=new_q_perp, q_para=new_q_para,
                    #                               los=los)
                    #q_perp_new,q_para_new = discretize_AP(cellsize,boxsize_ap,q_perp,q_para,los)
                    #data_positions_ap = get_distorted_positions(positions=data_positions, los=los,
                     #                                           q_perp=new_q_perp, q_para=new_q_para)


                   #Splitter: take the stretched part, split in twain.
                    #print(boxsize_ap//2,'Boxsizes...')
                    #split_mask =( (data_positions_ap[:,0]<boxsize_ap[0]//2) & (data_positions_ap[:,1]<boxsize_ap[1]//2)&(data_positions_ap[:,2] <boxsize_ap[2]//2))
                    #split_mask_2 = ((data_positions_ap[:,0]>=boxsize_ap[0]//2) & (data_positions_ap[:,1]>=boxsize_ap[1]//2)&(data_positions_ap[:,2]>=boxsize_ap[2]//2))
                  #  print(split_mask_2)
                    #print(np.max(data_positions_ap[split_mask,0]),np.min(data_positions_ap[split_mask,0]))
                    #print(np.max(data_positions_ap[split_mask_2,0]),np.min(data_positions_ap[split_mask_2,0]),'SHOULD EXCEED BOXSIZE/2')

                    #v_mask = (( voids_positions[:,0]<boxsize_ap[0]//2) & (voids_positions[:,1] < boxsize_ap[1]//2) & (voids_positions[:,2] < boxsize_ap[2]//2))
                    #v_mask_2 =(( voids_positions[:,0]>=boxsize_ap[0]//2) & (voids_positions[:,1] >= boxsize_ap[1]//2) & (voids_positions[:,2] >= boxsize_ap[2]//2))
                   
                    #print(np.max(voids_positions[v_mask,0]),np.min((voids_positions[v_mask,0])))
                    #print(np.max(voids_positions[v_mask_2,0]),np.min(voids_positions[v_mask_2,0]),'SHOULD EXCEED BOXSIZE/2') 
                    #print("max galaxy positions", data_positions_ap.max())
                    #print("max void positions", voids_positions.max())



                    result = TwoPointCorrelationFunction(
                        mode='smu', edges=edges, data_positions1=data_positions_ap,
                        data_positions2=voids_positions, estimator='auto', boxsize=boxsize_ap,
                        nthreads=4, compute_sepsavg=False, position_type='pos', los=los,
                        gpu=False,
                    )

                    output_dir = Path(f'/pscratch/sd/t/tsfraser/rev_test/multipoles/',
                        f'AbacusSummit_base_c{cosmo:03}_ph{phase:03}/z0.500/')
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    output_fn = Path(
                        output_dir,
                        f'VOID_FINALTEST_DISC_voxel_multipoles_Rs{smoothing_radius}_Rv10_c{cosmo:03}_ph{phase:03}_hod{hod}_los{los}_cut20.npy'
                    )
                    result.save(output_fn)
