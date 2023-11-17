import subprocess
import numpy as np
from pyrecon import RealMesh
import revolver.fastmodules as fastmodules
import logging
import time
import sys
import os


class VoxelVoids:
    def __init__(self, data_positions, boxsize=None, boxcenter=None,
        data_weights=None, randoms_positions=None, randoms_weights=None,
        cellsize=None, wrap=False, boxpad=1.5, handle=None):
        self.data_positions = data_positions
        self.randoms_positions = randoms_positions
        self.boxsize = boxsize
        self.boxcenter = boxcenter
        self.cellsize = cellsize
        self.boxpad = boxpad
        self.wrap = wrap

        self.logger = logging.getLogger('VoxelVoids')

        if data_weights is not None:
            self.data_weights = data_weights
        else:
            self.data_weights = np.ones(len(data_positions))

        if boxsize is None:
            if randoms_positions is None:
                raise ValueError(
                    'boxsize is set to None, but randoms were not provided.')
            if randoms_weights is None:
                self.randoms_weights = np.ones(len(randoms_positions))
            else:
                self.randoms_weights = randoms_weights

        self.handle = 'tmp' if handle is None else handle


    def set_density_contrast(self, smoothing_radius, check=False, ran_min=0.1, nthreads=1):
        self.logger.info('Setting density contrast')
        self.time = time.time()
        #print(os.path.isfile(f"{self.handle}.txt"),"does it exist when setting contrast?")
        if self.boxsize is None:
            # we do a first iteration to figure out the boxsize
            self.randoms_mesh = RealMesh(cellsize=self.cellsize, boxcenter=self.boxcenter, nthreads=nthreads,
                                         positions=self.randoms_positions, boxpad=self.boxpad)
            max_boxsize = np.max(self.randoms_mesh.boxsize)
            # now build the mesh with the fixed boxsize
            self.randoms_mesh = RealMesh(boxsize=max_boxsize, cellsize=self.cellsize,
                                         boxcenter=self.randoms_mesh.boxcenter, nthreads=nthreads,)
            self.randoms_mesh.assign_cic(positions=self.randoms_positions, wrap=self.wrap,
                                         weights=self.randoms_weights)
            self.data_mesh = RealMesh(boxsize=max_boxsize, cellsize=self.cellsize,
                                      boxcenter=self.randoms_mesh.boxcenter, nthreads=nthreads,)
            self.data_mesh.assign_cic(positions=self.data_positions, wrap=self.wrap,
                                    weights=self.data_weights)
            if check:
                mask_nonzero = self.randoms_mesh.value > 0.
                nnonzero = mask_nonzero.sum()
                if nnonzero < 2: raise ValueError('Very few randoms.')
            self.randoms_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True)
            self.data_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True,)
            sum_data, sum_randoms = np.sum(self.data_mesh.value), np.sum(self.randoms_mesh.value)
            alpha = sum_data * 1. / sum_randoms
            self.delta_mesh = self.data_mesh - alpha * self.randoms_mesh
            self.ran_min = ran_min
            threshold = self.ran_min * sum_randoms / len(self.randoms_positions)
            mask = self.randoms_mesh > threshold
            self.delta_mesh[mask] /= alpha * self.randoms_mesh[mask]
            self.delta_mesh[~mask] = 0.0
            del self.data_mesh
            # del self.randoms_mesh
        else:
            self.data_mesh = RealMesh(boxsize=self.boxsize, cellsize=self.cellsize,
                                    boxcenter=self.boxcenter, nthreads=nthreads,
                                    positions=self.randoms_positions, boxpad=self.boxpad)
            self.data_mesh.assign_cic(positions=self.data_positions, wrap=self.wrap,
                                    weights=self.data_weights)
            self.data_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True,)
            
            self.delta_mesh = self.data_mesh / np.mean(self.data_mesh) - 1.
            del self.data_mesh
        #print(self.delta_mesh,'deltas')
        self.logger.info("Delta mesh returned")
        return self.delta_mesh

    def find_voids(self):
        self.logger.info("Finding voids")
        self.nbins = np.array([int(self.delta_mesh.boxsize[0] / self.cellsize), int(self.delta_mesh.boxsize[1]/self.cellsize), int(self.delta_mesh.boxsize[2]/self.cellsize)])
        # write this to file for jozov-grid to read
        #print(os.path.isfile(f"{self.handle}.txt"),"does txt exist before jozov?")
        delta_mesh_flat = np.array(self.delta_mesh, dtype=np.float32)
        #print("Writing the mesh to .dat")
        with open(f'{self.handle}_delta_mesh_n{self.nbins[0]}{self.nbins[1]}{self.nbins[2]}d.dat', 'w') as F:
            delta_mesh_flat.tofile(F, format='%f')
        # now call jozov-grid
        bin_path  = os.path.join(os.path.dirname(__file__), 'c', 'jozov-grid.exe')
        self.logger.info("Passing cmd to jozov")
        cmd = [bin_path, "v", f"{self.handle}_delta_mesh_n{self.nbins[0]}{self.nbins[1]}{self.nbins[2]}d.dat",
               self.handle, str(self.nbins[0]),str(self.nbins[1]),str(self.nbins[2])]
        self.logger.info("Calling jozov now")
        subprocess.call(cmd)
        self.logger.info("Called jozov:")
    def postprocess_voids(self):
        self.logger.info("Post-processing voids")

        mask_cut = np.zeros(self.nbins[0]*self.nbins[1]*self.nbins[2], dtype='int')
        if self.boxsize is None:
            # identify "empty" cells for later cuts on void catalogue
            mask_cut = np.zeros(self.nbins[0]*self.nbins[1]*self.nbins[2], dtype='int')
            fastmodules.survey_mask(mask_cut, self.randoms_mesh.value, self.ran_min)
        self.mask_cut = mask_cut
        self.min_dens_cut = 1.0

        rawdata = np.loadtxt(f"{self.handle}.txt", skiprows=2)
        nvox = self.nbins[0]*self.nbins[1]*self.nbins[2]

        # load zone membership data
        # with open(f"{self.handle}.zone", 'r') as F:
        #     hierarchy = F.readlines()
        # hierarchy = np.asarray(hierarchy, dtype=str)

        # remove voids that: a) don't meet minimum density cut, b) are edge voids, or c) lie in a masked voxel
        select = np.zeros(rawdata.shape[0], dtype='int')
        fastmodules.voxelvoid_cuts(select, self.mask_cut, rawdata, self.min_dens_cut)
        select = np.asarray(select, dtype=bool)
        rawdata = rawdata[select]

        # void minimum density centre locations
        self.logger.info('Calculating void positions')   # 2ND MOST EXPENSIVE STEP?
        xpos, ypos, zpos = self.voxel_position(rawdata[:, 2])
        #print(xpos.min(),ypos.min(),zpos.min(),'min void positions')
        #print(xpos.max(),ypos.max(),zpos.max(),'max void positions')

        # void effective radii
        self.logger.info('Calculating void radii')
        #print(rawdata[-233,:],'what does a row of rawdata look like?')
        vols = (rawdata[:, 5] * self.cellsize ** 3.)
        rads = (3. * vols / (4. * np.pi)) ** (1. / 3)
        # void minimum densities (as delta)
        mindens = rawdata[:, 3] - 1.

        os.remove(f'{self.handle}.void')
        os.remove(f'{self.handle}.txt')
        os.remove(f'{self.handle}.zone')
        os.remove(f'{self.handle}_delta_mesh_n{self.nbins[0]}{self.nbins[1]}{self.nbins[2]}d.dat')

        self.logger.info(f"Found a total of {len(rawdata)} voids in {time.time() - self.time:.2f} s.")
        return np.c_[xpos, ypos, zpos], rads

    def voxel_position(self, voxel):
        #xind = np.array(voxel / (self.nbins[1]*self.nbins[2]), dtype=int)
       # yind = np.array((voxel - xind * self.nbins[0]*self.nbins[2]) / self.nbins[0], dtype=int) # possible bug?
        #zind = np.array(voxel % self.nbins[2], dtype=int)
        #print(voxel[0])
        voxel = voxel.astype('i')
        #print(voxel[0])
        all_vox = np.arange(0,self.nbins[0]*self.nbins[1]*self.nbins[2],dtype=int)
        vind = np.zeros((np.copy(all_vox).shape[0]),dtype=int) 
        xpos = np.zeros(vind.shape[0],dtype=float)
        ypos = np.zeros(vind.shape[0],dtype=float)
        zpos = np.zeros(vind.shape[0],dtype=float)
        all_vox = np.arange(0,self.nbins[0]*self.nbins[1]*self.nbins[2],dtype = int)
        xi = np.zeros(self.nbins[0]*self.nbins[1]*self.nbins[2])
        yi = np.zeros(self.nbins[1]*self.nbins[2])
        zi = np.arange(self.nbins[2])




        if self.boxsize is None:
            for i in range(self.nbins[1]):
                yi[i*(self.nbins[2]):(i+1)*(self.nbins[2])] =i
         
            for i in range(self.nbins[0]):
                xi[i*(self.nbins[1]*self.nbins[2]):(i+1)*(self.nbins[1]*self.nbins[2])] = i


            xpos = xi*self.delta_mesh.boxsize[0]/self.nbins[0]
            ypos = np.tile(yi,self.nbins[0])*self.delta_mesh.boxsize[1]/self.nbins[1]
            zpos = np.tile(zi,self.nbins[1]*self.nbins[0])*self.delta_mesh.boxsize[2]/self.nbins[2]


            xpos += self.delta_mesh.boxcenter[0] - self.delta_mesh.boxsize[0] / 2.
            ypos += self.delta_mesh.boxcenter[1] - self.delta_mesh.boxsize[1] / 2.           
            zpos += self.delta_mesh.boxcenter[2] - self.delta_mesh.boxsize[2] / 2.


            return xpos[voxel],ypos[voxel],zpos[voxel]

        else:
            for i in range(self.nbins[1]):
                yi[i*(self.nbins[2]):(i+1)*(self.nbins[2])] =i
            for i in range(self.nbins[0]):
                xi[i*(self.nbins[1]*self.nbins[2]):(i+1)*(self.nbins[1]*self.nbins[2])] = i


            xpos = xi*self.boxsize[0]/self.nbins[0]
            ypos = np.tile(yi,self.nbins[0])*self.boxsize[1]/self.nbins[1]
            zpos = np.tile(zi,self.nbins[1]*self.nbins[0])*self.boxsize[2]/self.nbins[2]


            return xpos[voxel],ypos[voxel],zpos[voxel]
 
        #for xind in range(self.nbins[0]) :
        #    for yind in range(self.nbins[1]):
         #       for zind in range(self.nbins[2]):
          #          vidx = zind +yind*self.nbins[2]+xind*self.nbins[1]*self.nbins[2]
           #         if self.boxsize is None:
            #            xpos[vidx] = xind * self.delta_mesh.boxsize[0] / self.nbins[0]
             #           ypos[vidx] = yind * self.delta_mesh.boxsize[1] / self.nbins[1]
              #          zpos[vidx] = zind * self.delta_mesh.boxsize[2] / self.nbins[2]

               #         xpos[vidx] += self.delta_mesh.boxcenter[0] - self.delta_mesh.boxsize[0] / 2.
                #        ypos[vidx] += self.delta_mesh.boxcenter[1] - self.delta_mesh.boxsize[1] / 2.
                 #       zpos[vidx] += self.delta_mesh.boxcenter[2] - self.delta_mesh.boxsize[2] / 2.
                  #  else:
                   #     xpos[vidx] = xind * self.boxsize[0] / self.nbins[0]
                    #    ypos[vidx] = yind * self.boxsize[1] / self.nbins[1]
                     #   zpos[vidx] = zind * self.boxsize[2] / self.nbins[2]

        # array of all positions computed. Now, use the ones in the voxel arrangement.
        # i.e.
        #print('voids with voxels',all_vox[voxel]) # = the voxels with voids...
        #print('Number of voxels containing voids: ', all_vox[voxel].shape[0]) 
        #print(type(xpos[voxel][0]))
        #return xpos[voxel], ypos[voxel], zpos[voxel]


# if not self.is_box:  # convert void centre coordinates from box Cartesian to sky positions
#     xpos += self.xmin
#     ypos += self.ymin
#     zpos += self.zmin
#     dist = np.sqrt(xpos**2 + ypos**2 + zpos**2)
#     redshift = self.cosmo.get_redshift(dist)
#     ra = np.degrees(np.arctan2(ypos, xpos))
#     dec = 90 - np.degrees(np.arccos(zpos / dist))
#     ra[ra < 0] += 360
#     xpos = ra
#     ypos = dec
#     zpos = redshift
#     # and an additional cut on any voids with min. dens. centre outside specified redshift range
#     select_z = np.logical_and(zpos > self.z_min, zpos < self.z_max)
#     rawdata = rawdata[select_z]
#     densratio = densratio[select_z]
#     hierarchy = hierarchy[select_z]
#     xpos = xpos[select_z]
#     ypos = ypos[select_z]
#     zpos = zpos[select_z]

# void average densities and barycentres
# avgdens = np.zeros(len(rawdata))
# barycentres = np.zeros((len(rawdata), 3))
# for i in range(len(rawdata)):
#     member_voxels = np.fromstring(hierarchy[i], dtype=int, sep=' ')[1:]
#     member_dens = np.zeros(len(member_voxels), dtype='float64')
#     fastmodules.get_member_densities(member_dens, member_voxels, self.rhoflat)
#     # member_dens = self.rhoflat[member_voxels]
#     avgdens[i] = np.mean(member_dens) - 1.
#     if self.use_barycentres:
#         member_x, member_y, member_z = self.voxel_position(member_voxels)
#         barycentres[i, 0] = np.average(member_x, weights=1. / member_dens)
#         barycentres[i, 1] = np.average(member_y, weights=1. / member_dens)
#         barycentres[i, 2] = np.average(member_z, weights=1. / member_dens)
# if self.use_barycentres and not self.is_box:
#     barycentres[:, 0] += self.xmin
#     barycentres[:, 1] += self.ymin
#     barycentres[:, 2] += self.zmin
#     dist = np.linalg.norm(barycentres, axis=1)
#     redshift = self.cosmo.get_redshift(dist)
#     ra = np.degrees(np.arctan2(barycentres[:, 1], barycentres[:, 0]))
#     dec = 90 - np.degrees(np.arccos(barycentres[:, 2] / dist))
#     ra[ra < 0] += 360
#     barycentres[:, 0] = ra
#     barycentres[:, 1] = dec
#     barycentres[:, 2] = redshift

# # record void lambda value, even though usefulness of this has only really been shown for ZOBOV voids so far
# void_lambda = avgdens * (rads ** 1.2)

# # create output array
# output = np.zeros((len(rawdata), 9))
# output[:, 0] = rawdata[:, 0]
# output[:, 1] = xpos
# output[:, 2] = ypos
# output[:, 3] = zpos
# output[:, 4] = rads
# output[:, 5] = mindens
# output[:, 6] = avgdens
# output[:, 7] = void_lambda
# output[:, 8] = densratio

# print('Total %d voids pass all cuts' % len(output))
# sys.stdout.flush()

# # sort in increasing order of minimum density
# sort_order = np.argsort(output[:, 5])
# output = output[sort_order]
# if self.use_barycentres:
#     barycentres = barycentres[sort_order]
# # save to file
# catalogue_file = self.output_folder + self.void_prefix + '_cat.txt'
# header = '%d voxels, %d voids\n' % (nvox, len(output))
# if self.is_box:
#     header += 'VoidID XYZ[3](Mpc/h) R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio'
# else:
#     header += 'VoidID RA Dec z R_eff(Mpc/h) delta_min delta_avg lambda_v DensRatio'
# np.savetxt(catalogue_file, output, fmt='%d %0.4f %0.4f %0.4f %0.4f %0.6f %0.6f %0.6f %0.6f', header=header)

# if self.use_barycentres:
#     if not os.access(self.output_folder + "barycentres/", os.F_OK):
#         os.makedirs(self.output_folder + "barycentres/")
#     catalogue_file = self.output_folder + 'barycentres/' + self.void_prefix + '_baryC_cat.txt'
#     output[:, 1:4] = barycentres
#     np.savetxt(catalogue_file, output, fmt='%d %0.4f %0.4f %0.4f %0.4f %0.6f %0.6f %0.6f %0.6f',
#                header=header)
