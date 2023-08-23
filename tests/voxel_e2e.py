import fitsio
from pathlib import Path
import numpy as np
from revolver import VoxelVoids
from pycorr import TwoPointCorrelationFunction
import matplotlib.pyplot as plt

data_dir = './'
data_fn = Path(data_dir) / 'mock_data.fits'
data = fitsio.read(data_fn)

x = data['X']
y = data['Y']
z = data['Z']
data_positions = np.c_[x, y, z]

boxsize = 1500
boxcenter = boxsize / 2
wrap = True
cellsize = 5.0
boxpad = 1.0

voxel = VoxelVoids(
    data_positions=data_positions,
    boxsize=boxsize,
    boxcenter=boxcenter,
    wrap=wrap,
    boxpad=boxpad,
    cellsize=cellsize,
)

voxel.set_density_contrast(smoothing_radius=10)
voxel.find_voids()
voids_positions, voids_radii = voxel.postprocess_voids()

redges = np.hstack([np.arange(0, 5, 1),
                    np.arange(7, 30, 3),
                    np.arange(31, 155, 5)])
muedges = np.linspace(-1, 1, 241)
edges = (redges, muedges)

result = TwoPointCorrelationFunction(
    mode='smu', edges=edges, data_positions1=data_positions,
    data_positions2=voids_positions, estimator='auto', boxsize=boxsize,
    nthreads=256, compute_sepsavg=False, position_type='pos', los='z',
)
s, multipoles = result(ells=(0, 2, 4), return_sep=True)


fig, ax = plt.subplots()
ax.plot(s, multipoles[0])
plt.show()


fig, ax = plt.subplots()
ax.hist(voids_radii, bins=20)
plt.show()

# VoxelVoids(data_positions=data_positions)
# (self, data_positions, boxsize=None, boxcenter=None,
#  18         data_weights=None, randoms_positions=None, randoms_weights=None,
#  19         cellsize=None, wrap=False, boxpad=1.5, nthreads=None)
