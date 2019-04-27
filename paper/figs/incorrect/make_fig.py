import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, Normalize
import cmocean 

cmap = cmocean.cm.dense_r

save = True

img_id = 'GDS_wide2_3135' # wrong
morphs = 'spheroid', 'disk', 'irregular', 'point_source'


# Plot image ===================================================================
h = fits.getdata('{}/data.fits'.format(img_id))[0,...]
fig = plt.figure(figsize=(5,5))

ax = fig.add_subplot(111)
im = ax.imshow(
    h,
    cmap='gray',
    origin='lower',
    interpolation="none",
)

# color bar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.05)
cb = fig.colorbar(im, cax=cax, orientation="vertical")

if save:
	plt.savefig('./{}/flux.png'.format(img_id), dpi=600)
else:
	plt.show()


h = h - h.min() + 1e-5
fig = plt.figure(figsize=(5,5))

ax = fig.add_subplot(111)
im = ax.imshow(
    h,
    cmap='gray',
    origin='lower',
    norm=LogNorm(),
    interpolation="none",
)

# color bar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.05)
cb = fig.colorbar(im, cax=cax, orientation="vertical")

if save:
	plt.savefig('./{}/flux-log.png'.format(img_id), dpi=600)
else:
	plt.show()
# Plot image ===================================================================

# Plot Labels ==================================================================
for morph in morphs:
	fig = plt.figure(figsize=(5,5))
	ax = fig.add_subplot(111)
	im = ax.imshow(fits.getdata('{}/output/top_{}.fits'.format(img_id, morph)),
					cmap=cmap,
					interpolation="none",
					vmin=0,
					vmax=1,
					origin='lower')
					
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="2%", pad=0.05)
	cb = fig.colorbar(im, cax=cax, orientation="vertical")
	
	plt.savefig('./{}/{}.png'.format(img_id, morph), dpi=600)



# Plot Labels ==================================================================


