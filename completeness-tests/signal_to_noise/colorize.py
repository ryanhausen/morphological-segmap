import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from astropy.io import fits


# all vals must be 0-1
RED = 0/3
GREEN = 1/3
BLUE = 2/3

ys = xs = slice(5, -5)
sph = fits.getdata('deVaucouleurs-re-9/top_spheroid.fits')[ys, xs]
dk = fits.getdata('deVaucouleurs-re-9/top_disk.fits')[ys, xs]
irr = fits.getdata('deVaucouleurs-re-9/top_irregular.fits')[ys, xs]
ps = fits.getdata('deVaucouleurs-re-9/top_point_source.fits')[ys, xs]
bkg = fits.getdata('deVaucouleurs-re-9/top_background.fits')[ys, xs]

hues = np.dstack([sph, dk, ps])
sorted_hues = np.argsort(-hues, axis=-1)

starting_hues = sorted_hues[:,:,0]/3

hue_shift = np.zeros_like(sph)
for i in range(hue_shift.shape[0]):
    for j in range(hue_shift.shape[1]):
        idx = sorted_hues[i, j, 1]
        hue_shift[i, j] = hues[i, j, idx]

hue_shift = hue_shift / 3

hue_shift_direction = sorted_hues[:,:,1]-sorted_hues[:,:,0]

starting_hues[hue_shift_direction==-2] = 1.0
hue_shift_direction[hue_shift_direction==-2] = -1
hue_shift_direction[hue_shift_direction==2] = 1

hue_shift_direction = hue_shift * hue_shift_direction

hsv = np.zeros([sph.shape[0], sph.shape[1], 3])

hsv[:,:,0] = starting_hues + hue_shift_direction
hsv[:,:,1] = 1 - irr
hsv[:,:,2] = 1 - bkg

rgb = hsv_to_rgb(hsv)

plt.figure()
plt.imshow(starting_hues, origin='bottom')


plt.figure()
plt.imshow(rgb, origin='bottom')


plt.figure()
plt.imshow(hue_shift_direction, origin='bottom')

plt.show()




