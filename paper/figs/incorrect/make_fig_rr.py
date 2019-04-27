import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from robertsons_rules.plotting import imshow

save = False

img_id = 'GDS_wide2_3135'
morphs = 'spheroid', 'disk', 'irregular', 'point_source'


# Plot image ===================================================================
h = fits.getdata('{}/data.fits'.format(img_id))[0,...]
if save:
	imshow(h, cmap='gray', out_dir='./flux.pdf')
else:
	imshow(h, cmap='gray')
	plt.show()
# Plot image ===================================================================

# Plot Labels ==================================================================

# Plot Labels ==================================================================


