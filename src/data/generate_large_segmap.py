import os

import numpy as np
from astropy.io import fits



def main():
    f_location = 'hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits'
    f160_header = fits.getheader(f_location)
    f160_data = fits.getdata(f_location)

    segmap = np.zeros_like(f160_data, dtype=np.int)

    segmap_files = os.listdir('./segmaps')
    for i, s in enumerate(segmap_files):
        print(i/len(segmap_files), end='\r')
        data = fits.getdata(f'./segmaps/{s}')
        header = fits.getheader(f'./segmaps/{s}')

        x = int(f160_header['CRPIX1'] - header['CRPIX1'])
        y = int(f160_header['CRPIX2'] - header['CRPIX2'])

        segmap[y:y+data.shape[0], x:x+data.shape[1]] = data
        del data
        del header

    fits.PrimaryHDU(header=f160_header, data=segmap).writeto('segmap.fits')





if __name__=='__main__':
    main()
