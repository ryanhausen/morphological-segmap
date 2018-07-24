
import numpy as np
from astropy.io import fits

def main():


    f160 = fits.getdata('../../data/big_imgs/hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits')
    f814 = fits.getdata('../../data/big_imgs/hlsp_candels_hst_acs_gs-tot_f814w_v1.0_drz.fits')

    for i in range(f160.shape[0]):
        print(i/f160.shape[0], end='\r')
        if f814[i,:].sum() != 0:
            for j in range(f160.shape[1]):
                if f160[i,j]==0:
                    f814[i,j] = 0

    header = fits.getheader('../../data/big_imgs/hlsp_candels_hst_acs_gs-tot_f814w_v1.0_drz.fits')
    fits.PrimaryHDU(header=header, data=f814).writeto('../../data/big_imgs/f814w_filtered.fits')


if __name__=='__main__':
    main()
