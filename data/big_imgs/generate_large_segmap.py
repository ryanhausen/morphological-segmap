import numpy as np
import sep
from astropy.io import fits



def main():
    a = fits.getdata('hlsp_candels_hst_wfc3_gs-tot_f160w_v1.0_drz.fits')
    mask = a!=0
    bkg = sep.Background(a.byteswap().newbyteorder(), mask=mask)
    ab = a - bkg
    objects, segmap = sep.extract(ab, 5, err=bkg.globalrms, mask=mask)
    fits.PrimaryHDU(segmap).writeto('segmap.fits')
    

if __name__=='__main__':
    main()
