import numpy as np
from astropy.io import fits

def main():
    morphs = [
        'spheroid',
        'disk',
        'irregular',
        'point_source',
        'background',
    ]
    
    n = fits.getdata('n.fits')
    
    for morph in morphs:
        a = fits.getdata('{}.fits'.format(morph))[-1, :, :]
        
        normed_to_n = np.divide(a, n, out=np.zeros_like(a), where=n!=0)
       
        fits.PrimaryHDU(data=normed_to_n).writeto('top_{}.fits'.format(morph),
                                                  overwrite=True)
        
        
        
    
if __name__=='__main__':
    main()
