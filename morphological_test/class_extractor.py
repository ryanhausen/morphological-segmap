import numpy as np
from astropy.io import fits
from skimage.morphology import watershed
from skimage.filters import sobel
from scipy import ndimage as ndi

def safe_fits(fits_file):
    f = fits.getdata(fits_file)
    img = f.copy()
    del f
    return img

def make_segmap(bkg):
    markers = np.zeros_like(bkg)
    markers[bkg>0.1] = 1
    markers[bkg==0] = 2

    segmented = ndi.binary_fill_holes(watershed(sobel(bkg), markers) - 1)
    labeled, _ = ndi.label(segmented)

    return labeled, segmented

def aggregate_classification(img,
                             src_map,
                             spheroid,
                             disk,
                             irreular,
                             point_source,
                             background):
    return scheme_flux_weighted_simple_mean(img,
                                            src_map,
                                            spheroid,
                                            disk,
                                            irreular,
                                            point_source,
                                            background)

def scheme_flux_weighted_simple_mean(img, src_map, sph, dk, irr, ps, bkg):
    classifications = np.zeros([4])

    for i, m in enumerate([sph, dk, irr, ps]):
        classifications[i] = np.mean(m[src_map] * img[src_map])

    return classifications / classifications.sum()

def get_classification(flux,
                       spheroid,
                       disk,
                       irregular,
                       point_source,
                       background):
    labeled, _ = make_segmap(background)

    id_guess = labeled[labeled.shape[0]//2, labeled.shape[1]//2]
    src_map = labeled==id_guess

    return aggregate_classification(flux,
                                    src_map,
                                    spheroid,
                                    disk,
                                    irregular,
                                    point_source,
                                    background)
