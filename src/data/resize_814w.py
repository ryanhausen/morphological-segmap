import numpy as np

from astropy.extern import six

from reproject.utils import parse_input_data, parse_output_projection
from reproject.interpolation.core_celestial import _reproject_celestial
from reproject.interpolation.high_level import ORDER

#https://github.com/astrofrog/reproject/issues/37#issuecomment-267645100
def reproject_interp_chunk_2d(input_data, output_projection, shape_out=None, hdu_in=0,
                              order='bilinear', blocks=(1000, 1000)):
    """
    For a 2D image, reproject in chunks
    """

    array_in, wcs_in = parse_input_data(input_data, hdu_in=hdu_in)
    wcs_out, shape_out = parse_output_projection(output_projection, shape_out=shape_out)

    if isinstance(order, six.string_types):
        order = ORDER[order]

    # Create output arrays
    array = np.zeros(shape_out, dtype=np.float32)
    footprint = np.zeros(shape_out, dtype=np.float32)

    for imin in range(0, array.shape[0], blocks[0]):
        imax = min(imin + blocks[0], array.shape[0])
        for jmin in range(0, array.shape[1], blocks[1]):
            jmax = min(jmin + blocks[1], array.shape[1])
            shape_out_sub = (imax - imin, jmax - jmin)
            wcs_out_sub = wcs_out.deepcopy()
            wcs_out_sub.wcs.crpix[0] -= jmin
            wcs_out_sub.wcs.crpix[1] -= imin
            array_sub, footprint_sub = _reproject_celestial(array_in, wcs_in, wcs_out_sub,
                                                            shape_out=shape_out_sub, order=order)
            array[imin:imax, jmin:jmax] = array_sub
            footprint[imin:imax, jmin:jmax] = footprint_sub

    return array, footprint


if __name__ == '__main__':

    from astropy.io import fits

    hdu_in = fits.open('hlsp_candels_hst_acs_gs-tot_f814w_v1.0_drz.fits')[0]
    hdu_out = fits.open('hlsp_candels_hst_wfc3_gs-tot_f125w_v1.0_drz.fits')[0]

    array, footprint = reproject_interp_chunk_2d(hdu_in, hdu_out.header, blocks=(1000, 1000))

    fits.writeto('hlsp_candels_hst_acs_gs-tot_f814w_v1.0_drz.fits', array, overwrite=True)
