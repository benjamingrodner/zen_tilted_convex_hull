# Fuctions for dealing with HiPRFISH spectral images from zeiss i880

# =============================================================================
# Imports
# =============================================================================

from skimage.registration import phase_cross_correlation
import numpy as np
import pandas as pd
from skimage.measure import regionprops
# import javabridge
# import bioformats
import xml.etree.ElementTree as ET
import re
import aicspylibczi as aplc
from cv2 import resize, INTER_CUBIC, INTER_NEAREST
from scipy import stats


# =============================================================================
# Functions
# =============================================================================


def _shift_images(image_stack, shift_vectors, max_shift):
    image_registered = [np.zeros(image.shape) for image in image_stack]
    # shift_filter_mask = [
    #         np.full(
    #                 (image.shape[0], image.shape[1]),
    #                 False, dtype = bool
    #                 )
    #         for image in image_stack
    #         ]
    # ims_arr = np.array([ims.shape for ims in image_stack])
    # ims_shft = []
    # for ims, shft in zip(ims_arr, np.array(shift_vectors)):
    #     ims_shft.append(np.diff(ims+shft))
    # ims_max = np.max(image_shapes, axis=0)
    # shfts_max = np.max(np.array(shift_vectors), axis=0)
    for i in range(len(image_stack)):
        image_shape = image_stack[i].shape
        shift_row = int(shift_vectors[i][0])
        shift_col = int(shift_vectors[i][1])
        print(i, shift_row, shift_col)
        if np.abs(shift_row) > max_shift:
            shift_row = 0
        if np.abs(shift_col) > max_shift:
            shift_col = 0
        original_row_min = int(np.maximum(0, shift_row))
        original_row_max = int(image_shape[0] + np.minimum(0, shift_row))
        original_col_min = int(np.maximum(0, shift_col))
        original_col_max = int(image_shape[1] + np.minimum(0, shift_col))
        registered_row_min = int(-np.minimum(0, shift_row))
        registered_row_max = int(image_shape[0] - np.maximum(0, shift_row))
        registered_col_min = int(-np.minimum(0, shift_col))
        registered_col_max = int(image_shape[1] - np.maximum(0, shift_col))
        image_registered[i][original_row_min: original_row_max, original_col_min: original_col_max, :] = image_stack[i][registered_row_min: registered_row_max, registered_col_min: registered_col_max, :]
        # shift_filter_mask[i][original_row_min: original_row_max, original_col_min: original_col_max] = True
    return image_registered


def _get_shift_vectors(image_sum):
    # Find shift vectors
    shift_vectors = [
            phase_cross_correlation(
                    np.log(image_sum[0]+1), np.log(image_sum[i]+1)
                    )[0]
            for i in range(1,len(image_sum))
            ]
    shift_vectors.insert(0, np.asarray([0.0,0.0]))
    return shift_vectors


def _size_images(image_list):
    print([im.shape for im in image_list])
    max_r = np.max([im.shape[0] for im in image_list])
    max_c = np.max([im.shape[1] for im in image_list])
    image_stack = []
    for im in image_list:
        im_resz = np.zeros((max_r, max_c, im.shape[2]))
        im_resz[:im.shape[0], :im.shape[1], :im.shape[2]] = im
        image_stack.append(im_resz)
    return image_stack


def register_shifts(image_list, max_shift=20):
    # Make all images the same size in case stitching made variations
    image_stack = _size_images(image_list)
    # Get projection for each channel
    image_sum = [np.sum(image, axis = 2) for image in image_stack]
    # Get the shifts
    shift_vectors = _get_shift_vectors(image_sum)
    print(shift_vectors)
    # Shift the images
    image_registered = _shift_images(image_stack, shift_vectors, max_shift)
    shp = np.min([ims.shape[:2] for ims in image_registered], axis=0)
    print(shp)
    image_registered_trimmed = [im[:shp[0],:shp[1],:] for im in image_registered]
    # Get outputs
    image_registered_cat = np.concatenate(image_registered_trimmed, axis = 2)
    image_registered_max = np.max(image_registered_cat, axis = 2)
    image_registered_sum = np.sum(image_registered_cat, axis = 2)
    return(image_registered_cat, image_registered_max, image_registered_sum, shift_vectors)


def get_cell_average_spectra(seg, raw):
    n_cells = np.unique(seg).shape[0] - 1
    avgint = np.empty((n_cells, raw.shape[2]))
    for k in range(0, raw.shape[2]):
        cells = regionprops(seg, intensity_image = raw[:,:,k])
        avgint[:,k] = [x.mean_intensity for x in cells]
    return avgint


def plot_cell_spectra(ax, arr_spec, kwargs):
    x = np.arange(arr_spec.shape[1])
    X = np.ones(arr_spec.shape) * x
    ax.plot(X.T, arr_spec.T, **kwargs)
    return(ax)


def recolor_image(im, dict_label_bbox, dict_label_alt, threeD=0):
    shape = im.shape + (threeD,) if threeD else im.shape
    im_alt = np.zeros(shape)
    for label, bbox in dict_label_bbox.items():
        if isinstance(bbox, str):
            bbox = eval(bbox)
        box = (im[bbox[0]:bbox[2],bbox[1]:bbox[3]] == label)*1
        if label == 1: print(np.unique(box))
        box = box[...,None] if threeD else box
        alt = dict_label_alt[label]
        alt = eval(alt) if isinstance(alt, str) else alt
        box_alt = (box * alt) + im_alt[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        # if label==1: print(box_alt.shape, np.unique(box_alt))
        im_alt[bbox[0]:bbox[2],bbox[1]:bbox[3]] = box_alt
    return(im_alt)


def plot_nn_dists(ax, nn_dists, kwargs):
    nn_dists_sort = np.sort(nn_dists)
    x = np.arange(nn_dists_sort.shape[0]) + 1
    ax.scatter(x, nn_dists_sort, **kwargs)
    return ax


def get_metadata_value_old(czi_fn, search_term):
    ims_meta_str = bioformats.get_omexml_metadata(path=czi_fn)
    ims_meta = ET.fromstring(ims_meta_str)
    name_space = re.search('(?<=xmlns=)["a-zA-Z0-9://.-]+',ims_meta_str)[0]
    name_space_fmt = "{{{}}}".format(name_space[1:-1])
    # name_space_fmt = "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"
    origin_meta_datas = ims_meta.findall(".//{}OriginalMetadata".format(name_space_fmt))
    # Iterate in founded origins
    values = []
    for origin in origin_meta_datas:
        # print(origin)
        key = origin.find("{}Key".format(name_space_fmt)).text
        # print(key)
        if search_term in key:
            value = origin.find("{}Value".format(name_space_fmt)).text
            values.append(value)
            # print("Value: {}".format(value))
    return values


def get_metadata_value(fn, search_term):
    czi = aplc.CziFile(fn)
    results = []
    for n in czi.meta.iter():
        if search_term in n.tag:
            results.append(n.text)
    return results

def start_javabridge():
    javabridge.start_vm(class_path=bioformats.JARS)
    return


def get_resolution(fn, dim='X'):
    czi = aplc.CziFile(fn)
    for n in czi.meta.iter():
        if 'Scaling' in n.tag:
            if dim in n.tag:
                resolution = float(n.text)
    return resolution

def get_laser(fn):
    czi = aplc.CziFile(fn)
    for n in czi.meta.iter():
        if 'ExcitationWavelength' in n.tag:
            # if dim in n.tag:
            laser = int(float(n.text))
    return laser

def load_raw(fn, m, mtype):
    czi = aplc.CziFile(fn)
    if mtype == 'M':
        raw, sh = czi.read_image(M=m)
    elif mtype == 'S':
        raw, sh = czi.read_image(S=m)
    else:
        raw, sh = czi.read_image()
    return raw

def get_ntiles(fn):
    dimshape = aplc.CziFile(fn).get_dims_shape()[0]
    keys = list(dimshape.keys())
    if 'M' in keys:
        M = dimshape['M'][1]
        mtype = 'M'
    elif 'S' in keys:
        M = dimshape['S'][1]
        mtype = 'S'
    else:
        M = 1
        mtype = ''
    return M, mtype


def reshape_aics_image(m_img):
    '''
    Given an AICS image with just XY and CHannel,
    REshape into shape (X,Y,C)
    '''
    img = np.squeeze(m_img)
    img = np.transpose(img, (1,2,0))
    return img


def center_image(im, dims, ul_corner):
    shp = im.shape
    if not all([dims[i] == shp[i] for i in range(len(dims))]):
        shp_new = dims if len(shp) == 2 else dims + (shp[2],)
        temp = np.zeros(shp_new)
        br_corner = np.array(ul_corner) + np.array(shp[:2])
        temp[ul_corner[0]:br_corner[0], ul_corner[1]:br_corner[1]] = im
        im = temp
    return im


def resize_hipr(im, hipr_res, mega_res, dims='none', ul_corner=(0,0)):
    # im = np.load(in_fn)
    factor_resize = hipr_res / mega_res
    hipr_resize = resize(
            im,
            None,
            fx = factor_resize,
            fy = factor_resize,
            interpolation = INTER_NEAREST
            )
    if isinstance(dims, str): dims = hipr_resize.shape
    hipr_resize = center_image(hipr_resize, dims, ul_corner)
    # if out_fn: np.save(out_fn, hipr_resize)
    return hipr_resize


def match_resolutions_and_size(raws, resolutions):
    res_mode = stats.mode(resolutions)[0][0]
    for r, res in zip(raws, resolutions):
        if res == res_mode:
            shape_std = np.array(r.shape[:2])
    for i, (r, res) in enumerate(zip(raws, resolutions)):
        if res > res_mode:
            r_resize = resize_hipr(r, res, res_mode)
            # print(r_resize.shape, shape_std)
            c = (np.array(r_resize.shape[:2]) - shape_std) // 2
            raws[i] = r_resize[
                c[0]:c[0] + shape_std[0], 
                c[1]:c[1] + shape_std[1],
                :
            ]
        elif res < res_mode:
            raise ValueError('Can currently only resize if resolution is lower')
    return raws


def max_norm(raw, c=['min','max'], type='max'):
    if type == 'max':
        im = np.max(raw, axis=2)
    elif type == 'sum':
        im = np.sum(raw, axis=2)
    mn = np.min(im) if c[0] == 'min' else c[0]
    mx = np.max(im) if c[1] == 'max' else c[1]
    im = np.clip(im, mn, mx)
    return (im - mn) / (mx - mn)


def replace_outlier_shifts(sh_i, iqr_mult=1.5):
    if len(sh_i) > 2:
        q = np.quantile(sh_i, [0.25, 0.5, 0.75], axis=0)
        iqr = q[2] - q[0]
        ol_plus = q[2] + iqr_mult * iqr
        ol_minus = q[0] - iqr_mult * iqr
        shifts_red = []
        inds_replace = []
        for k, s in enumerate(sh_i):
            bool_std = any(s > ol_plus) or (any(s < ol_minus))
            bool_z = all(s == np.array([0, 0]))
            if bool_std or bool_z:
                inds_replace.append(k)
            else:
                shifts_red.append(s)
        if inds_replace:
            sh_mean = np.median(shifts_red, axis=0).astype(int)
            for k in inds_replace:
                # print('Replaced', sh_i[k,:], 'with', sh_mean)
                sh_i[k, :] = sh_mean
    return sh_i


def get_reference_spectra(barcodes, bc_type, ref_dir, fmt="08_18_2018_enc_{}_avgint.csv"):
    if bc_type == '5bit_no633':
        barcodes_str = [str(bc).zfill(5) for bc in barcodes]
        barcodes_10bit = [bc[0] + "0" + bc[1] + "0000" + bc[2:] for bc in barcodes_str]
    elif bc_type == '7bit_no405':
        barcodes_str = [str(bc).zfill(7) for bc in barcodes]
        barcodes_10bit = [bc[0] + '0' + bc[1:4] + '00' + bc[4:] for bc in barcodes_str]
    barcodes_b10 = [int(str(bc), 2) for bc in barcodes_10bit]
    st = 32
    en = 32 + 57
    ref_avgint_cols = [i for i in range(st, en)]

    ref_spec = []
    for bc in barcodes_b10:
        fn = ref_dir + "/" + fmt.format(bc)
        ref = pd.read_csv(fn, header=None)
        ref = ref[ref_avgint_cols].values
        ref_spec.append(ref)
    return ref_spec