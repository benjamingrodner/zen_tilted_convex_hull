# Python module
# =============================================================================
# Created By  : Ben Grodner
# Created Date: 2022_01_28
# =============================================================================
"""
Contains functions to preprocess and image, get the background, segment an
image using the neighbor2d algorithm, and get the segmentation properties.
"""
# =============================================================================

# from skimage.util import pad
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes, binary_opening
from skimage.segmentation import watershed, relabel_sequential
from skimage.measure import label, regionprops_table, regionprops
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from numba import cuda, float32, njit
import math
from neighbor2d import line_profile_2d_v2
import skimage.restoration as skr
from skimage.filters import difference_of_gaussians
import scipy.ndimage as ndi
import pandas as pd

# =============================================================================
# Referenced functions
# =============================================================================


def _get_line_matrices(patch_size, phi_range, increment):
    intervals = np.empty((2,))
    line_matrices = np.empty((patch_size, 2, phi_range))
    for phi in range(phi_range):
        angle_index = phi
        intervals[0] = int(np.round(increment * np.cos(phi * np.pi / phi_range)))
        intervals[1] = int(np.round(increment * np.sin(phi * np.pi / phi_range)))
        max_interval = intervals[np.argmax(np.abs(intervals))]
        interval_signs = np.sign(intervals)
        line_n = int(2 * np.abs(max_interval) + 1)
        if line_n < patch_size:
            line_diff = int((patch_size - line_n) / 2)
            for li in range(line_n):
                h1 = interval_signs[0] * li * (2 * np.abs(intervals[0]) + 1) / line_n
                line_matrices[li + line_diff, 0, angle_index] = int(
                    np.sign(h1) * np.floor(np.abs(h1)) + increment - intervals[0]
                )
                h2 = interval_signs[1] * li * (2 * np.abs(intervals[1]) + 1) / line_n
                line_matrices[li + line_diff, 1, angle_index] = int(
                    np.sign(h2) * np.floor(np.abs(h2)) + increment - intervals[1]
                )
            for li in range(line_diff):
                line_matrices[li, :, angle_index] = line_matrices[
                    line_diff, :, angle_index
                ]
            for li in range(line_diff):
                line_matrices[li + line_n + line_diff, :, angle_index] = line_matrices[
                    line_n + line_diff - 1, :, angle_index
                ]
        else:
            for li in range(line_n):
                h1 = interval_signs[0] * li * (2 * np.abs(intervals[0]) + 1) / line_n
                line_matrices[li, 0, angle_index] = int(
                    np.sign(h1) * np.floor(np.abs(h1)) + increment - intervals[0]
                )
                h2 = interval_signs[1] * li * (2 * np.abs(intervals[1]) + 1) / line_n
                line_matrices[li, 1, angle_index] = int(
                    np.sign(h2) * np.floor(np.abs(h2)) + increment - intervals[1]
                )
    return line_matrices


# @njit
def _get_line_profile(lp, image_padded, line_matrices, patch_size):
    for i in range(lp.shape[0]):
        for j in range(lp.shape[1]):
            image_patch = image_padded[i : i + patch_size, j : j + patch_size]
            for t in range(lp.shape[2]):
                for li in range(lp.shape[3]):
                    vli = int(line_matrices[li, 0, t])
                    vlj = int(line_matrices[li, 1, t])
                    lp[i, j, t, li] = image_patch[vli, vlj]
    return lp


# %% codecell
# Use cuda shared memory
@cuda.jit()
def _line_profile_gpu_02(line_profile, image_padded):
    # Assign constant memory
    line_matrices_c = cuda.const.array_like(line_matrices_g)
    tpb_c = cuda.const.array_like(tpb_a)[0]

    # indexing on block
    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # initiate shared arrays
    image_padded_sh = cuda.shared.array(shape=(TPBPS, TPBPS), dtype=float32)
    # line_profile_sh = cuda.shared.array(shape=(TPB, TPB, T, PS), dtype=float32)
    PS = line_matrices_c.shape[0]
    if y < line_profile.shape[0] - PS and x < line_profile.shape[1] - PS:
        # Load shared arrays with values based on the thread location
        image_padded_sh[ty, tx] = image_padded[y, x]

        # Get some threads to load a second value into shared memory
        # add the tile width to the right side
        TPB = tpb_c
        if tx < PS - 1:
            image_padded_sh[ty, TPB + tx] = image_padded[y, TPB + x]
        # bottom side
        if ty < PS - 1:
            image_padded_sh[TPB + ty, tx] = image_padded[TPB + y, x]
        # bottom right corner
        if ((TPB - tx) < (PS - 1)) and ((TPB - ty) < (PS - 1)):
            sx = TPB - tx - 1
            sy = TPB - ty - 1
            image_padded_sh[TPB + sy, TPB + sx] = image_padded[
                y + 2 * sy + 1, x + 2 * sx + 1
            ]
        cuda.syncthreads()

        # Now get the line profiles for the block
        T = line_matrices_c.shape[2]
        image_patch = image_padded_sh[ty : ty + PS, tx : tx + PS]
        for t in range(T):
            for li in range(PS):
                vli = int(line_matrices_c[li, 0, t])
                vlj = int(line_matrices_c[li, 1, t])
                line_profile[y, x, t, li] = image_patch[vli, vlj]


def _get_line_profile_cuda_02(
    lp_shape, image_padded, line_matrices, patch_size, tpb=32, gpu_id=0
):
    # Define Blocks
    threadsperblock = (tpb, tpb)
    blockspergrid_x = int(math.ceil(image_padded.shape[0] / threadsperblock[1]))
    blockspergrid_y = int(math.ceil(image_padded.shape[1] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    global tpb_a
    # global PS
    global TPBPS
    global line_matrices_g
    # global T
    tpb_a = np.array([tpb])
    # PS = line_matrices.shape[0]
    TPBPS = tpb + line_matrices.shape[0]
    # T = line_matrices.shape[2]
    line_matrices_g = line_matrices

    # Select a device to use
    with cuda.gpus[gpu_id]:
        cuda.current_context().push()
        # transfer arrays to device
        line_profile_gm = cuda.device_array(lp_shape)
        image_padded_gm = cuda.to_device(image_padded)
        # line_matrices_gm = cuda.to_device(line_matrices)
        # Run
        _line_profile_gpu_02[blockspergrid, threadsperblock](
            line_profile_gm, image_padded_gm
        )
        cuda.synchronize()
        # transfer back to CPU
        line_profile = line_profile_gm.copy_to_host()
        # del line_profile_gm, image_padded_gm
        # Delete the context
        cuda.current_context().reset()
        cuda.current_context().pop()
    return line_profile


def _evaluate_line_profile(image_lp, window):
    image_lp = np.nan_to_num(image_lp)
    image_lp_min = np.min(image_lp, axis=3)
    image_lp_max = np.max(image_lp, axis=3)
    image_lp_max = image_lp_max - image_lp_min
    image_lp = image_lp - image_lp_min[:, :, :, None]
    image_lp_max = image_lp_max + 1e-5
    image_lp_rel_norm = image_lp / image_lp_max[:, :, :, None]
    image_lp_rnc = image_lp_rel_norm[:, :, :, window]
    image_lprns = np.average(image_lp_rnc, axis=2)
    image_lprn_lq = np.percentile(image_lp_rnc, 25, axis=2)
    image_lprn_uq = np.percentile(image_lp_rnc, 75, axis=2)
    image_lprn_qcv = np.zeros(image_lprn_uq.shape)
    image_lprn_qcv_pre = (image_lprn_uq - image_lprn_lq) / (
        image_lprn_uq + image_lprn_lq + 1e-8
    )
    image_lprn_qcv[image_lprn_uq > 0] = image_lprn_qcv_pre[image_lprn_uq > 0]
    return image_lprns * (1 - image_lprn_qcv)


def _lne(image, window=5, phi_range=9, increment=3):
    image_padded = np.pad(image, window, mode="edge").astype(np.double)
    patch_size = window * 2 + 1
    # lp_shape = (image.shape[0], image.shape[1], phi_range, patch_size)
    # line_profile_empty = np.zeros(lp_shape)
    # line_matrices = _get_line_matrices(patch_size=patch_size, phi_range=phi_range, increment=increment)
    # image_lp = _get_line_profile(lp=line_profile_empty, image_padded=image_padded,
    #                              line_matrices=line_matrices, patch_size=patch_size)
    # image_lp = _get_line_profile_cuda_02(
    #     lp_shape=lp_shape, image_padded=image_padded,
    #     line_matrices=line_matrices, patch_size=patch_size, gpu_id=1
    #     )
    image_lp = line_profile_2d_v2(image_padded, int(patch_size), int(phi_range))
    return _evaluate_line_profile(image_lp=image_lp, window=window)


def _get_rough_segmentation(im_lne, n_clust=2):
    image_final_clustered = MiniBatchKMeans(
        n_clusters=n_clust, batch_size=100000, random_state=42
    )
    image_final_clustered = image_final_clustered.fit_predict(im_lne.reshape(-1, 1))
    image_final_clustered = image_final_clustered.reshape(im_lne.shape)
    list_i0 = []
    for i in range(n_clust):
        image_ = im_lne * (image_final_clustered == i)
        i0 = np.average(image_[image_ > 0])
        list_i0.append(i0)
    intensity_rough_seg = image_final_clustered == np.argmax(list_i0)
    intensity_rough_seg = binary_opening(intensity_rough_seg)
    intensity_rough_seg = remove_small_objects(intensity_rough_seg, 1)
    return binary_fill_holes(intensity_rough_seg)


def _normalize_zero_one(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def _get_watershed_seeds():
    peak_local_max(image, min_distance=1, indices=False)


# =============================================================================
# Wrapper functions
# =============================================================================


def max_projection(raw, channels):
    # maximum projection on raw channel(s)
    if len(raw.shape) == 3:
        if channels == "all":
            im_list = [raw[:, :, i] for i in range(raw.shape[2])]
        else:
            im_list = [raw[:, :, i] for i in channels]
        if len(im_list) > 1:
            raw_2D = np.max((np.dstack(im_list)), axis=2)
        else:
            raw_2D = im_list[0]
    else:
        raw_2D = raw
    return raw_2D


def pre_process(image, denoise=False, gauss=0, log=False, diff_gauss=(0,)):
    if log:
        image = np.log10(image + 1e-15)
        image = _normalize_zero_one(image)
    if denoise:
        patch_kw = dict(
            patch_size=5, patch_distance=6  # 5x5 patches
        )  # 13x13 search area
        sigma_est = skr.estimate_sigma(image)
        # image = skr.denoise_nl_means(image, h=0.8*sigma_est, sigma=sigma_est,
        #                             fast_mode=True, **patch_kw)
        image = skr.denoise_tv_chambolle(image, weight=denoise)
    if gauss:
        image = ndi.gaussian_filter(image, sigma=gauss, order=0)
    if len(diff_gauss) == 2:
        image = difference_of_gaussians(image, diff_gauss[0], diff_gauss[1])
        image = _normalize_zero_one(image)
    return image


def get_background_mask(
    image,
    bg_filter=True,
    bg_file=False,
    bg_log=False,
    bg_smoothing=0,
    n_clust_bg=2,
    top_n_clust_bg=1,
    bg_threshold=0,
):
    if bg_filter:
        if bg_file:
            return np.load(bg_file) > 0
        else:
            if bg_threshold:
                return image > bg_threshold
            else:
                if bg_log:
                    image = np.log10(image + 1)
                    image = _normalize_zero_one(image)
                if bg_smoothing:
                    image = ndi.gaussian_filter(image, sigma=bg_smoothing, order=0)
                cluster = MiniBatchKMeans(
                    n_clusters=n_clust_bg, batch_size=100000, random_state=42
                )
                shape_ = image.reshape(np.prod(image.shape), 1)
                image_bkg_filter = cluster.fit_predict(shape_)
                image_bkg_filter = image_bkg_filter.reshape(image.shape)
                i_list = []
                for n in range(n_clust_bg):
                    image_ = image * (image_bkg_filter == n)
                    i_n = np.average(image_[image_ > 0])
                    i_list.append(i_n)
                i_list = np.argsort(i_list)[::-1]
                _image_bkg_filter_mask = np.zeros(image_bkg_filter.shape, dtype=bool)
                for tn in range(top_n_clust_bg):
                    _image_bkg_filter_mask += image_bkg_filter == i_list[tn]
                return _image_bkg_filter_mask
    else:
        return np.ones(image.shape, dtype=bool)


# def segment(image, window=5, n_clust=2, bg_log=False, bg_smoothing=0, bg_filter=True,
#             n_clust_bg=2, top_n_clust_bg=1, bg_threshold=0, small_objects=50):
def segment(
    image,
    background_mask=np.array([]),
    window=5,
    n_clust=2,
    small_objects=50,
    seed_image="input",
):
    im_lne = _lne(image, window=window)
    rough_seg = _get_rough_segmentation(im_lne, n_clust=n_clust)
    # background_filter = _get_background_filter(image, bg_log=bg_log, bg_smoothing=bg_smoothing,
    #                                            bg_filter=bg_filter, n_clust_bg=n_clust_bg,
    #                                            top_n_clust_bg=top_n_clust_bg, bg_threshold=bg_threshold)
    background_filter = (
        background_mask if background_mask.shape[0] > 0 else np.ones(image.shape)
    )
    watershed_input = image * background_filter
    # seeds = label(rough_seg)
    seeds = label(peak_local_max(image, min_distance=1, indices=False))
    # if isinstance(seed_image, str):
    #     if seed_image == 'input':
    #         seeds = peak_local_max(image, min_distance=1, indices=False)
    #     elif seed_image == 'lne':
    #         seeds = peak_local_max(im_lne, min_distance=1, indices=False)
    # mask = background_filter
    # print('yay')
    mask = rough_seg * background_filter
    # im_seg = rough_seg*background_filter
    im_seg = watershed(-watershed_input, seeds, mask=mask, watershed_line=True)
    im_seg = label(im_seg)
    return remove_small_objects(im_seg, small_objects)

def segment_02(
    image,
    background_mask=np.array([]),
    window=5,
    n_clust=2,
    small_objects=50,
    seed_image="input",
):
    im_lne = _lne(image, window=window)
    rough_seg = _get_rough_segmentation(im_lne, n_clust=n_clust)
    background_filter = (
        background_mask if background_mask.shape[0] > 0 else np.ones(image.shape)
    )
    watershed_input = image * background_filter
    seeds = label(peak_local_max(image*rough_seg, min_distance=1, indices=False))
    # seeds = label(rough_seg)
    mask = background_filter
    im_seg = watershed(-watershed_input, seeds, mask=mask, watershed_line=False)
    im_seg = label(im_seg)
    return remove_small_objects(im_seg, small_objects)

def measure_regionprops(seg, raw=None):
    if isinstance(raw, type(None)):
        raw = np.zeros(seg.shape)
    sp_ = regionprops(seg, intensity_image=raw)
    properties = [
        "label",
        "centroid",
        "area",
        "max_intensity",
        "mean_intensity",
        "min_intensity",
        "bbox",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "eccentricity",
        "perimeter",
    ]
    df = pd.DataFrame([])
    for p in properties:
        df[p] = [s[p] for s in sp_]
    for j in range(2):
        df["centroid-" + str(j)] = [r["centroid"][j] for i, r in df.iterrows()]
    for j in range(4):
        df["bbox-" + str(j)] = [r["bbox"][j] for i, r in df.iterrows()]
    # regions = regionprops_table(seg, intensity_image = raw,
    #                             properties=['label','centroid','area','max_intensity',
    #                             'mean_intensity','min_intensity', 'bbox',
    #                             'major_axis_length', 'minor_axis_length',
    #                             'orientation','eccentricity','perimeter'])
    # return pd.DataFrame(regions)
    return df


def measure_regionprops_3d(seg, raw=None):
    if isinstance(raw, type(None)):
        raw = np.zeros(seg.shape)
    sp_ = regionprops(seg, intensity_image=raw)
    properties = [
        "label",
        "centroid",
        "area",
        "max_intensity",
        "mean_intensity",
        "min_intensity",
        "bbox",
        "major_axis_length",
        "minor_axis_length",
    ]
    df = pd.DataFrame([])
    for p in properties:
        df[p] = [s[p] for s in sp_]
    for j in range(2):
        df["centroid-" + str(j)] = [r["centroid"][j] for i, r in df.iterrows()]
    for j in range(4):
        df["bbox-" + str(j)] = [r["bbox"][j] for i, r in df.iterrows()]
    # regions = regionprops_table(seg, intensity_image = raw,
    #                             properties=['label','centroid','area','max_intensity',
    #                             'mean_intensity','min_intensity', 'bbox',
    #                             'major_axis_length', 'minor_axis_length',
    #                             'orientation','eccentricity','perimeter'])
    # return pd.DataFrame(regions)
    return df


def seg_2_rgb(seg, dict_lab_col, dict_lab_bbox):
    """
    Convert segmentation numpy array to RGB image

    seg - 2d array(int or str) - objects are adjacent pixels with the same value
    dict_lab_col - dictionary(int or str -> list or tup) - map objects in seg to an RGB or RGBA color
    dict_lab_bbox - dictinoary(int or str -> list(int) or tup(int) or str(comma separated integers)) -
        map objects in seg to a bounding box
        bounding box has format (row min, column min, row max, column max)
    """
    rgb_shape = len(dict_lab_col[list(dict_lab_col.keys())[0]])  # Determine RGB or RGBA
    rgb = np.zeros(seg.shape + (rgb_shape,))  # initiate empty image
    for l, b in dict_lab_bbox.items():
        b = eval(b) if isinstance(b, str) else b  # convert to list if string
        rgb_sub = rgb[b[0] : b[2], b[1] : b[3]]  # extract current RGB bbox
        seg_sub = seg[b[0] : b[2], b[1] : b[3]] == l  # Extract object
        color = np.array(dict_lab_col[l])  # Get the object's new color
        rgb_nonobj = rgb_sub * np.dstack(
            [~seg_sub] * rgb_shape
        )  # keep everything already existing in the rgb
        rgb_obj = seg_sub[:, :, None] * color[None, :]  # recolor the object
        rgb[b[0] : b[2], b[1] : b[3], :] = (
            rgb_nonobj + rgb_obj
        )  # write the recolored object to the rgb
    return rgb


def expand_sn(string, sample_names):
    return [string.format(sample_name=sn) for sn in sample_names]


def expand_channels(string, seg_type, sample_names, config):
    channels = config[seg_type]["channels"]
    return [
        string.format(sample_name=sn, channel=ch)
        for ch in channels
        for sn in sample_names
    ]


def expand_all_channels(string, sample_names, config):
    ch_cell = config["cell_seg"]["channels"]
    ch_spot = config["spot_seg"]["channels"]
    return [
        string.format(sample_name=sn, channel_cell=ch_c, channel_spot=ch_s)
        for ch_s in ch_spot
        for ch_c in ch_cell
        for sn in sample_names
    ]


def re_segment_with_classif(seg, classification, bboxes, labels):
    shape = classification.shape
    seg_alt = np.zeros(shape)
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        if isinstance(bbox, str):
            bbox = eval(bbox)
        box_seg_bool = seg[bbox[0] : bbox[2], bbox[1] : bbox[3]] == label
        box_cl = classification[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        for bc in np.unique(box_cl):
            box_cl_bc = box_cl == bc
            box_alt_bool = (box_cl_bc * box_seg_bool) * 1.0
            lab = float(np.max(seg_alt) + 1)
            box_alt_lab = box_alt_bool * lab
            box_im_alt = seg_alt[bbox[0] : bbox[2], bbox[1] : bbox[3]]
            box_alt = box_im_alt + box_alt_lab
            seg_alt[bbox[0] : bbox[2], bbox[1] : bbox[3]] = box_alt
    seg_alt = seg_alt.astype(np.int64)
    return seg_alt


def cluster_averages(image, n_clusters):
    cluster = MiniBatchKMeans(n_clusters=n_clusters, batch_size=100000, random_state=42)
    shape_ = image.reshape(np.prod(image.shape), 1)
    image_bkg_filter = cluster.fit_predict(shape_)
    image_bkg_filter = image_bkg_filter.reshape(image.shape)
    averages = []
    for n in range(n_clusters):
        image_ = image * (image_bkg_filter == n)
        averages.append(np.average(image_[image_ > 0]))
    return averages


def seg_2_rgb_dict(seg, dict_lab_col, dict_lab_bbox, raw=None):
    """
    Convert segmentation numpy array to RGB image based on dictionary mapping

    seg - 2d array(int or str) - objects are adjacent pixels with the same value
    dict_lab_col - dictionary(int or str -> list or tup) - map objects in seg to an RGB or RGBA color
    dict_lab_bbox - dictinoary(int or str -> list(int) or tup(int) or str(comma separated integers)) -
        map objects in seg to a bounding box
        bounding box has format (row min, column min, row max, column max)
    """
    rgb_shape = len(dict_lab_col[list(dict_lab_col.keys())[0]])  # Determine RGB or RGBA
    rgb = np.zeros(seg.shape + (rgb_shape,))  # initiate empty image
    for l, b in dict_lab_bbox.items():
        b = eval(b) if isinstance(b, str) else b  # convert to list if string
        rgb_sub = rgb[b[0] : b[2], b[1] : b[3]]  # extract current RGB bbox
        seg_sub = seg[b[0] : b[2], b[1] : b[3]] == l  # Extract object
        color = np.array(dict_lab_col[l])  # Get the object's new color
        rgb_nonobj = rgb_sub * np.dstack(
            [~seg_sub] * rgb_shape
        )  # keep everything already existing in the rgb
        if raw is None:
            raw_sub = np.ones_like(seg_sub)  
        else:
            raw_sub = raw[b[0] : b[2], b[1] : b[3]]
        rgb_obj = seg_sub[:, :, None] * color[None, :] * raw_sub[:,:,None]  # recolor the object
        rgb[b[0] : b[2], b[1] : b[3], :] = (
            rgb_nonobj + rgb_obj
        )  # write the recolored object to the rgb
    return rgb