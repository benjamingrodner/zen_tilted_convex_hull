import os
import numpy as np
import pandas as pd
from PIL import Image
import aicspylibczi as aplc
import xml.etree.ElementTree as ET

def get_resolution(fn, dim='X'):
    czi = aplc.CziFile(fn)
    for n in czi.meta.iter():
        if 'Scaling' in n.tag:
            if dim in n.tag:
                resolution = float(n.text)
    return resolution

def get_metadata_value(fn, search_term):
    czi = aplc.CziFile(fn)
    results = []
    for n in czi.meta.iter():
        if search_term in n.tag:
            results.append(n.text)
    return results

def load_raw(fn, m, mtype):
    czi = aplc.CziFile(fn)
    if mtype == 'M':
        raw, sh = czi.read_image(M=m)
    elif mtype == 'S':
        raw, sh = czi.read_image(S=m)
    else:
        raw, sh = czi.read_image()
    return raw

def reshape_aics_image(m_img):
    '''
    Given an AICS image with just XY and CHannel,
    REshape into shape (X,Y,C)
    '''
    img = np.squeeze(m_img)
    img = np.transpose(img, (1,2,0))
    return img

def main():
    """
    sc_stitch_prep.py
    Author: Ben Grodner
    Last Edit: 17 Apr 2024

    Purpose:
        Convert positions from μm to pixels so that FIJI can do stitching.
        Also save czi scenes as tiff for FIJI
    """
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "-ifn",
        "--image_filename",
        dest="image_filename",
        type=str,
        help="Image czi file where tiles are saved as scenes.",
    )
    parser.add_argument(
        "-chf",
        "--convex_hull_fn",
        dest="convex_hull_fn",
        type=str,
        help="Positions csv used to do the tile scan.",
    )
    parser.add_argument("-od", "--output_dir", dest="output_dir", type=str, help="Output directory filepath")
    args = parser.parse_args()

    # Make output dir
    if not os.path.exists(args.ouput_dir): 
        os.makedirs(args.ouput_dir)
        print('Made dir:', args.output_dir)

    # Write tif files 
    ntiles = int(get_metadata_value(args.image_filename, 'SizeS')[0])
    for s in range(ntiles):
        raw = load_raw(args.image_filename, s, 'S')
        raw = reshape_aics_image(raw)
        im_sum = np.sum(raw, axis=2).astype(np.uint8)
        im = Image.fromarray(im_sum)
        im.save(args.output_dir + '/' + str(s) + '.tif')
    
    # Read positions file
    coords = pd.read_csv(args.convex_hull_fn, header=None)
    coords.columns = ["x", "y", "z"]

    # Convert to pix
    res_umpix = get_metadata_value(args.image_filename, 'ScalingX') * 1e6
    coords["xpix"] = coords["x"] / res_umpix
    coords["ypix"] = coords["y"] / res_umpix

    # Write new coords file for FIJI format
    lines = ['dim = 2']
    for s in range(12):
        xy = tuple(coords.loc[s,['xpix','ypix']].values)
        line = str(s) + '.tif;;' + str(xy)
        lines.append(line)
        
    coords_yinv_fn = args.output_dir + '/stitch_coords_pix.txt'
    with open(coords_yinv_fn, 'w') as f:
        for l in lines:
            f.write(l + '\n')
    print('Wrote:', coords_yinv_fn)