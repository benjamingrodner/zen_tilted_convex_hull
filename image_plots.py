import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from copy import copy
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import cm
from matplotlib import colors
# from numba import njit
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
import os
from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.colors as pltcolors

def get_discrete_colorbar(vals, cmp, integers=True):
    l = max(vals)-min(vals)
    cmp_ = cm.get_cmap(cmp,lut=int(l+1))
    cmp_bounds = np.arange(int(l+2)) - 0.5
    norm = colors.BoundaryNorm(cmp_bounds,cmp_.N)
    image=plt.imshow(np.array([list(vals)]), cmap=cmp_, norm=norm)
    plt.gca().set_visible(False)
    cbar = plt.colorbar(image,ticks=vals,orientation="horizontal")
    if integers:
        cbar.set_ticklabels([str(int(v)) for v in vals])
    else:
        cbar.set_ticklabels([str(v) for v in vals])
    return(cbar)


def subplot_square_images(im_list, subplot_dims, im_inches=5, cmaps=(), clims=(), zoom_coords=(), scalebar_resolution=0, axes_off=True, discrete=()):
    sd1, sd2 = subplot_dims
    figsize=(sd2*im_inches,sd1*im_inches)
    # figsize=(sd2*im_inches,1.02375*sd1*im_inches)
    fig, axes = plt.subplots(sd1,sd2, figsize=figsize)
    for i, (ax, im) in enumerate(zip(fig.axes, im_list)):
        cmap = cmaps[i] if cmaps else 'inferno'
        im_ = im[~np.isnan(im)]
        clim = clims[i] if clims else (np.min(im_), np.max(im_))
        clim = clim if clim else (np.min(im_), np.max(im_))
        if cmap:
            ax.imshow(im, cmap=cmap, clim=clim, interpolation="none")
        else:
            ax.imshow(im, clim=clim)
        zc = zoom_coords if zoom_coords else (0,im.shape[0],0,im.shape[1])
        ax.set_ylim(zc[1],zc[0])
        ax.set_xlim(zc[2],zc[3])
        if axes_off:
            ax.set_axis_off()
        if i == 0 and scalebar_resolution:
            scalebar = ScaleBar(scalebar_resolution, 'um', frameon = False, color = 'white', box_color = 'white')
            plt.gca().add_artist(scalebar)
    plt.subplots_adjust(wspace=0,hspace=0,left=0,right=1,bottom=0,top=1)
    discrete = discrete if discrete else np.zeros((len(im_list),))
    cbars = []
    for im, cl, cmp, d in zip(im_list, clims, cmaps, discrete):
        if cmp:
            fig2 = plt.figure(figsize=(9, 1.5))
            if d:
                vals = np.sort(np.unique(im))
                vals = vals[~np.isnan(vals)]
                vals = vals if len(cl)==0 else vals[(vals>=cl[0]) & (vals<=cl[1])]
                cbar = get_discrete_colorbar(vals, d)
            else:
                if cl:
                    image=plt.imshow(im, cmap=cmp, clim=cl)
                else:
                    image=plt.imshow(im, cmap=cmp)
                plt.gca().set_visible(False)
                cbar = plt.colorbar(image,orientation="horizontal")
        cbars.append(fig2)
    return(fig, fig.axes, cbars)


def seg2rgb(seg):
    return label2rgb(seg,  bg_label = 0, bg_color = (0,0,0))

def save_fig(filename, bbox_inches='tight', dpi=500, transp=True):
    plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,transparent=transp)


def _image_figure(dims, dpi=500):
    fig = plt.figure(figsize=(dims[0], dims[1]))
    ax = plt.Axes(fig, [0., 0., 1., 1.], )
    # ax.set_axis_off()
    fig.add_axes(ax)
    return(fig, ax)


def plot_image(
            im, im_inches=5, cmap='inferno', clims=('min','max'), zoom_coords=(), scalebar_resolution=0,
            axes_off=True, discrete=False, cbar_ori='horizontal', dpi=500,
            norm=None, cb_col='white', ft=12
        ):
    s = im.shape
    dims = (im_inches*s[1]/np.max(s), im_inches*s[0]/np.max(s))
    fig, ax = _image_figure(dims, dpi=dpi)
    im_ = im[~np.isnan(im)]
    llim = np.min(im_) if clims[0]=='min' else clims[0]
    ulim = np.max(im_) if clims[1]=='max' else clims[1]
    clims = (llim, ulim)
    if len(s) > 2:
        ax.imshow(im, interpolation="none")
    else:
        ax.imshow(im, cmap=cmap, clim=clims, interpolation="none", norm=norm)
    zc = zoom_coords if zoom_coords else (0,im.shape[0],0,im.shape[1])
    ax.set_ylim(zc[1],zc[0])
    ax.set_xlim(zc[2],zc[3])
    if axes_off:
        ax.set_axis_off()
    if scalebar_resolution:
        scalebar = ScaleBar(
                scalebar_resolution, 'um', frameon = False,
                color = cb_col, box_color = cb_col,
                font_properties={'size':ft}
            )
        plt.gca().add_artist(scalebar)
    cbar = []
    fig2 = []
    if len(s) == 2:
        if cbar_ori == 'horizontal':
            fig2 = plt.figure(figsize=(dims[0], dims[0]/10))
        elif cbar_ori == 'vertical':
            fig2 = plt.figure(figsize=(dims[1]/10, dims[1]))
        if discrete:
            vals = np.sort(np.unique(im))
            vals = vals[~np.isnan(vals)]
            vals = vals[(vals>=clims[0]) & (vals<=clims[1])]
            cbar = get_discrete_colorbar(vals, cmap)
        else:
            image=plt.imshow(im, cmap=cmap, clim=clims, norm=norm)
            plt.gca().set_visible(False)
            cbar = plt.colorbar(image,orientation=cbar_ori)
    return(fig, ax, [fig2, cbar])


def plot_seg_outline(ax, seg, col=(0,1,0)):
    cmap = copy(plt.cm.get_cmap('gray'))
    cmap.set_bad(alpha = 0)
    cmap.set_over(col, 1.0)
    im_line = find_boundaries(seg, mode = 'outer')
    im_line = im_line.astype(float)
    im_line[im_line == 0] = np.nan
    clims = (0,0.9)
    extent = (0,seg.shape[1],0,seg.shape[0])
    ax.imshow(im_line, cmap=cmap, clim=clims, interpolation='none')
    return ax


def get_ROC_curves():
    rd = {}
    # iterate through pos/neg
    for j in J:
        print(i,j)
        sn_fovs = sn_dict[i][j]
        sp_df_all = pd.DataFrame([])
        cell_count = 0
        # Combine fovs
        for k, sn in enumerate(sn_fovs):
            # Get seg
            seg = np.load(seg_dir + '/cell_seg/' + sn[1] + '_cell_seg.npy')
            cell_count += np.unique(seg).shape[0]
            # Get spot params
            sp_df_fn = seg_dir + '/spot_analysis/' + sn[1] + '_max_props_cid.csv'
            sp_df = pd.read_csv(sp_df_fn)
            sp_df['cell_id_fov'] = sp_df.cell_id.astype(str) + '_' + str(k)
            sp_df_all = sp_df_all.append(sp_df)
        # Filter by distance
        sp_df_cell = sp_df_all[(sp_df_all.cell_dist <= max_dist)]
        # Get threshold curves
        # nsrs = [sp_df_cell.loc[(sp_df_cell.intensity < l),'cell_id_fov'] for l in x]
        psrs = [sp_df_cell.loc[(sp_df_cell.intensity >= l),'cell_id_fov'] for l in x]
        rd[j] = {'c':cell_count, 'p':psrs}
        # rd[j] = {'c':cell_count, 'n':nsrs,'p':psrs}
    # calculate values
    # Rs = {}
    # for j in J:
    FPR = [ns.unique().shape[0] / rd['neg']['c'] for ns in rd['neg']['p']]
    TNR = [1-fpr for fpr in FPR]
    TPR = [ns.unique().shape[0] / rd['pos']['c'] for ns in rd['pos']['p']]
    FNR = [1 - tpr for tpr in TPR]
        # Rs[j + '_NR'] = NR
        # Rs[j + '_PR'] = PR
    PPV = [ps.unique().shape[0] / (ps.unique().shape[0] + nps.unique().shape[0])
            for ps, nps in zip(rd['pos']['p'], rd['neg']['p'])]
    FOR = [(rd['pos']['c'] - ps.unique().shape[0]) / ((rd['pos']['c'] -\
            ps.unique().shape[0]) + (rd['neg']['c'] - nps.unique().shape[0]) + 1e-15)
            for ps, nps in zip(rd['pos']['p'], rd['neg']['p'])]
    # Save values
    roc_df = pd.DataFrame({'x':x,'TNR':TNR,'FPR':FPR,'FNR':FNR,'TPR':TPR,'PPV':PPV,'FOR':FOR})
    roc_df.to_csv(roc_df_fnt.format(gfn), index=False)


def general_plot(shp=(1,1), xlabel='', ylabel='', ft=12, dims=(5,3), col='k', lw=1, pad=0):
    fig, ax = plt.subplots(shp[0], shp[1], figsize=(dims[0], dims[1]),  tight_layout={'pad': pad})
    for i in ax.spines:
        ax.spines[i].set_linewidth(lw)
    ax.spines['top'].set_color(col)
    ax.spines['bottom'].set_color(col)
    ax.spines['left'].set_color(col)
    ax.spines['right'].set_color(col)
    ax.tick_params(direction='in', labelsize=ft, color=col, labelcolor=col)
    ax.set_xlabel(xlabel, fontsize=ft, color=col)
    ax.set_ylabel(ylabel, fontsize=ft, color=col)
    ax.patch.set_alpha(0)
    return(fig, ax)


def plot_ROC_curves(roc_df, xlims=[], dims=(5,4), thresholds=[],
                    col='k'):
    fig, ax = ip.general_plot(xlabel='Threshold',dims=dims)
    ax.plot(roc_df.x, roc_df.PPV,label='PPV')
    ax.plot(roc_df.x, roc_df.FOR,label='FOR')
    ax.plot(roc_df.x, roc_df.TPR, label='TPR')
    ax.plot(roc_df.x, roc_df.FPR, label='FPR')
    # ax.plot(roc_df.x, roc_df.TNR, label='TNR')
    # ax.plot(roc_df.x, roc_df.FNR,label='FNR')
    ax.plot([threshold]*2, [0,1], col)
    ax.legend()
    ax.set_xlim(xlims)
    return figs


def filter_seg_objects(seg, props, filter):
    seg_new = seg.copy()
    remove_cids = props.loc[props[filter] == 0, ['label','bbox']].values
    for i, (c, b) in enumerate(remove_cids):
        b = eval(b) if isinstance(b, str) else b
        b_sub = seg_new[b[0]:b[2],b[1]:b[3]]
        b_sub = b_sub * (b_sub != c)
        seg_new[b[0]:b[2],b[1]:b[3]] = b_sub
    return seg_new


def load_output_file(config, fmt, sample_name='', cell_chan='', spot_chan='',channel=''):
    fn = (config['output_dir'] + '/'
                + config[fmt].format(sample_name=sample_name,
                                         channel=channel,
                                         cell_chan=cell_chan,
                                         spot_chan=spot_chan))
    ext = os.path.splitext(fn)[1]
    if ext == '.npy':
        return np.load(fn)
    elif ext == '.csv':
        return pd.read_csv(fn)
    elif ext == '.json':
        with open(fn) as f:
            file = json.load(f)
        return file
    else:
        raise ValueError('must be csv or npy or json file')
    return


def save_png_pdf(basename, bbox_inches='tight', dpi=1000):
    for ext in ['.pdf','.png']:
        fn = basename + ext
        if bbox_inches:
            plt.savefig(fn, transparent=True, bbox_inches=bbox_inches, dpi=dpi)
        else:
            plt.savefig(fn, transparent=True, dpi=dpi)


def load_output_file(config, fmt, sample_name='', cell_chan='', spot_chan=''):
    fn = (config['output_dir'] + '/'
                + config[fmt].format(sample_name=sample_name,
                                         cell_chan=cell_chan,
                                         spot_chan=spot_chan))
    ext = os.path.splitext(fn)[1]
    if ext == '.npy':
        return np.load(fn)
    elif ext == '.csv':
        return pd.read_csv(fn)
    elif ext == '.json':
        with open(fn) as f:
            file = json.load(f)
        return file
    else:
        raise ValueError('must be csv or npy file')
    return


def zero_one_thresholding(im, clims):
    im = im/np.max(im)
    im[im > clims[1]] = clims[1]
    im[im < clims[0]] = clims[0]
    im = (im - clims[0]) / (clims[1] - clims[0])
    return im


def color_bc_image(im_bc, barcodes, cols, im_int=np.array([])):
    threeD = isinstance(cols[0], tuple) or isinstance(cols[0], list)
    if threeD:
        im_col = np.zeros(im_bc.shape + (len(cols[0]),))
        if im_int.shape[0]==0:
            for bc, c in zip(barcodes, cols):
                im_col += (im_bc==bc)[...,None] * c
        else:
            for bc, c in zip(barcodes, cols):
                temp = im_int*(im_bc==bc)
                im_col += temp[...,None] * c
    else:
        im_col = np.zeros(im_bc.shape)
        if im_int.shape[0]==0:
            for bc, c in zip(barcodes, cols):
                im_col += (im_bc==bc) * c
        else:
            for bc, c in zip(barcodes, cols):
                temp = im_int*(im_bc==bc)
                im_col += temp * c
    return im_col


def taxon_legend(taxon_names, taxon_colors, label_color='k', face_color='w', taxon_counts=[],
                 text_shift_vh=(6,0.15), ft=20, dims=(6,10), lw=2, ylabel='Genus'):
    fig, ax = general_plot(ylabel=ylabel, dims=dims, col=label_color, lw=lw, ft=ft)
    vals = np.repeat(10,len(taxon_names))
    y_pos = np.arange(len(taxon_names))
    ax.barh(y_pos, vals, align='center', color=taxon_colors)
    if len(taxon_counts)>0:
        # Label the legend with counts
        for i, (v, c) in enumerate(zip(vals, taxon_counts)):
            ax.text(v - text_shift_vh[0], i + text_shift_vh[1] ,
                    str(c), color='k', fontsize=ft)
    # Remove the boundaries
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(face_color)
    # Add the genus names to the y axis
    ax.set_yticks(y_pos)
    ax.set_xticks([])
    ax.set_yticklabels(taxon_names)
    ax.invert_yaxis()
    return(fig, ax)

def get_line_histogram_x_vals(bin_edges):
    return 0.5*(bin_edges[1:]+bin_edges[:-1]).flatten()


def highlight_taxa(ax, hipr_bc, highlighted_barcodes, cols):
    '''
    Overlay segmented taxa on an image
    ax: (matplotlib.pyplot Axes object)
            Containins the image over which to show segmented taxa.
    hipr_bc: (numpy array of integers of shape (m x n))
            Image segmentation with objects labeled by cell barcode.
    highlighted_barcodes: (list of integers of length k)
            Which barcodes from 'hipr_bc' to overlay on 'ax'.
    colors: (list of tuples length k)
            RGB values to use. The tuple 'colors[i]' is used on
            'highlighted_barcodes[i]'.
    '''
    # Get size of axes to project the overlay
    extent = 0, hipr_bc.shape[1], hipr_bc.shape[0], 0
    for bc, col in zip(highlighted_barcodes, cols):
        # Get mask
        mask = 1*(hipr_bc == bc)
        # mask[mask > 0] = 1
        # Set the zero values as see through
        mask = mask.astype(np.float64)
        mask[mask == 0] = np.nan
        # Set up colormap such that nan values are see-through
        cmap_temp = copy(plt.cm.get_cmap('gray'))
        cmap_temp.set_bad(alpha = 0)
        # Set barcode values as the assigned color
        cmap_temp.set_over(col, 1.0)
        # Overlay the cells on the image
        ax.imshow(
                mask,
                cmap=cmap_temp,
                norm=pltcolors.Normalize(vmin=0, vmax=0.1),
                extent=extent,
                interpolation='none'
                )
    return ax


def convert_y_to_log_scale(ax, lims, ft, ylabel):
    llim, ulim = lims
    ax.set_ylim(llim,ulim)
    ticks = [llim, (llim + (ulim - llim)/2), ulim]
    yticklabels = ['{}'.format(np.round(tick, 1)) for tick in ticks]
    ax.set_yticks(ticks)
    ax.set_yticklabels(yticklabels, size = int(ft*3//4))
    ax.set_ylabel(ylabel + ' ($log_{10}$)')
    return(ax)


def adjust_ylims(ax, lims=('min','max'), values=[], log_scale=False, ft=12, ylabel='', int_ticks=False):
    v = np.concatenate(values) if len(values) > 0 else values
    llim = np.min(v) if lims[0]=='min' else float(lims[0])
    ulim = np.max(v) if lims[1]=='max' else float(lims[1])
    newlims = (llim, ulim)
    if log_scale:
        convert_y_to_log_scale(ax, lims=newlims, ft=ft, ylabel=ylabel)
    else:
        ax.set_ylim(llim, ulim)
        if int_ticks:
            ax.set_yticks(range(int(llim), int(np.ceil(ulim))))
    return ax


def hex_to_rgb(hex):
    h = hex.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))


def violin_dot_plot(ax, values, positions,
                    jit=0.1, y_jit=0, ft=12, dot_factor=0.5, transparency=0.7, col='k', bw='scott',
                    colors=[], lw=1, widths=[], scat=True, means=True, alpha=0):
    widths = widths if widths else jit*4
    for v, p in zip(values, positions):
        jitter_x = np.random.normal(scale=jit, size=len(v)) # TODO Adjust jit scale with number of spots
        x_list = np.repeat(p, len(v)) + jitter_x
        jitter_y = np.random.normal(scale=y_jit, size=len(v)) if y_jit else 0
        y_list = np.array(v) + jitter_y
        ymean = np.mean(v)
        if scat:
            ax.scatter(x_list, y_list, s=ft*dot_factor, marker='.', edgecolors='none', alpha=transparency,
                       c=col)
        if means:
            ax.hlines(ymean, p - jit*2, p + jit*2, colors='red', linewidth=lw)
    parts = ax.violinplot(values, positions=positions, widths=widths,
                          showmeans=False, showmedians=False,
                          showextrema=False, bw_method=bw)
    colors = colors if colors else [(0.1216, 0.4667, 0.7059)]*len(values)
    colors = [hex_to_rgb(c) if len(c) == 7 else c for c in colors]
    for pc, col_ in zip(parts['bodies'], colors):
        fcol = col_ + (alpha,)
        pc.set(alpha=None, facecolor=fcol, edgecolor=col_, linewidth=lw)
    ax.set_xticklabels([])
    ax.tick_params(labelbottom=False, bottom=False, direction='in', labelsize=ft,
                   color=col, labelcolor=col, length=lw*2)
    return ax


def check_dir(fn):
    d = os.path.split(fn)[0]
    if not os.path.exists(d):
        os.makedirs(d)
        print('Made dir:', d)
    return