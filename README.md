# Zen tilted convex hull

These scripts are for creating a tilted convex hull to be used with Zen black software.

You need:
- A set of positions in μm from Zen that define an outline in xy coordinates (.pos). for this I export positions from the "Convex Hull" tab under the Zen "Tile Scan" tab.
- A set of positions (>=3) in μm from Zen that are used as support points to perform a regression and generate a tilted plane (.pos). For this I export positions from the Zen "Positions" tab. 
- The edge size of each FOV in μm. This should be in the metadata if you take a "Snap".

Required packages:
- aicspylibczi
- numpy
- pandas
- PIL
- py-xml
- matplotlib
- scipy

## Usage: 

### Tile scan

```
python sc_get_convex_hull_v02.py -ofn [outline filename] -spf [support points filename] -d [edge size] -o [output directory]
```

Inspect the "convex_hull_xyz.csv", "plot_convex_hull_xy.png", and "plot_convex_hull_xyz.png" files to make sure it all looks correct.

Now import the "positions_convex_hull.pos" to the Zen "Positions" tab. Make sure "Tile scan" is unchecked and "Positions" is checked, then run the experiment. 

The .czi image file output contains all the tiles as "Scenes", so if you load the CZI, the shape will be something like "XYS".

### Stitching

In order to stitch the tiles, I suggest using the [FIJI stitching plugin](https://imagej.net/plugins/image-stitching).

To convert the convex hull xy coordinates from μm to pixels for the plugin, walk through the "nb_stitch_prep.ipynb" notebook using Jupyter. 

Now open FIJI and select the Grid/Collection stitching under Plugins > Stitching. "Type" is "Positions from file". Scenes to be stitched are in "scenes_tif" directory from the notebook output. Positions file is "scenes_tif/stitch_coords_pix.txt". 

The output of the FIJI stitching program is the stitched image and a new stitched positions file. 
