import re
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes


def get_pos_file_lines(xyz, lines_initial):
    lines = lines_initial + [
        b"\tNumberPositions = " + bytes(str(len(xyz)), "utf-8") + b"\r\n"
    ]
    for i, c in enumerate(xyz):
        lines += [
            b"\tBEGIN Position" + bytes(str(i + 1), "utf-8") + b" Version = 10001\r\n",
            b"\t\tX = " + bytes(str(c[0]), "utf-8") + b" \xb5m\r\n",
            b"\t\tY = " + bytes(str(c[1]), "utf-8") + b" \xb5m\r\n",
            b"\t\tZ = " + bytes(str(c[2]), "utf-8") + b" \xb5m\r\n",
            b"\tEND\r\n"
        ]
    return lines


def get_lines_initial(fn):
    with open(fn, "rb") as f:
        lines = f.readlines()
    return lines[:8]


def general_plot(xlabel="", ylabel="", ft=12, dims=(5, 3), col="k", lw=1, pad=0):
    fig, ax = plt.subplots(figsize=(dims[0], dims[1]), tight_layout={"pad": pad})
    for i in ax.spines:
        ax.spines[i].set_linewidth(lw)
    ax.spines["top"].set_color(col)
    ax.spines["bottom"].set_color(col)
    ax.spines["left"].set_color(col)
    ax.spines["right"].set_color(col)
    ax.tick_params(direction="in", labelsize=ft, color=col, labelcolor=col)
    ax.set_xlabel(xlabel, fontsize=ft, color=col)
    ax.set_ylabel(ylabel, fontsize=ft, color=col)
    ax.patch.set_alpha(0)
    return (fig, ax)


# define plane from 3 coordinates with z
def get_xyz_convexhull(support_points, convex_hull_coords):
    # calculate normal vector
    sp = [np.array(s) for s in support_points]
    if len(sp) < 3:
        raise ValueError("Not enough support points, currently:" + str(len(sp)))
    v1 = sp[1] - sp[0]
    v2 = sp[2] - sp[0]
    nv = np.cross(v1, v2, axisa=-1, axisb=-1, axisc=-1, axis=None)

    # return z for given x,y
    coords = []
    for X, Y in convex_hull_coords:
        Z = (
            nv[0] * sp[0][0]
            + nv[1] * sp[0][1]
            + nv[2] * sp[0][2]
            - nv[0] * X
            - nv[1] * Y
        ) / nv[2]
        c = [X, Y, Z]
        c = [round(c_, 3) for c_ in c]
        coords.append(c)

    return coords


def parse_relative_pos_list_xyz(fn):
    with open(fn, "rb") as f:
        lines = f.readlines()
    coordinates = []
    i = 0
    c = [0, 0, 0]
    for l in lines:
        val = re.findall(b"(?<=[XYZ]\s\=\s)[\-0-9\.]+", l)
        if b"X =" in l:
            c[0] = float(val[0])
            i += 1
        elif b"Y =" in l:
            c[1] = float(val[0])
            i += 1
        elif b"Z =" in l:
            # print(l)
            c[2] = float(val[0])
            i += 1
        if i == 3:
            coordinates.append(c)
            i = 0
            c = [0, 0, 0]
    return coordinates[1:]


def parse_convex_hull_relative_pos_list_xy(fn):
    with open(fn, "rb") as f:
        lines = f.readlines()
    coordinates = []
    i = 0
    c = [0, 0]
    for l in lines:
        val = re.findall(b"(?<=[XY]\s\=\s)[\-0-9\.]+", l)
        if b"X =" in l:
            c[0] = float(val[0])
            i += 1
        elif b"Y =" in l:
            c[1] = float(val[0])
            i += 1
        if i == 2:
            coordinates.append(c)
            i = 0
            c = [0, 0]
    return coordinates[1:]


def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham's line algorithm
    Returns the indices of the pixels on the line between (x0, y0) and (x1, y1)
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    line_indices = []

    while (x0, y0) != (x1, y1):
        line_indices.append((x0, y0))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    line_indices.append((x1, y1))  # Include the end point

    return line_indices


def line_indices_between_points(point1, point2):
    """
    Returns the indices of the pixels on the line between two points in a numpy array
    """
    x0, y0 = point1
    x1, y1 = point2
    line_indices = bresenham_line(x0, y0, x1, y1)
    return np.array(line_indices)


def get_xy_convex_hull(coordinates, tile_dist):
    # Convert coordinates to matrix indices
    c_arr = np.array(coordinates)
    mn = np.min(c_arr, axis=0)
    c_zeroed = c_arr - mn
    c_tile = np.ceil(c_zeroed / tile_dist).astype(int)

    # Get a matrix
    mx_tile = np.max(c_tile, axis=0)
    arr = np.zeros(mx_tile + 1)

    # Get lines between matrix indices and add to matrix
    ln = c_tile.shape[0]
    for i in range(ln):
        p0 = c_tile[i, :]
        i1 = i + 1 if i < ln - 1 else 0
        p1 = c_tile[i1, :]
        li = line_indices_between_points(p0, p1)
        arr[li[:, 0], li[:, 1]] = 1

    # Fill within the edges
    arr_fill = binary_fill_holes(arr)

    # Convert back to real coordinates
    ind = np.where(arr_fill)
    ind = np.hstack([ind[0][:, None], ind[1][:, None]])
    coords_tile = (ind * tile_dist) + mn

    return coords_tile


def main():
    """
    sc_get_convex_hull.py
    Author: Ben Grodner
    Last Edit: 28 Feb 2024

    Purpose:
        Generate a tilted plane for Zen imaging.

        Given a file with a set of positions defining a region in X-Y,
        a file with three positions defining a tilted plane in Z,
        and the size of each tile scan,
        generate a new file of positions for a tile scan
        over a convex hull in X-Y over a tilted plane in Z.
    """
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "-chpos",
        "--convex_hull_relpos_fn",
        dest="convex_hull_relpos_fn",
        type=str,
        help="filename for set of positions defining a region in X-Y (Files must be in Zen '.pos' format)",
    )
    parser.add_argument(
        "-sp",
        "--support_points_relpos_fn",
        dest="support_points_relpos_fn",
        type=str,
        help="Filename containing three support points defining a tilted plane (Files must be in Zen '.pos' format)",
    )
    parser.add_argument(
        "-td",
        "--tile_dist",
        dest="tile_dist",
        type=str,
        help="Distance between the center of each tile. Also defines the tile edge size (assumes square tiles)",
    )
    parser.add_argument("-od", "--output_dir", dest="output_dir", type=str, help="Output directory filepath")
    args = parser.parse_args()

    # Get convex hull points
    pos_list_convexhull = parse_convex_hull_relative_pos_list_xy(
        args.convex_hull_relpos_fn
    )
    convex_hull_coords = get_xy_convex_hull(pos_list_convexhull, float(args.tile_dist))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print('Made dir:', args.output_dir)

    # Plot 2d convex hull
    mn = np.min(np.array(convex_hull_coords), axis=0)
    mx = np.max(np.array(convex_hull_coords), axis=0)
    dims = ((mx - mn) // 500).tolist()
    ax = plt.subplot() #general_plot(dims=dims, pad=1)
    print("x", np.array(convex_hull_coords)[:, 0])
    print("y", np.array(convex_hull_coords)[:, 1])
    ax.scatter(np.array(convex_hull_coords)[:, 0], np.array(convex_hull_coords)[:, 1])
    out_fn = args.output_dir + "/plot_xy_convex_hull.png"
    plt.savefig(out_fn)

    # Save 3d coordinates
    support_points = parse_relative_pos_list_xyz(args.support_points_relpos_fn)
    xyz = get_xyz_convexhull(support_points, convex_hull_coords)
    xyz = np.array(xyz)
    out_fn = args.output_dir + "/xyz_coordinates.csv"
    np.savetxt(out_fn, xyz, delimiter=",", fmt="%1.3f")

    # Plot 3d convex hull
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="k")
    out_fn = args.output_dir + "/plot_xyz_convex_hull.png"
    plt.savefig(out_fn)

    lines_initial = get_lines_initial(args.convex_hull_relpos_fn)
    pos_file_lines = get_pos_file_lines(xyz, lines_initial)
    pos_file_lines += [b"END\r\n"]
    out_fn = args.output_dir + '/positions_convex_hull.pos'
    with open(out_fn, 'wb') as f:
        f.writelines(pos_file_lines)

    return


if __name__ == "__main__":
    main()
