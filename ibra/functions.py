# -*- coding: utf-8 -*-
"""
Miscellaneous functions for plotting, logging, data output, fitting etc
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must be set before any other matplotlib import
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import logging
from timeit import default_timer as timer
import h5py
from tifffile import TiffWriter
import os
import cv2
from scipy.optimize import curve_fit
from sklearn import linear_model
from scipy import ndimage
from loess import loess_1d

rcParams['font.family'] = 'serif'

def _render_anim_frame(args):
    """Render one animation frame to a numpy RGB array.

    Args:
        args: tuple of (i, val, frame_num, im_orig, im_back, im_framef_row,
                        labels_col, propf_col, mask_col, X, Y, X1, Y1, zmax, elev1, azim1, elev4, azim4)
    Returns:
        (i, rgb_array) where rgb_array is uint8 (H, W, 3)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    (i, val, frame_num, im_orig, im_back, im_framef_row,
     labels_col, propf_col, mask_col, X, Y, X1, Y1, zmax, elev1, azim1, elev4, azim4) = args

    fig = plt.figure(figsize=(20, 10), dpi=72)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    ax1.view_init(elev=elev1, azim=azim1)
    ax2.view_init(elev=elev1, azim=azim1)
    ax3.view_init(elev=elev1, azim=azim1)
    ax4.view_init(elev=elev4, azim=azim4)

    # Panel 1 — original frame
    ax1.plot_surface(X1, Y1, im_orig, cmap=cm.bwr, linewidth=0, antialiased=False)
    ax1.set_title("{} Frame: {}".format(val.capitalize(), frame_num + 1))
    ax1.set_zlim(0, zmax)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.grid(False)

    # Panel 2 — background surface
    minmax = np.ptp(np.ravel(im_back))
    ax2.plot_surface(X, Y, im_back, cmap=cm.bwr, linewidth=0, antialiased=False)
    ax2.set_title("Min to Max (Background): {}".format(minmax))
    ax2.set_zlim(0, zmax)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.grid(False)

    # Panel 3 — background subtracted frame
    ax3.plot_surface(X1, Y1, im_framef_row, cmap=cm.bwr, linewidth=0, antialiased=False)
    ax3.set_title("Background Subtracted Image")
    ax3.set_zlim(0, zmax)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.grid(False)

    # Panel 4 — DBSCAN tile signal percentage
    ax4.clear()
    signal = labels_col.copy()
    signal[signal > 0] = 0
    psignal = -np.float32(np.sum(signal)) / np.float32(labels_col.size)
    ax4.set_title("Percentage of Tiles with Signal: %0.2f" % psignal)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_zlim(0, 1)
    ax4.set_xticks([0, 0.5, 1])
    ax4.set_yticks([0, 0.5, 1])
    ax4.set_zticks([0, 0.5, 1])
    ax4.grid(False)
    ax4.set_xlabel('Variance', labelpad=-1)
    ax4.set_ylabel('Skewness', labelpad=-1)
    ax4.set_zlabel('Median', labelpad=-1)
    ax4.tick_params(axis="x", direction="out", pad=-2)
    ax4.tick_params(axis="y", direction="out", pad=-2)
    ax4.tick_params(axis="z", direction="out", pad=-2)
    xyz = propf_col[mask_col]
    xyz2 = propf_col[~mask_col]
    ax4.scatter(xyz2[:, 0], xyz2[:, 1], xyz2[:, 3], c='red')
    ax4.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 3], c='blue', s=40)

    # Render to numpy array (buffer_rgba replaces tostring_rgb removed in matplotlib 3.8)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgb = rgba[:, :, :3]

    plt.close(fig)
    return (i, rgb)


# Create animation of background subtraction
def background_animation(verbose, stack, work_out_path, frange):
    """Render background subtraction animation and write MP4 (H.264) via imageio + ffmpeg."""

    # Start time
    time_start = timer()

    # Pre-compute values shared across all frames
    X1, Y1 = np.int16(np.meshgrid(np.arange(stack.siz2), np.arange(stack.siz1)))
    zmax = float(np.amax(stack.im_origf))

    # Render frames serially — matplotlib 3D rendering holds the GIL throughout,
    # so ThreadPoolExecutor/ProcessPoolExecutor both run slower than serial.
    frames_rgb = []
    for i in range(len(frange)):
        args = (
            i, stack.val, frange[i],
            stack.im_origf[:, :, i],
            stack.im_backf[:, :, i],
            stack.im_framef[i, :, :],
            stack.labelsf[:, i],
            stack.propf[:, :, i],
            stack.maskf[:, i],
            stack.X, stack.Y,
            X1, Y1, zmax,
            15., 30., 30., 230.,
        )
        frames_rgb.append(_render_anim_frame(args)[1])

    # Determine output filename
    if max(np.ediff1d(frange, to_begin=frange[0])) > 1:
        fname_base = work_out_path + '_' + stack.val + '_specific'
        num = 1
        while os.path.isfile(fname_base + str(num) + '.mp4'):
            num += 1
        out_path = fname_base + str(num) + '.mp4'
    else:
        out_path = (work_out_path + '_' + stack.val + '_frames'
                    + str(frange[0] + 1) + '_' + str(frange[-1] + 1) + '.mp4')

    # Write MP4 (H.264) via imageio + ffmpeg
    import imageio
    writer = imageio.get_writer(out_path, fps=2, codec='libx264',
                                pixelformat='yuv420p', macro_block_size=None,
                                output_params=['-crf', '23'])
    for frame_rgb in frames_rgb:
        writer.append_data(frame_rgb)
    writer.close()

    # End time
    time_end = timer()
    time_elapsed = str(int(time_end - time_start) + 1)
    if verbose:
        print((stack.val.capitalize() + " (Background Animation) Time: " + time_elapsed + " second(s)"))


def logit(path):
    """Logging data"""
    logger = logging.getLogger('back')
    hdlr = logging.FileHandler(path+'.log')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(20)

    return logger


# Maximum frame count before the animation warning fires in parameter_extraction.
# Exported so parameter_extraction can import it without duplicating the value.
ANIM_FRAME_WARN = 50


def h5(data, val, path, frange):
    """Saving the image stack as a .h5 file.

    Uses gzip compression. lzf was tested but proved slower on the sparse
    uint16 image arrays produced by background subtraction — gzip finds more
    redundancy in this data. Compression level left at the h5py default (4).
    """
    with h5py.File(path, 'a') as f:

        if val in f:
            # Determine the frange attribute key for this dataset
            if val in ('acceptor', 'acceptori', 'acceptorb'):
                frange_attr = 'acceptor_frange'
            elif val in ('donor', 'donori', 'donorb'):
                frange_attr = 'donor_frange'
            else:
                frange_attr = 'ratio_frange'

            # Read stored frange — handle three cases:
            #   1. new format: stored as a dataset (normal path going forward)
            #   2. old format: stored as an attribute (files written before this fix)
            #   3. missing: frange was never written due to a previous crash
            # Cases 2 and 3 fall back to treating the existing data as a full
            # replacement so the merge path is skipped and the file is rewritten cleanly.
            if frange_attr in f:
                orange = f[frange_attr][:]
            elif frange_attr in f.attrs:
                orange = f.attrs[frange_attr]
            else:
                orange = frange  # crash remnant — force full replacement

            # Fast path: if the incoming frange is identical to the stored frange
            # this is a full replacement — skip the merge entirely, just delete
            # and rewrite. This avoids reading, deserialising and re-compressing
            # the entire existing dataset, which dominates save time on re-runs.
            # Partial writes (individual frames or sub-ranges) still go through
            # the merge path so per-frame tuning is preserved.
            if np.array_equal(np.sort(frange), np.sort(orange)):
                del f[val]
                res = np.array(data)
                res_range = frange
            else:
                # Partial update — merge new frames into existing dataset
                orig = f[val]
                orig_dict = dict(zip(orange, orig))
                new_dict = dict(zip(frange, data))

                for key in frange:
                    orig_dict[key] = new_dict[key]

                orig_dict_sorted = sorted(orig_dict.items())
                res_range, res = list(zip(*orig_dict_sorted))
                res = np.array(res)

                del f[val]

        else:
            # Dataset does not exist yet — create fresh
            res = np.array(data)
            res_range = frange

        # Save the image pixel data and frange.
        # frange is stored as a dataset (not an attribute) because large stacks
        # (e.g. 9000 frames) produce arrays that overflow the HDF5 object header
        # attribute size limit (~64 KB), causing a write crash.
        if val in ('acceptor', 'donor'):
            f.create_dataset(val, data=res, shape=res.shape, dtype=np.uint16, compression='gzip')
            frange_key = val + '_frange'
            if frange_key in f:
                del f[frange_key]
            f.create_dataset(frange_key, data=np.array(res_range), dtype=np.int64)
        elif val == 'ratio':
            f.create_dataset(val, data=res, shape=res.shape, dtype=np.uint8, compression='gzip')
            frange_key = val + '_frange'
            if frange_key in f:
                del f[frange_key]
            f.create_dataset(frange_key, data=np.array(res_range), dtype=np.int64)
        else:
            f.create_dataset(val, data=res, shape=res.shape, dtype=np.float16, compression='gzip')


def time_evolution(acceptor, donor, work_out_path, name, ylabel, h5_save, single_channel=False):
    """Median channel intensity per frame.

    In single_channel mode, plots a single blue line with no legend.
    In two-channel mode, plots acceptor (purple) and donor (orange) with legend.
    """
    acceptor_plot = sorted(acceptor.items())
    xa, ya = list(zip(*acceptor_plot))
    xplot = [x + 1 for x in xa]

    vals = ['acceptori','donori','acceptornz','donornz']
    if (ylabel == 'Median Intensity/Bit Depth'):
        names = vals[:2]
        dec = 1
    elif (ylabel == 'Foreground/Total Image Pixels'):
        names = vals[2:]
        dec = 2

    if h5_save and not single_channel:
        # Only save to HDF5 in two-channel mode — single-channel has no _ratio_back.h5
        donor_plot = sorted(donor.items())
        _, yd = list(zip(*donor_plot))
        ya_arr = np.array(ya)
        yd_arr = np.array(yd)

        with h5py.File(work_out_path+'_ratio_back.h5', 'a') as f:
            if (names[0] in f):
                del f[names[0]]
            f.create_dataset(names[0], data=ya_arr, shape=ya_arr.shape, dtype=np.uint16, compression='gzip')

            if (names[1] in f):
                del f[names[1]]
            f.create_dataset(names[1], data=yd_arr, shape=yd_arr.shape, dtype=np.uint16, compression='gzip')

    # Set up plot
    fig, ax = plt.subplots(figsize=(12, 8))

    if single_channel:
        # Single line in acceptor blue, no legend
        ax.plot(xplot, ya, c=(0.62745098, 0.152941176, 0.498039216), marker='*')
    else:
        donor_plot = sorted(donor.items())
        _, yd = list(zip(*donor_plot))
        ax.plot(xplot, ya, c=(0.62745098, 0.152941176, 0.498039216), marker='*')
        ax.plot(xplot, yd, c=(1, 0.517647059, 0), marker='*')
        plt.legend(['Acceptor', 'Donor'], fancybox=None, fontsize=18)

    plt.ylabel(ylabel, labelpad=15, fontsize=22)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=dec))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    plt.yticks(fontsize=18)
    plt.xlabel('Frame Number', labelpad=15, fontsize=22)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(fontsize=18)

    plt.savefig(work_out_path + name, bbox_inches='tight')
    plt.close(fig)


def detect_freak_frames(channeli_dict, k=5, window=51):
    """Detect freak frames by comparing each value to its local rolling median.

    Uses median absolute deviation (MAD) of the residuals from a rolling median
    so that slow bleaching, oscillations, and gradual trends are ignored.
    Only sharp local outliers — single frames or small clusters that deviate
    strongly from their immediate neighbourhood — are flagged.

    Args:
        channeli_dict: dict of {frame_index: median_intensity_value}
        k:             detection threshold in MAD units (default 5)
        window:        rolling median half-window in frames (default 51, i.e. ±25 frames)
                       must be odd; will be forced odd if even value is passed
    Returns:
        List of dicts, one per flagged frame:
            frame_number   (1-based, matching display convention)
            frame_index    (0-based, raw key from channeli_dict)
            value          (measured intensity value)
            local_median   (rolling median at that frame)
            deviation_mad  (residual expressed in MAD units)
        Empty list if no frames are flagged.
    """
    if window % 2 == 0:
        window += 1
    half = window // 2

    # Sort by frame index so the rolling window is meaningful
    sorted_items = sorted(channeli_dict.items())
    indices = np.array([x[0] for x in sorted_items])
    values  = np.array([x[1] for x in sorted_items], dtype=np.float64)
    n = len(values)

    # Rolling median — reflect at boundaries to avoid edge bias
    rolling_med = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        rolling_med[i] = np.median(values[lo:hi])

    residuals = values - rolling_med

    # Global MAD of residuals (robust scale estimator)
    mad = np.median(np.abs(residuals - np.median(residuals)))
    if mad == 0:
        # Fallback: use std-based scale if MAD is zero (flat signal)
        mad = np.std(residuals) * 0.6745  # 0.6745 makes it consistent with Gaussian MAD
    if mad == 0:
        return []  # Completely flat signal — nothing to flag

    deviation_mad = residuals / mad

    flagged = []
    for i in range(n):
        if np.abs(deviation_mad[i]) > k:
            flagged.append({
                'frame_number':  int(indices[i]) + 1,
                'frame_index':   int(indices[i]),
                'value':         float(values[i]),
                'local_median':  float(rolling_med[i]),
                'deviation_mad': float(deviation_mad[i]),
            })

    return flagged


def block(data, size):
    """Reshape image stack for faster processing"""
    return (data.reshape(data.shape[0] // size, size, -1, size)
            .swapaxes(1, 2)
            .reshape(-1, size, size))


def tiff(data, path):
    """Write out a TIFF stack"""
    with TiffWriter(path, bigtiff=True) as tif:
        for i in range(data.shape[0]):
            tif.write(data[i,:,:])


def bleach_fit(brange, crange, channeli_dict, fitter):
    """Fit bleach decay curve and return correction multiplier array.

    Args:
        brange:        frame indices used for fitting
        crange:        frame indices to correct (brange[0] to end)
        channeli_dict: dict of {frame_index: median_intensity}
        fitter:        'linear', 'exponential', or 'loess'
    Returns:
        1D array of correction multipliers, length = len(crange)
    """
    # Extract intensity values over the fit range
    x = brange.astype(np.float64)
    y = np.array([channeli_dict[i] for i in brange], dtype=np.float64)
    x_corr = crange.astype(np.float64)

    if fitter == 'linear':
        reg = linear_model.LinearRegression()
        reg.fit(x.reshape(-1, 1), y)
        y_fit = reg.predict(x_corr.reshape(-1, 1))

    elif fitter == 'exponential':
        def exp_func(x, a, b, c):
            return a * np.exp(-b * x) + c
        try:
            popt, _ = curve_fit(exp_func, x, y, p0=[y[0], 0.001, y[-1]], maxfev=5000)
            y_fit = exp_func(x_corr, *popt)
        except RuntimeError:
            reg = linear_model.LinearRegression()
            reg.fit(x.reshape(-1, 1), y)
            y_fit = reg.predict(x_corr.reshape(-1, 1))

    elif fitter == 'loess':
        _, y_fit, _ = loess_1d.loess_1d(x, y, xnew=x_corr, frac=0.5)

    # Return correction multiplier: y[0] / fitted_value
    # Avoids division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(y_fit != 0, y[0] / y_fit, 1.0)
    return corr


def ratio_calc(acceptor, donor):
    """Calculate 8-bit ratio stack from acceptor and donor arrays"""
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_raw = np.where(donor > 0, acceptor.astype(np.float32) / donor.astype(np.float32), 0)

    ratio_max = np.amax(ratio_raw)
    if ratio_max > 0:
        ratio = np.uint8(ratio_raw / ratio_max * 255)
    else:
        ratio = np.zeros_like(ratio_raw, dtype=np.uint8)

    return ratio, ratio_raw
