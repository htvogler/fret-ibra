# -*- coding: utf-8 -*-
"""
Background subtraction with the DBSCAN clustering algorithm
using the higher moments of the intensity distribution per tile
"""

from __future__ import print_function, division

import numpy as np
import scipy as sp
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
import cv2
import math
import pims
import os
from functions import background_animation, logit, h5, block, tiff, time_evolution
from timeit import default_timer as timer
from skimage.external.tifffile import TiffWriter
import concurrent.futures
import multiprocessing

# #############################################################################

# Create image stack class
class stack():

    def __init__(self, work_inp_path, val, ext):
        self.val = val

        # Import stack
        im_path = work_inp_path + '_' + self.val + '.' + ext
        self.im_stack = pims.open(im_path)
        self.siz1, self.siz2 = self.im_stack.frame_shape

    # Set frame parameters as instance variables
    def set_frame_parameters(self, win):
        # Test to find suggested values of nwindow
        if (self.siz1 <= 1400 or self.siz2 <= 1400):
            win_test = range(20, 37, 4)
        else:
            win_test = range(24, 41, 4)

        win_res = [0] * 4
        for winn, winc in enumerate(win_test):
            if (self.siz1 % winc == 0 and self.siz2 % winc == 0):
                win_res[winn] = 1

        sug = win_test[win_res.index(1)]

        # Check nwindows parameter
        ass_str1 = f"nwindows must be a factor of the X and Y image resolution (suggested value: {sug})"
        ass_str2 = f"nwindows should be increased (suggested value: {sug})"
        assert (self.siz1 % win == 0), ass_str1
        assert (self.siz2 % win == 0), ass_str1
        assert (self.siz1 / win <= 80.0), ass_str2
        assert (self.siz2 / win <= 80.0), ass_str2

        # Find frame size and set window size
        self.dim = np.int16(self.siz2 / win)
        self.height = np.int16(win)
        self.width = np.int16(self.siz1 / self.dim)

        # Create underlying background mesh
        self.X, self.Y = np.int16(np.meshgrid(np.arange(self.height), np.arange(self.width)))
        self.XY = np.column_stack((np.ravel(self.X), np.ravel(self.Y)))

        # Setup grid for intensity weighted centroid calculation
        grid = np.indices((self.dim, self.dim))
        offset = (self.dim - 1) * 0.5
        self.dist_grid = np.sqrt(np.square(np.subtract(grid[0], offset)) + np.square(np.subtract(grid[1], offset)))

    # Set processing constants as instance variables
    # Note: logger is intentionally excluded — it is not picklable and must not be
    # passed to worker processes. Logging is handled by the main process after
    # results are collected.
    def set_class_constants(self, verbose, res, logger, frange, eps):
        self.verbose = verbose
        self.res = res
        self.logger = logger
        self.frange = frange
        self.eps = eps

    # Preallocate arrays for speed
    def metric_prealloc(self):
        length = len(self.frange)
        rows = self.height * self.width
        self.im_origf = np.empty((self.siz1, self.siz2, length), dtype=np.uint16)
        self.propf = np.empty((rows, 5, length), dtype=np.float32)
        self.maskf = np.empty((rows, length), dtype=np.bool)
        self.labelsf = np.empty((rows, length), dtype=np.int8)
        self.im_backf = np.empty((self.width, self.height, length), dtype=np.int16)
        self.im_framef = np.empty((length, self.siz1, self.siz2), dtype=np.uint16)

    # Update metrics on a per frame basis
    def metric_update(self, result):
        pos = result[0]
        self.im_origf[:, :, pos] = result[1]
        self.im_backf[:, :, pos] = result[2]
        self.im_framef[pos, :, :] = result[3]
        self.propf[:, :, pos] = result[4]
        self.maskf[:, pos] = result[5].tolist()
        self.labelsf[:, pos] = result[6]

    # Use log file to print frame metrics
    def logger_update(self, h5_save, time_elapsed):
        if (max(np.ediff1d(self.frange, to_begin=self.frange[0])) > 1):
            self.logger.info('(Background Subtraction) ' + self.val + '_eps: ' + str(self.eps) + ', frames: ' + ",".join(
                map(str, [x + 1 for x in self.frange])) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))
        else:
            self.logger.info('(Background Subtraction) ' + self.val + '_eps: ' + str(self.eps) + ', frames: ' + str(self.frange[0] + 1) + '-' + str(
                self.frange[-1] + 1) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))

    # Run background subtraction stack workflow
    def stack_workflow(self, parallel):
        # Only use parallel processing if explicitly enabled AND frame count
        # exceeds the threshold. Below the threshold, process spawn overhead
        # (bootstrapping worker processes) exceeds the parallelisation benefit.
        PARALLEL_FRAME_THRESHOLD = 60
        use_parallel = parallel and len(self.frange) > PARALLEL_FRAME_THRESHOLD

        if use_parallel:
            # Ensure spawn start method is used for cross-platform compatibility.
            # spawn is the default on Windows and macOS (Python 3.8+) and is safe
            # everywhere. fork (Linux default) can cause deadlocks with certain
            # libraries (OpenCV, pims). force=True is used because the context may
            # already be set earlier in the process.
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass

            # Build self-contained frame params to pass to workers.
            # The logger is deliberately excluded — it is not picklable.
            # Any DBSCAN errors are returned in the result tuple and logged
            # by the main process after collection.
            frame_params = _FrameParams(
                val=self.val,
                siz1=self.siz1,
                siz2=self.siz2,
                dim=self.dim,
                height=self.height,
                width=self.width,
                X=self.X,
                Y=self.Y,
                XY=self.XY,
                dist_grid=self.dist_grid,
                verbose=self.verbose,
                res=self.res,
                eps=self.eps,
            )

            # Submit one job per frame
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(_run_frame, np.asarray(self.im_stack[count]), count, pos, frame_params): pos
                    for pos, count in enumerate(self.frange)
                }

            # Collect results and log any DBSCAN errors in the main process
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result[7] is not None:
                    self.logger.error("".join((self.val, '_eps: ', str(self.eps),
                                               ', frame: ', str(result[7] + 1), " (eps value too low)")))
                self.metric_update(result)

        else:
            for pos, count in enumerate(self.frange):
                fr = frame(np.asarray(self.im_stack[count]), count, pos, self)
                result = fr.frame_workflow()
                if result[7] is not None:
                    self.logger.error("".join((self.val, '_eps: ', str(self.eps),
                                               ', frame: ', str(result[7] + 1), " (eps value too low)")))
                self.metric_update(result)


# Lightweight picklable container for the frame parameters that workers need.
# Replaces the class-variable pattern which breaks under spawn.
class _FrameParams:
    __slots__ = ('val', 'siz1', 'siz2', 'dim', 'height', 'width',
                 'X', 'Y', 'XY', 'dist_grid', 'verbose', 'res', 'eps')

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# Module-level function required for pickling under spawn.
# Methods on instances cannot be pickled by the default pickler,
# so the worker entry point must be a plain module-level function.
def _run_frame(im_frame_arr, count, pos, params):
    fr = frame(im_frame_arr, count, pos, params)
    return fr.frame_workflow()


# Create single image frame class
class frame():
    def __init__(self, im_frame, count, pos, params):
        self.im_frame = im_frame
        self.im_frame_orig = im_frame
        self.count = count
        self.pos = pos
        self.params = params

    # Calculate pixel properties per tile
    def properties(self):
        p = self.params
        tile_prop = np.empty([p.width * p.height, 5], dtype=np.float32)
        self.im_tile = block(self.im_frame, p.dim)

        for i in range(tile_prop.shape[0]):
            im_tile_flat = np.ravel(self.im_tile[i, :, :])
            tile_prop[i, 0] = sp.stats.moment(im_tile_flat, moment=2, axis=0)
            tile_prop[i, 1] = sp.stats.moment(im_tile_flat, moment=3, axis=0)
            tile_prop[i, 2] = sp.stats.moment(im_tile_flat, moment=4, axis=0)
            tile_prop[i, 3] = np.median(im_tile_flat)

            centroid_intensity = np.multiply(self.im_tile[i, :, :], p.dist_grid)
            tile_prop[i, 4] = np.sum(np.uint32(centroid_intensity))

        self.im_median = np.copy(tile_prop[:, 3])

        tile_min = np.amin(tile_prop, axis=0)
        tile_ptp = np.ptp(tile_prop, axis=0)

        for j in range(tile_prop.shape[1]):
            tile_prop[:, j] = list(map(lambda x: (x - tile_min[j]) / tile_ptp[j], tile_prop[:, j]))

        self.tile_prop = tile_prop

    # Cluster tiles into background and signal
    def clustering(self):
        p = self.params
        db = DBSCAN(eps=p.eps, min_samples=int(p.height * 1.25)).fit(self.tile_prop)
        self.core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        self.core_samples_mask[db.core_sample_indices_] = True
        self.labels = np.int8(db.labels_)

    # Subtract median background from frame intensities
    def subtraction(self):
        p = self.params
        im_median_mask = np.multiply(self.im_median, (self.labels + 1))
        pos_front = np.int16(np.where(im_median_mask == 0)[0])
        XY_back = np.delete(p.XY, pos_front, axis=0)
        im_median_mask_back = np.delete(im_median_mask, pos_front, axis=0)
        self.im_frame = np.zeros([p.siz1, p.siz2])
        self.dbscan_error = None

        try:
            self.XY_interp_back = np.uint16(griddata(XY_back, im_median_mask_back, (p.X, p.Y), method='nearest'))

            for i, j in enumerate(self.XY_interp_back.flat):
                rem = int(np.floor(i / p.height))
                mod = i % p.height
                self.im_frame[rem * p.dim:(rem + 1) * p.dim, mod * p.dim:(mod + 1) * p.dim] = np.subtract(self.im_tile[i, :, :], j)
                self.im_frame[self.im_frame > np.amax(self.im_frame_orig)] = 0
                self.im_frame[self.im_frame < 0] = 0

        except:
            # Signal DBSCAN failure back to the main process for logging
            self.XY_interp_back = np.zeros((p.width, p.height))
            self.dbscan_error = self.count

    # Apply bilateral smoothing filter to preserve edges
    def filter(self):
        p = self.params
        filtered = cv2.bilateralFilter(np.float32(self.im_frame),
                                       np.int16(math.ceil(9 * p.siz2 / 320)),
                                       p.width * 0.5, p.width * 0.5)
        self.im_frame = np.uint16(filtered)

    # Run frame background subtraction workflow
    def frame_workflow(self):
        p = self.params
        if p.verbose:
            print((p.val.capitalize() + ' (Background Subtraction) Frame Number: ' + str(self.count + 1)))
        self.properties()
        self.clustering()
        self.subtraction()
        self.filter()

        # Result tuple: pos, im_orig, XY_interp_back, im_frame, tile_prop,
        #               core_samples_mask, labels, dbscan_error (None if ok)
        return (self.pos, self.im_frame_orig, self.XY_interp_back, self.im_frame,
                self.tile_prop, self.core_samples_mask, self.labels, self.dbscan_error)


def _compute_channeli(im_frame, res_local):
    """Compute Otsu-masked median intensity and foreground pixel fraction for a single frame.

    Uses np.median on the masked foreground pixels directly — equivalent to
    scipy.ndimage.median with a binary label mask but orders of magnitude faster
    since it avoids the label-region iteration overhead.

    Returns:
        (channeli, nz) — masked median intensity as % of bit depth, and
                         foreground pixel count as % of total pixels.
    """
    import cv2
    mult = np.float32(255) / np.float32(res_local)
    ires = 100 / np.float32(res_local)
    ipix = 100 / float(im_frame.size)
    frame_scaled = np.uint8(np.float32(im_frame) * mult)
    if np.amax(frame_scaled) > 3:
        _, thresh = cv2.threshold(frame_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(frame_scaled, 3, 255, cv2.THRESH_BINARY)
    nz = np.count_nonzero(thresh) * ipix
    foreground = im_frame[thresh > 0]
    channeli = float(np.median(foreground) * ires) if len(foreground) > 0 else 0.0
    return (channeli, nz)


def background(verbose, logger, work_inp_path, work_out_path, ext, res, module, eps, win,
               parallel, anim_save, h5_save, tiff_save, frange, single_channel=False):
    # Determine channel label from module and single_channel flag
    # In single-channel mode, always label output 'acceptor' so downstream tools find it under the standard key
    if single_channel or module == 0:
        val = 'acceptor'
    else:
        val = 'donor'

    # Start time
    time_start = timer()

    # Create stack class from input TIFF file
    all = stack(work_inp_path, val, ext)

    # Frame number check
    assert (max(frange) < len(all.im_stack)), "frame numbers not found in input TIFF stack"

    # Assign frame parameters
    all.set_frame_parameters(win)

    # Assign class constants
    all.set_class_constants(verbose, res, logger, frange, eps)

    # Preallocation of tile metrics
    all.metric_prealloc()

    # Run image processing workflow
    all.stack_workflow(parallel)

    # End time
    time_end = timer()
    time_elapsed = str(int(time_end - time_start) + 1)
    if verbose:
        print((val.capitalize() + " (Background Subtraction) Time: " + time_elapsed + " second(s)"))

    # Update log file with background subtraction data
    all.logger_update(h5_save, time_elapsed)

    # Save animation of frame metrics
    if anim_save:
        background_animation(verbose, all, work_out_path, frange)

    # Save background subtracted stack as HDF5
    if h5_save:
        h5_time_start = timer()
        h5(all.im_framef, val, work_out_path + '_back.h5', frange=frange)

    # Compute per-frame foreground masked median intensity and pixel count.
    # Always computed — needed for quality PNGs and (when h5_save) for HDF5 output.
    res_local = all.res
    frames_list = [all.im_framef[i, :, :] for i in range(all.im_framef.shape[0])]

    # channeli computation is fast (numpy median on boolean-indexed array)
    # so spawn overhead from ProcessPoolExecutor would dominate — always serial
    channeli_results = [_compute_channeli(f, res_local) for f in frames_list]

    channeli = np.array([r[0] for r in channeli_results], dtype=np.float16)
    channelnz = np.array([r[1] for r in channeli_results], dtype=np.float32)

    if h5_save:
        h5(channeli, val + 'i', work_out_path + '_back.h5', frange=frange)

    # Quality assessment PNGs — produced after every module 0 or 1 run.
    # Show per-frame bg-subtracted intensity and foreground pixel count.
    # Useful for identifying freak frames and estimating bleach curves
    # before running module 4. In two-channel mode each module run produces
    # its own per-channel PNG (acceptor_ or donor_ prefixed).
    channeli_dict = dict(zip(frange, channeli.astype(float)))
    channelnz_dict = dict(zip(frange, channelnz.astype(float)))
    time_evolution(channeli_dict, channeli_dict,
                   work_out_path, '_' + val + '_intensity_nonbleach.png',
                   'Median Intensity/Bit Depth', h5_save=False, single_channel=True)
    time_evolution(channelnz_dict, channelnz_dict,
                   work_out_path, '_' + val + '_pixelcount.png',
                   'Foreground/Total Image Pixels', h5_save=False, single_channel=True)
    if verbose:
        print("Saving quality assessment PNGs for " + val + " channel")

    if h5_save:
        h5_time_end = timer()
        if verbose:
            print(("Saving " + val.capitalize() + " HDF5 stack in " + work_out_path + '.h5' + ' [Time: ' + str(int(h5_time_end - h5_time_start) + 1) + " second(s)]"))

    # Save background-subtracted acceptor/donor images as TIFF
    if tiff_save:
        tiff_time_start = timer()
        tiff(all.im_framef, work_out_path + '_' + val + '_back.tif')
        tiff_time_end = timer()

        if verbose:
            print(("Saving " + val.capitalize() + " TIFF stack in " + work_out_path + '_back_' + val + '.tif' + ' [Time: ' + str(int(tiff_time_end - tiff_time_start) + 1) + " second(s)]"))
