# -*- coding: utf-8 -*-
"""
Background subtraction with the DBSCAN clustering algorithm
using the higher moments of the intensity distribution per tile
"""

from __future__ import print_function, division

import numpy as np
import scipy as sp
from scipy import ndimage
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
import cv2
import math
import pims
import os
import csv
from functions import background_animation, logit, h5, block, tiff, time_evolution, detect_freak_frames
from timeit import default_timer as timer
from tifffile import TiffWriter
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
        # Check nwindows parameter
        # dim = image_width / nwindow (tile size in pixels). Both axes must be
        # divisible by dim so the block() reshape works. siz2 % win == 0 guarantees
        # dim is an integer; siz1 % dim == 0 is the actual reshape requirement
        # (NOT siz1 % win == 0, which is a stricter and incorrect check).
        dim_candidate = self.siz2 // win if win > 0 else 0

        def _suggest():
            win_test = range(20, 37, 4) if (self.siz1 <= 1400 or self.siz2 <= 1400) else range(24, 41, 4)
            for winc in win_test:
                if self.siz2 % winc == 0 and self.siz1 % (self.siz2 // winc) == 0:
                    return winc
            return None

        assert (self.siz2 % win == 0), (
            f"image width ({self.siz2}px) must be divisible by nwindow={win}"
            + (f" (suggested value: {_suggest()})" if _suggest() else ""))
        assert (self.siz1 % dim_candidate == 0), (
            f"image height ({self.siz1}px) must be divisible by tile size "
            f"{dim_candidate}px (= width {self.siz2} / nwindow {win}). "
            f"Crop height to a multiple of {dim_candidate}px.")
        assert (self.siz1 / win <= 80.0), (
            f"nwindows should be increased"
            + (f" (suggested value: {_suggest()})" if _suggest() else ""))
        assert (self.siz2 / win <= 80.0), (
            f"nwindows should be increased"
            + (f" (suggested value: {_suggest()})" if _suggest() else ""))

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
    def set_class_constants(self, verbose, res, logger, frange, eps, declutter_radius=0,
                             background_method='dbscan', tophat_size=0):
        self.verbose = verbose
        self.res = res
        self.logger = logger
        self.frange = frange
        self.eps = eps
        self.declutter_radius = declutter_radius
        self.background_method = background_method
        self.tophat_size = tophat_size

    # Preallocate arrays for speed
    def metric_prealloc(self):
        length = len(self.frange)
        rows = self.height * self.width
        self.im_origf = np.empty((self.siz1, self.siz2, length), dtype=np.uint16)
        self.propf = np.empty((rows, 5, length), dtype=np.float32)
        self.maskf = np.empty((rows, length), dtype=bool)
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
        declutter_note = ', declutter_radius: ' + str(self.declutter_radius)
        if self.background_method != 'dbscan':
            declutter_note += ', background_method: ' + self.background_method + ', tophat_size: ' + str(self.tophat_size)
        if (max(np.ediff1d(self.frange, to_begin=self.frange[0])) > 1):
            self.logger.info('(Background Subtraction) ' + self.val + '_eps: ' + str(self.eps) + declutter_note + ', frames: ' + ",".join(
                map(str, [x + 1 for x in self.frange])) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))
        else:
            self.logger.info('(Background Subtraction) ' + self.val + '_eps: ' + str(self.eps) + declutter_note + ', frames: ' + str(self.frange[0] + 1) + '-' + str(
                self.frange[-1] + 1) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))

    # Run background subtraction stack workflow
    def stack_workflow(self, parallel):
        # Only use parallel processing if explicitly enabled AND frame count
        # exceeds the threshold. Below the threshold, process spawn overhead
        # (bootstrapping worker processes) exceeds the parallelisation benefit.
        PARALLEL_FRAME_THRESHOLD = 60
        # background_method='tophat' is cheap enough (~4ms/frame measured, vs
        # ~850ms/frame for dbscan's per-tile moment loop + clustering) that the
        # fixed per-frame IPC cost of dispatching to a worker process (pickling
        # the frame array across process boundaries) exceeds its own compute
        # cost at any frame count, not just below PARALLEL_FRAME_THRESHOLD --
        # confirmed on real data: a 200-frame run took 37s with parallel=1 vs
        # 5s with parallel=0, the opposite of what parallel=1 is meant to buy.
        # Always run tophat serially regardless of the parallel cfg setting.
        use_parallel = parallel and len(self.frange) > PARALLEL_FRAME_THRESHOLD and self.background_method != 'tophat'

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
                declutter_radius=self.declutter_radius,
                background_method=self.background_method,
                tophat_size=self.tophat_size,
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
                 'X', 'Y', 'XY', 'dist_grid', 'verbose', 'res', 'eps',
                 'declutter_radius', 'background_method', 'tophat_size')

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

    # Suppress thin, static, high-contrast artifacts (e.g. reflective/fluorescent
    # device edges — microchannel walls, chip fiducials) before tile statistics
    # are computed. These share the same per-tile variance/skew/median signature
    # as real signal at the tile resolution `properties()`/`clustering()` operate
    # on, so DBSCAN cannot separate them no matter how eps is tuned — but they are
    # reliably much narrower than genuine signal (a cell, a growing tube). Grayscale
    # morphological opening with a disk narrower than the real signal but wider
    # than the artifact removes anything that can't contain the disk anywhere along
    # its length, at native pixel resolution, with no dependence on how long a given
    # pixel has carried real signal (unlike a temporal/per-pixel-history approach,
    # which misclassifies a permanently-occupied pixel as static background).
    # No-op when declutter_radius is 0 (default) — off unless explicitly requested.
    def declutter(self):
        p = self.params
        r = p.declutter_radius
        if not r:
            return
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        footprint = (x ** 2 + y ** 2) <= r ** 2
        self.im_frame = ndimage.grey_opening(self.im_frame, footprint=footprint)

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

    # Alternative to properties()+clustering()+subtraction(): estimate the
    # background via a single large-radius grayscale morphological opening
    # instead of per-tile DBSCAN classification. A structuring element wider
    # than any real signal in the frame can never "fit inside" that signal, so
    # the opening always reaches past it into genuinely surrounding pixels --
    # the background estimate at any point is built entirely from real
    # background, structurally, regardless of how bright or dim the real signal
    # there happens to be. This makes it immune to two failure modes the tile/
    # DBSCAN method has for a thin, persistent structure (e.g. a growing tube):
    # (1) a tile partially covered by real signal getting its own inflated
    # median used as background, corrupting the whole tile block; (2) DBSCAN
    # correctly recognising real signal as statistically distinct from
    # background, but the *interpolated* background value substituted for it
    # still nearly matching the signal's own brightness once that margin has
    # eroded (e.g. from photobleaching over a long timelapse) -- neither
    # failure depends on tile size or eps, both were confirmed on real data
    # (HV202_1_14, see the fret-ibra session notes), and this method sidesteps
    # both by never needing to classify anything in the first place.
    #
    # Only valid for real signal narrower than tophat_size -- NOT a general
    # replacement for the DBSCAN method, which is why this is opt-in
    # (background_method='tophat') rather than the default. A real foreground
    # region *wider* than tophat_size (e.g. a root cross-section, bulk
    # cytoplasmic signal) would get treated as background and erased -- the
    # DBSCAN method has no such width assumption and remains the right default
    # for those sample types.
    def tophat_subtraction(self):
        p = self.params
        bg = ndimage.grey_opening(self.im_frame, size=p.tophat_size)
        tophat = self.im_frame.astype(np.float32) - bg.astype(np.float32)
        tophat[tophat < 0] = 0
        self.im_frame = tophat
        self.dbscan_error = None

        # Downsample the actual background estimate to the tile grid, purely
        # for the existing animation's "background surface" panel -- real
        # numbers, unlike the placeholders below.
        bg_tiles = block(bg, p.dim)
        self.XY_interp_back = np.uint16(
            bg_tiles.reshape(bg_tiles.shape[0], -1).mean(axis=1)
        ).reshape(p.width, p.height)

        # Placeholders only: DBSCAN diagnostics (per-tile stats/labels/core
        # mask) don't apply to this method, but metric_update/
        # background_animation still expect arrays of these shapes.
        n_tiles = p.width * p.height
        self.tile_prop = np.zeros((n_tiles, 5), dtype=np.float32)
        self.core_samples_mask = np.zeros(n_tiles, dtype=bool)
        self.labels = np.zeros(n_tiles, dtype=np.int8)

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
        self.declutter()
        if p.background_method == 'tophat':
            self.tophat_subtraction()
        else:
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


# Number of frames sampled across the requested range to auto-estimate a
# declutter radius (see _estimate_declutter_radius). Evenly spaced, not the
# full stack — cheap enough to run on every declutter_auto request even for
# multi-thousand-frame stacks, and 30 frames already gives thousands of pooled
# skeleton-width readings in practice, far more than needed for a stable Otsu split.
DECLUTTER_AUTO_SAMPLE_FRAMES = 30

# Minimum pooled width samples required before trusting an Otsu split at all —
# below this there isn't enough data to distinguish a real bimodal population
# from noise.
DECLUTTER_AUTO_MIN_SAMPLES = 200

# Required ratio between the wide cluster's low end and the narrow cluster's
# high end for the two to count as "clearly separated". Below this ratio the
# two populations aren't confidently distinct enough to place a radius between
# them safely.
DECLUTTER_AUTO_MIN_SEPARATION_RATIO = 1.5


def _estimate_declutter_radius(im_stack, sample_frame_indices):
    """Estimate a declutter radius from a sample of raw frames, with no
    per-frame human judgement call about which frame is "representative".

    Approach: in each sampled frame, isolate bright ridge-like structures from
    the smooth illumination background via a white top-hat filter, then record
    a local-width reading (2x the Euclidean distance transform) at every
    skeleton pixel of the resulting mask. Pool these readings across all
    sampled frames with NO attempt to label any specific connected component
    as "the real signal" — an earlier version of this did exactly that (take
    the single largest connected component as the tube) and it broke on a real
    frame where the tube itself was split into two similarly-sized mask pieces
    by a local dip in brightness, a real and unavoidable segmentation ambiguity,
    not a rare edge case. Pooling every skeleton pixel's width regardless of
    which component it belongs to sidesteps that: a static, thin, high-contrast
    device artifact (e.g. a microchannel wall) and a genuine growing signal (e.g.
    a tube) each occupy their own tight width range, and a real artifact/signal
    pair shows up as two well-separated clusters in the pooled distribution
    across thousands of readings, regardless of any single frame's own
    connectivity being fragmented.

    Finds the natural split between the two clusters via Otsu's method (the
    same technique already used elsewhere in this module for intensity
    thresholding) applied to the pooled width values instead of pixel
    intensities. Requires the split to be both well-populated and cleanly
    separated (see the DECLUTTER_AUTO_* constants above) before trusting it —
    an ambiguous or single-population distribution means there's no reliable
    basis for a radius, and this fails safe to 0 (disabled) rather than
    guessing, since a wrong automatic radius could silently erode real signal
    or leave the artifact untouched with no visible warning that it happened.

    Returns (radius: int, info: dict) — info always has a 'reason' key
    explaining the outcome, plus supporting numbers when a radius was found.
    """
    from skimage.morphology import skeletonize
    from skimage.filters import threshold_otsu

    widths = []
    for idx in sample_frame_indices:
        fr = np.asarray(im_stack[idx]).astype(np.float32)
        # Top-hat size (31) is a fixed, generous multiple of the width range
        # this function is designed to separate (a handful of px vs a few tens
        # of px) — large enough to strip smooth illumination gradients without
        # touching either cluster of genuine ridge structures.
        bg = ndimage.grey_opening(fr, size=31)
        tophat = fr - bg
        tophat[tophat < 0] = 0
        if not np.any(tophat):
            continue
        thr = np.percentile(tophat, 97)
        mask = tophat > thr
        if not mask.any():
            continue
        dist = ndimage.distance_transform_edt(mask)
        skel = skeletonize(mask)
        if skel.any():
            widths.append(2 * dist[skel])

    if not widths:
        return 0, {'reason': 'no bright ridge-like structures found in sampled frames'}

    widths = np.concatenate(widths)
    if widths.size < DECLUTTER_AUTO_MIN_SAMPLES:
        return 0, {'reason': 'too few width samples ({}) to estimate reliably'.format(widths.size)}

    otsu_diameter = float(threshold_otsu(widths.astype(np.float64)))
    below = widths[widths < otsu_diameter]
    above = widths[widths >= otsu_diameter]

    if below.size == 0 or above.size == 0:
        return 0, {'reason': 'no bimodal separation found — single population of widths'}

    narrow_p90 = float(np.percentile(below, 90))
    wide_p10 = float(np.percentile(above, 10))

    info = {
        'n_frames_sampled': len(sample_frame_indices),
        'n_width_samples': int(widths.size),
        'otsu_diameter': otsu_diameter,
        'narrow_cluster_p90': narrow_p90,
        'wide_cluster_p10': wide_p10,
    }

    if wide_p10 < narrow_p90 * DECLUTTER_AUTO_MIN_SEPARATION_RATIO:
        info['reason'] = 'narrow/wide width clusters not clearly separated'
        return 0, info

    info['reason'] = 'ok'
    return max(1, round(otsu_diameter / 2)), info


def background(verbose, logger, work_inp_path, work_out_path, ext, res, module, eps, win,
               parallel, anim_save, h5_save, tiff_save, frange, single_channel=False,
               declutter_radius=0, declutter_auto=False, background_method='dbscan',
               tophat_size=0):
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

    # Auto-estimate the declutter radius from a sample of raw frames, overriding
    # any manually-set declutter_radius for this run. Falls back to whatever
    # declutter_radius was passed in (0 by default) if no reliable estimate can
    # be found — see _estimate_declutter_radius for why that can happen.
    if declutter_auto:
        n_sample = min(DECLUTTER_AUTO_SAMPLE_FRAMES, len(frange))
        sample_positions = np.linspace(0, len(frange) - 1, n_sample).astype(int)
        sample_frame_indices = np.unique(frange[sample_positions])
        estimated_radius, info = _estimate_declutter_radius(all.im_stack, sample_frame_indices)
        if estimated_radius:
            declutter_radius = estimated_radius
            logger.info('(Background Subtraction) ' + val + ' declutter_auto: estimated radius '
                        + str(estimated_radius) + ' ' + str(info))
            if verbose:
                print((val.capitalize() + ' (Background Subtraction) declutter_auto: estimated radius {} ({})'
                      .format(estimated_radius, info)))
        else:
            logger.info('(Background Subtraction) ' + val + ' declutter_auto: no reliable radius found ('
                        + info.get('reason', '') + '), falling back to declutter_radius=' + str(declutter_radius))
            if verbose:
                print((val.capitalize() + ' (Background Subtraction) declutter_auto: no reliable radius found ({}), '
                      'falling back to declutter_radius={}'.format(info.get('reason', ''), declutter_radius)))

    # Assign class constants
    all.set_class_constants(verbose, res, logger, frange, eps, declutter_radius,
                             background_method, tophat_size)

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

    # Freak frame detection — flag frames that deviate strongly from their
    # local rolling median (MAD-based). Slow bleaching and natural oscillations
    # are ignored; only sharp local spikes or dips are reported.
    freak_frames = detect_freak_frames(channeli_dict)
    if freak_frames:
        csv_path = work_out_path + '_' + val + '_freak_frames.csv'
        with open(csv_path, 'w', newline='') as csvf:
            writer = csv.DictWriter(csvf, fieldnames=[
                'frame_number', 'frame_index', 'value', 'local_median', 'deviation_mad'])
            writer.writeheader()
            writer.writerows(freak_frames)
        if verbose:
            print((val.capitalize() + " freak frames detected: " + str(len(freak_frames))
                   + " — saved to " + csv_path))

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
