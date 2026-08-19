import os
import configparser
from functions import logit, ANIM_FRAME_WARN
import background_subtraction as bs
import ratiometric_processing as rp
import numpy as np

def main_extract(cfname,tiff_save,verbose,h5_save,anim_save):
    # Initialize config files
    config = configparser.ConfigParser()
    config.read(cfname)

    # Initialize input/output paths
    inp_path = config['File Parameters'].get('input_path').encode("utf-8").decode()
    fname = config['File Parameters'].get('filename').encode("utf-8").decode()
    ext = config['File Parameters'].get('extension').encode("utf-8").decode()
    current_path = os.getcwd()

    # Check for optional donor filename — if absent or empty, run in single-channel mode
    second_channel_raw = config['File Parameters'].get('second_channel', '').encode("utf-8").decode().strip()
    single_channel = second_channel_raw.lower() in ('', '0', 'no', 'false', 'off')

    # Finalize input/output paths
    if inp_path[:2] == '..':
        work_inp_path = current_path[:-5] + inp_path[2:]
    elif inp_path[0] == '.':
        work_inp_path = current_path[:-5] + inp_path[1:]
    else:
        work_inp_path = inp_path

    # Ensure that input path exists
    if not os.path.exists(work_inp_path):
        raise IOError("Input path does not exist")

    # Create FRET-IBRA_results folder in the input directory if it doesn't exist
    results_root = work_inp_path + '/FRET-IBRA_results'
    if not os.path.exists(results_root):
        os.makedirs(results_root)

    work_inp_path += '/' + fname
    work_out_path = results_root + '/' + fname + '/'
    if not os.path.exists(work_out_path):
        os.makedirs(work_out_path)
    work_out_path += fname

    # Input options for continuous or manual frames
    frames = config['File Parameters'].get('frames')
    if (':' in frames):
        start, stop = frames.split(':')
        start = int(start)
        stop = int(stop)
        assert (stop >= start), "last frame should be greater than the first frame"
        frange = np.arange(start - 1, stop)
    else:
        frange = frames.split(',')
        frange = np.array([int(x) - 1 for x in frange])

    assert (min(frange) >= 0), "frames should only contain positive integers"

    # Input modules
    module = int(config['Modules'].get('option'))

    assert (module >= 0), "option should be between 0 and 4"
    assert (module <= 4), "option should be between 0 and 4"

    # Single-channel mode: warn if a two-channel module was selected
    if single_channel and module in (1, 2, 3):
        print("\nWarning: second_channel is not set or set to 0 (single-channel mode) but option {} requires a donor channel.".format(module))
        print("In single-channel mode only option 0 (background subtraction, acceptor) is valid.")
        answer = input("Continue with option 0 instead? [y/n]: ").strip().lower()
        if answer == 'y':
            module = 0
        else:
            raise SystemExit("Aborted. Please set option = 0 in your config file for single-channel mode.")

    # Input TIFF file resolution
    resolution = int(config['File Parameters'].get('resolution'))
    res_types = [8, 12, 16]

    assert (resolution in res_types), "resolution must be 8, 12, or 16-bit"
    res = np.power(2, resolution) - 1

    # Input parallel option
    parallel_raw = config['File Parameters'].get('parallel', '').strip()
    parallel = parallel_raw.lower() in ('1', 'yes', 'true', 'on')

    # Open log file
    logger = logit(work_out_path)

    # Log whether running in single-channel or two-channel mode
    if single_channel:
        logger.info('Running in single-channel mode (second_channel not set or 0)')
    else:
        logger.info('Running in two-channel mode (second_channel = {})'.format(second_channel_raw))

    # Animation frame count warning — fires before any processing begins so the
    # user is not left waiting for a prompt after a long background subtraction run.
    # Only shown when -a or -e was passed (anim_save=True) and the frame range
    # exceeds the threshold. Fires once regardless of how many channels module 3
    # will process.
    if anim_save and (module <= 1 or module == 3) and len(frange) > ANIM_FRAME_WARN:
        print(("\nWarning: animation requested for {} frames. "
               "3D surface rendering is slow — this may take a very long time.\n"
               "For eps tuning, run a short frame range (10-20 frames) instead.\n"
               "Continue anyway? [y/n]: ").format(len(frange)), end='')
        answer = input().strip().lower()
        if answer != 'y':
            anim_save = False
            print("Background animation disabled for this run.")

    # Module 3 runs the full pipeline from scratch — warn if existing HDF5 output files
    # are present, since they may contain carefully tuned per-frame results
    if module == 3 and h5_save:
        existing = [f for f in (work_out_path + '_back.h5', work_out_path + '_ratio_back.h5')
                    if os.path.exists(f)]
        if existing:
            print("\nWarning: the following output files already exist and will be overwritten by module 3:")
            for f in existing:
                print("  {}".format(f))
            print("If you have tuned individual frames using modules 0 or 1, those results will be lost.")
            answer = input("Continue and overwrite? [y/n]: ").strip().lower()
            if answer == 'y':
                for f in existing:
                    os.remove(f)
                    logger.info('Removed existing output file for fresh run: {}'.format(f))
            else:
                raise SystemExit("Aborted. Run modules 0, 1, 2 and 4 sequentially to preserve per-frame tuning.")

    # Background module options
    if (module <= 1 or module == 3):
        # Input window tile size and eps values for DBSCAN clustering algorithm
        win = int(config['Background Parameters'].get('nwindow'))
        eps = float(config['Background Parameters'].get('eps'))

        # Radius (px) of the disk used for a pre-DBSCAN grayscale morphological
        # opening that suppresses thin, static, high-contrast artifacts (e.g.
        # reflective/fluorescent microchannel-device edges) which share the same
        # per-tile statistics as real signal at this resolution. 0/empty (default)
        # disables it — opt-in, since it changes pixel values before every
        # downstream step. Pick a radius between the artifact's width and the
        # real signal's width (e.g. via a quick top-hat + distance-transform
        # measurement on one frame) — it is not a universal constant.
        declutter_raw = config['Background Parameters'].get('declutter_radius', '').strip()
        declutter_radius = int(declutter_raw) if declutter_raw else 0
        assert (declutter_radius >= 0), "declutter_radius must be a non-negative integer (0 disables it)"

        # When enabled, the radius above is ignored and instead estimated per-run
        # from a sample of raw frames (see _estimate_declutter_radius in
        # background_subtraction.py) — removes the need to hand-pick a
        # "representative" frame and measure it manually on every new stack.
        # Falls back to declutter_radius above if no reliable estimate is found.
        declutter_auto_raw = config['Background Parameters'].get('declutter_auto', '').strip()
        declutter_auto = declutter_auto_raw.lower() in ('1', 'yes', 'true', 'on')

        # Which algorithm estimates the background. 'dbscan' (default) is the
        # original per-tile clustering method and remains correct for sample
        # types where real signal can be wider than any one tile (roots, bulk
        # cytoplasmic/membrane signal, etc). 'tophat' estimates the background
        # as a single large grayscale morphological opening instead — a
        # structuring element wider than the real signal can never fit inside
        # it, so the background estimate is always built from genuinely
        # surrounding pixels, never from the signal itself. Only correct for
        # signal narrower than tophat_size below (e.g. a thin growing tube) —
        # NOT a general replacement for 'dbscan', which is why it's opt-in.
        # Independent of declutter_radius/declutter_auto above: decluttering
        # removes things narrower than the real signal (e.g. a device
        # artifact), this removes everything wider than it (the background
        # trend) — different scales, different jobs, usable separately or
        # together.
        background_method = config['Background Parameters'].get('background_method', '').strip().lower() or 'dbscan'
        assert (background_method in ('dbscan', 'tophat')), "background_method must be 'dbscan' or 'tophat'"

        # Radius... rather, size (px) of the morphological opening used to
        # estimate the background when background_method='tophat'. Must be
        # comfortably larger than the real signal's own width (e.g. via the
        # same top-hat + distance-transform measurement used for
        # declutter_radius) — too small and it erodes the signal itself into
        # the background estimate; too large and it stops tracking genuine
        # local illumination variation. Ignored when background_method='dbscan'.
        tophat_size_raw = config['Background Parameters'].get('tophat_size', '').strip()
        tophat_size = int(tophat_size_raw) if tophat_size_raw else 0
        assert (tophat_size >= 0), "tophat_size must be a non-negative integer"
        assert (background_method != 'tophat' or tophat_size > 0), "tophat_size must be set (> 0) when background_method='tophat'"

        assert (win >= 10), "nwindow should be between 10 and 100"
        assert (win <= 100), "nwindow should be between 10 and 100"
        assert (eps > 0), "eps value must be a positive float between 0 and 1"
        assert (eps <= 1), "eps value must be a positive float between 0 and 1"
        assert (int(anim_save == True) + int(h5_save == True) > 0), "animation and/or h5_save must be activated"

        # Run the background subtraction algorithm for either acceptor or donor stack
        if module <= 1:
            bs.background(verbose, logger, work_inp_path, work_out_path, ext, res, module, eps, win, parallel, anim_save,
                      h5_save, tiff_save, frange, single_channel=single_channel, declutter_radius=declutter_radius,
                      declutter_auto=declutter_auto, background_method=background_method, tophat_size=tophat_size)
        # Automated background + ratio modules
        elif module == 3:
            # Run the background subtraction algorithm for the acceptor stack
            bs.background(verbose, logger, work_inp_path, work_out_path, ext, res, 0, eps, win, parallel, anim_save,
                          h5_save, tiff_save, frange, declutter_radius=declutter_radius, declutter_auto=declutter_auto,
                          background_method=background_method, tophat_size=tophat_size)

            # Run the background subtraction algorithm for the donor stack
            bs.background(verbose, logger, work_inp_path, work_out_path, ext, res, 1, eps, win, parallel, anim_save,
                          h5_save, tiff_save, frange, declutter_radius=declutter_radius, declutter_auto=declutter_auto,
                          background_method=background_method, tophat_size=tophat_size)

    # Ratio image module (two-channel only)
    if (module == 2 or module == 3):
        # Input crop dimensions
        crop = config['Ratio Parameters'].get('crop').split(',')
        crop = list(map(int, crop))

        # Input options for image registration and the union between donor and accepter channels
        register_raw = config['Ratio Parameters'].get('register', '').strip()
        register = register_raw.lower() in ('1', 'yes', 'true', 'on') if register_raw else True
        union_raw = config['Ratio Parameters'].get('union', '').strip()
        union = union_raw.lower() in ('1', 'yes', 'true', 'on') if union_raw else True

        # Run the ratio image processing algorithm
        rp.ratio(verbose, logger, work_out_path, crop, res, register, union, h5_save, tiff_save, frange)

    # Bleach correction module
    if (module == 4):
        # Input the bleaching range for the acceptor channel
        acceptor_bleach_raw = config['Bleach Parameters'].get('acceptor_bleach_range', '').strip()
        if not acceptor_bleach_raw or ':' not in acceptor_bleach_raw:
            raise ValueError("acceptor_bleach_range must be set as a colon-separated range (e.g. 1:100)")
        acceptor_bound = list(map(int, acceptor_bleach_raw.split(':')))

        # Input the bleaching range for the donor channel (ignored in single-channel mode)
        donor_bleach_raw = config['Bleach Parameters'].get('donor_bleach_range', '').strip()
        if not single_channel:
            if not donor_bleach_raw or ':' not in donor_bleach_raw:
                raise ValueError("donor_bleach_range must be set as a colon-separated range (e.g. 1:100)")
        donor_bound = list(map(int, donor_bleach_raw.split(':'))) if donor_bleach_raw and ':' in donor_bleach_raw else acceptor_bound

        assert (acceptor_bound[1] >= acceptor_bound[0]), "acceptor_bleach_range last frame should be >= acceptor_bleach_range first frame"
        assert (donor_bound[1] >= donor_bound[0]), "donor_bleach_range last frame should be >= donor_bleach_range first frame"

        # Input bleach correction for fitting and correcting image median intensity
        fitter = config['Bleach Parameters'].get('fit')
        fits = ['linear', 'exponential', 'loess']

        assert (fitter in fits), "fit should be either linear, exponential or loess"

        # Read crop parameters — applied to the corrected output stack before saving.
        # [0,0,0,0] means no crop. Parsed here so bleach() is self-contained.
        crop_raw = config['Ratio Parameters'].get('crop', '').strip()
        if crop_raw:
            try:
                crop = list(map(int, crop_raw.split(',')))
            except ValueError:
                crop = [0, 0, 0, 0]
        else:
            crop = [0, 0, 0, 0]

        # Run bleach correction algorithm
        rp.bleach(verbose, logger, work_out_path, acceptor_bound, donor_bound, fitter, h5_save, tiff_save, frange,
                  single_channel=single_channel, crop=crop)

    # Output message
    print("Processing is complete")
