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

        assert (win >= 10), "nwindow should be between 10 and 70"
        assert (win <= 70), "nwindow should be between 10 and 70"
        assert (eps > 0), "eps value must be a positive float between 0 and 1"
        assert (eps <= 1), "eps value must be a positive float between 0 and 1"
        assert (int(anim_save == True) + int(h5_save == True) > 0), "animation and/or h5_save must be activated"

        # Run the background subtraction algorithm for either acceptor or donor stack
        if module <= 1:
            bs.background(verbose, logger, work_inp_path, work_out_path, ext, res, module, eps, win, parallel, anim_save,
                      h5_save, tiff_save, frange, single_channel=single_channel)
        # Automated background + ratio modules
        elif module == 3:
            # Run the background subtraction algorithm for the acceptor stack
            bs.background(verbose, logger, work_inp_path, work_out_path, ext, res, 0, eps, win, parallel, anim_save,
                          h5_save, tiff_save, frange)

            # Run the background subtraction algorithm for the donor stack
            bs.background(verbose, logger, work_inp_path, work_out_path, ext, res, 1, eps, win, parallel, anim_save,
                          h5_save, tiff_save, frange)

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

        # Run bleach correction algorithm
        rp.bleach(verbose, logger, work_out_path, acceptor_bound, donor_bound, fitter, h5_save, tiff_save, frange,
                  single_channel=single_channel)

    # Output message
    print("Processing is complete")
