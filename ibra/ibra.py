#!/usr/bin/env python2.7
#  -*- coding: utf-8 -*-
"""
FRET-IBRA is used to create fully processed ratiometric images from input donor and acceptor intensity TIFF images
"""

import os, sys, getopt
import configparser
from functions import logit
import background_subtraction as bs
import ratiometric_processing as rp
import numpy as np

__version__='0.2.0'

def usage():
    print("")
    print("Program: FRET-IBRA (FRET-Image Background-subtracted Ratiometric Analysis)")
    print("Version: {}".format(__version__))
    print("")
    print("Usage:  ibra -c <config file> [Options]")
    print("")
    print("Options: -t   Output TIFF stack")
    print("         -v   Print progress output (verbose)")
    print("         -s   Save as HDF5 file")
    print("         -a   Save background subtraction animation (only background module)")
    print("         -e   Use all output options")
    print("         -h   Print usage")
    print("")

def main():
    # Print usage file
    if '-h' in sys.argv[1:]:
        usage()
        sys.exit()

    # Initialize flags
    tiff_save = False
    verbose = False
    h5_save = False
    anim_save = False

    # Check if config file is available
    if "-c" not in sys.argv[1:]:
        usage()
        raise IOError("Config file not provided")
    else:
        options, remainder = getopt.getopt(sys.argv[1:], 'c:tvsaeh')
        for opt, arg in options:
            if opt in ('-c'):
                cfname = arg

            if opt in ('-t'):
                print("hERE")
                tiff_save = True

            if opt in ('-v'):
                verbose = True

            if opt in ('-s'):
                h5_save = True

            if opt in ('-a'):
                anim_save = True

            if opt in ('-e'):
                tiff_save = True
                verbose = True
                h5_save = True
                anim_save = True

    # Initialize config files
    config = configparser.ConfigParser()
    config.read(cfname)

    # Initialize input/output paths
    inp_path = config['File Parameters'].get('input_path').encode("utf-8")
    fname = config['File Parameters'].get('filename').encode("utf-8")
    current_path = os.getcwd()

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

    work_inp_path += '/' + fname
    work_out_path = current_path + '/' + fname + '/'
    if not os.path.exists(work_out_path):
        os.makedirs(work_out_path)
    work_out_path += fname

    # Input options for continuous or manual frames
    frames = config['File Parameters'].get('frames')
    if (':' in frames):
        start,stop = frames.split(':')
        start = int(start)
        stop = int(stop)
        assert (stop >= start), "last frame should be greater than the first frame"
        frange = np.arange(start-1,stop)
    else:
        frange = frames.split(',')
        frange = [int(x) - 1 for x in frange]

    assert (min(frange) >= 0), "frames should only contain positive integers"

    # Input modules
    background = config['Modules'].getboolean('background')
    ratio = config['Modules'].getboolean('ratio')
    bleach = config['Modules'].getboolean('bleach')

    assert (int(background==True)+int(ratio==True)+int(bleach==True) < 2), "only one module can be run at a time"

    # Open log file
    logger = logit(work_out_path)

    # Background module options
    if (background):
        # Input window tile size and eps values for DBSCAN clustering algorithm
        win = int(config['Background Parameters'].get('window'))
        YFP_eps = float(config['Background Parameters'].get('acceptor_eps'))
        CFP_eps = float(config['Background Parameters'].get('donor_eps'))
        eps = [YFP_eps, CFP_eps]

        assert (win > 0), "window should be a positive integer"
        assert (YFP_eps >= 0), "YFP_eps should be a float >= 0"
        assert (CFP_eps >= 0), "CFP_eps should be a float >= 0"
        assert (int(anim_save==True)+int(h5_save==True) > 0), "animation and/or h5_save must be activated"

        # Run the background subtraction algorithm
        bs.background(verbose,logger,work_inp_path,work_out_path,eps,win,anim_save,h5_save,tiff_save,frange)

    # Ratio image module
    if (ratio):
        # Input crop dimensions
        crop = config['Ratio Parameters'].get('crop').split(',')
        crop = map(int,crop)

        # Input TIFF file resolution
        resolution = int(config['Ratio Parameters'].get('resolution'))
        res_types = [8, 12, 16]

        assert (resolution in res_types), "resolution must be 8, 12, or 16-bit"
        res = np.power(2,resolution)-1

        # Input options for image registration and the union between donor and accepter channels, and output option for saving in HDF5
        register = config['Ratio Parameters'].getboolean('register')
        union = config['Ratio Parameters'].getboolean('union')

        # Run the ratio image processing algorithm
        rp.ratio(verbose,logger,work_out_path,crop,res,register,union,h5_save,tiff_save,frange)

    # Bleach correction module
    if (bleach):
        # Input the bleaching range for donor and accepter channels
        YFP_range = config['Bleach Parameters'].get('acceptor_bleach_range').split(':')
        CFP_range = config['Bleach Parameters'].get('donor_bleach_range').split(':')
        YFP_range = map(int,YFP_range)
        CFP_range = map(int,CFP_range)

        assert (YFP_range[1] >= YFP_range[0]), "acceptor_bleach_range last frame should be >= acceptor_bleach_range first frame"
        assert (CFP_range[1] >= CFP_range[0]), "donor_bleach_range last frame should be >= donor_bleach_range first frame"

        # Input bleach correction for fitting and correcting image median intensity
        fitter = config['Bleach Parameters'].get('fit')
        fits = ['linear','exponential']

        assert (fitter in fits), "fit should be either linear or exponential"

        # Run bleach correction algorithm
        rp.bleach(verbose,logger,work_out_path,YFP_range,CFP_range,fitter,h5_save,tiff_save)

if __name__ == "__main__":
    main()
