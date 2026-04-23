# -*- coding: utf-8 -*-
"""
Image registration, union, ratiometric processing and bleach correction
"""

import numpy as np
import numpy.testing as test
import imreg_dft as ird
import cv2
from functions import h5, logit, time_evolution, tiff, bleach_fit, ratio_calc
from timeit import default_timer as timer
import h5py
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage

# Bleach correction module
def bleach(verbose,logger,work_out_path,acceptor_bound,donor_bound,fitter,h5_save,tiff_save,frange,single_channel=False,crop=None):
    # Start time
    time_start = timer()

    if single_channel:
        # Single-channel mode: read acceptor stack and masked median intensities from _back.h5
        try:
            with h5py.File(work_out_path + '_back.h5', 'r') as f3:
                ratio_frange = np.array(f3.attrs['acceptor_frange'])
                acceptor = np.array(f3['acceptor'])
                acceptori = dict(zip(ratio_frange, np.array(f3['acceptori'])))
        except KeyError:
            raise ImportError("acceptori not found in _back.h5 — re-run background subtraction (module 0) to regenerate")
        except:
            raise ImportError(work_out_path + "_back.h5 not found")
        donori = acceptori  # placeholder — not used in single-channel mode
        donor = None
    else:
        # Two-channel mode: prefer _ratio_back.h5 (0+1+2+4 path); fall back to _back.h5 (0+1+4 path)
        import os as _os
        _ratio_h5 = work_out_path + '_ratio_back.h5'
        _back_h5  = work_out_path + '_back.h5'
        if _os.path.exists(_ratio_h5):
            _two_ch_source = 'ratio'
            try:
                with h5py.File(_ratio_h5, 'r') as f3:
                    ratio_frange = np.array(f3.attrs['ratio_frange'])
                    acceptor  = np.array(f3['acceptor'])
                    donor     = np.array(f3['donor'])
                    acceptori = dict(zip(ratio_frange, np.array(f3['acceptori'])))
                    donori    = dict(zip(ratio_frange, np.array(f3['donori'])))
            except:
                raise ImportError(work_out_path + "_ratio_back.h5 could not be read")
        elif _os.path.exists(_back_h5):
            _two_ch_source = 'back'
            try:
                with h5py.File(_back_h5, 'r') as f3:
                    ratio_frange = np.array(f3.attrs['acceptor_frange'])
                    acceptor  = np.array(f3['acceptor'])
                    donor     = np.array(f3['donor'])
                    acceptori = dict(zip(ratio_frange, np.array(f3['acceptori'])))
                    donori    = dict(zip(ratio_frange, np.array(f3['donori'])))
            except KeyError as e:
                raise ImportError("_back.h5 missing expected dataset: {} — re-run modules 0 and 1".format(e))
            except:
                raise ImportError(work_out_path + "_back.h5 could not be read")
        else:
            raise ImportError("Neither _ratio_back.h5 nor _back.h5 found — run modules 0+1 before bleach correction")

    # Fit and correct acceptor channel intensity
    nframes = acceptor.shape[0]
    if (acceptor_bound[1] > acceptor_bound[0]):
        acceptor_bound = np.subtract(acceptor_bound, 1)
        assert (sum(~np.isin(acceptor_bound,ratio_frange)) == 0), "acceptor_bleach_range should be within processed frame range"

        # Range of frames to fit (brange) and range to correct (acceptor_crange)
        acceptor_brange = np.arange(acceptor_bound[0], acceptor_bound[1] + 1)
        acceptor_crange = np.arange(acceptor_brange[0], ratio_frange[-1] + 1)

        # Find correction multiplier
        acceptor_corr = bleach_fit(acceptor_brange,acceptor_crange,acceptori,fitter)

        # Update bleaching correction factor
        acceptorb = np.concatenate((np.ones(nframes-len(acceptor_crange)),acceptor_corr.reshape(-1)),axis=0)

        # Update image
        acceptor[:,:,:] = np.uint16(np.multiply(acceptor[:,:,:],acceptorb.reshape(-1,1,1)))

        # Update image median intensity
        acceptori_frange = np.array([acceptori[x] for x in ratio_frange])
        acceptori = dict(zip(ratio_frange, np.float16(np.multiply(acceptori_frange,acceptorb.reshape(-1)))))

        # Save acceptor bleaching factors
        if (h5_save):
            if single_channel:
                h5_path = work_out_path + '_back.h5'
            else:
                h5_path = work_out_path + ('_back.h5' if _two_ch_source == 'back' else '_ratio_back.h5')
            h5(acceptorb,'acceptorb',h5_path,ratio_frange)

    # Fit and correct donor channel intensity (two-channel mode only)
    if not single_channel and (donor_bound[1] > donor_bound[0]):
        donor_bound = np.subtract(donor_bound, 1)
        assert (sum(~np.isin(donor_bound, ratio_frange)) == 0), "donor_bleach_range should be within processed frame range"

        # Range of frames to fit (brange) and range to correct (donor_crange)
        donor_brange = np.arange(donor_bound[0], donor_bound[1] + 1)
        donor_crange = np.arange(donor_brange[0],ratio_frange[-1] + 1)

        # Find correction multiplier
        donor_corr = bleach_fit(donor_brange,donor_crange,donori,fitter)

        # Update bleaching correction factor
        donorb = np.concatenate((np.ones(nframes-len(donor_crange)),donor_corr.reshape(-1)),axis=0)

        # Update image
        donor[:,:,:] = np.uint16(np.multiply(donor[:,:,:],donorb.reshape(-1,1,1)))

        # Update image median intensity
        donori_frange = np.array([donori[x] for x in ratio_frange])
        donori = dict(zip(ratio_frange, np.float16(np.multiply(donori_frange,donorb.reshape(-1)))))

        # Save donor bleaching factors
        if (h5_save):
            _donorb_path = work_out_path + ('_back.h5' if _two_ch_source == 'back' else '_ratio_back.h5')
            h5(donorb,'donorb',_donorb_path,ratio_frange)

    # End time
    time_end = timer()
    time_elapsed = str(int(time_end - time_start)+1)

    if (verbose):
        print("(Bleach Correction) Time: " + time_elapsed + " second(s)")

    # Update log file
    if single_channel:
        logger.info('(Bleach Correction) single-channel, acceptor_bleach_frames: ' + str(acceptor_bound[0]+1) + '-' + str(ratio_frange[-1] + 1)
                    + ', fit: ' + str(fitter) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))
    else:
        logger.info('(Bleach Correction) ' + 'acceptor_bleach_frames: ' + str(acceptor_bound[0]+1) + '-' + str(ratio_frange[-1] + 1)
                    + ', donor_bleach_frames: ' + str(donor_bound[0]+1) + '-' + str(ratio_frange[-1] + 1) + ', fit: ' + str(fitter)
                    + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))

    # Create plot to show median intensity over frame number after bleaching
    time_evolution(acceptori,donori,work_out_path,'_intensity_bleach.png','Median Intensity/Bit Depth',h5_save=False,single_channel=single_channel)

    # Apply spatial crop to corrected stacks before saving if crop parameters provided.
    # crop = [x0, y0, x1, y1]; [0,0,0,0] or None means no crop.
    if crop is not None and crop != [0, 0, 0, 0]:
        Ydim, Xdim = acceptor.shape[1:]
        x0, y0 = crop[0], crop[1]
        x1 = crop[2] if crop[2] != 0 else Xdim
        y1 = crop[3] if crop[3] != 0 else Ydim
        acceptor = acceptor[:, y0:y1, x0:x1]
        if not single_channel and donor is not None:
            donor = donor[:, y0:y1, x0:x1]

    # Save bleach corrected output
    if (h5_save or tiff_save):
        if single_channel:
            # Single-channel: save corrected acceptor stack back to _back.h5 and TIFF
            if (h5_save):
                h5_time_start = timer()
                h5(acceptor,'acceptor',work_out_path+'_back.h5',frange)
                with h5py.File(work_out_path+'_back.h5','a') as _f:
                    _f.attrs['bleach_corrected'] = True
                h5_time_end = timer()
                if (verbose):
                    print("Saving bleach corrected Acceptor stack in " + work_out_path+'_back.h5' + ' [Time: ' + str(int(h5_time_end - h5_time_start) + 1) + " second(s)]")
            if (tiff_save):
                tiff_time_start = timer()
                tiff(acceptor, work_out_path + '_acceptor_back_bleach.tif')
                tiff_time_end = timer()
                if (verbose):
                    print("Saving bleach corrected Acceptor TIFF in " + work_out_path + '_acceptor_back_bleach.tif' + ' [Time: ' + str(int(tiff_time_end - tiff_time_start)+1) + " second(s)]")
        else:
            # Two-channel: save bleach corrected channel stacks.
            # 0+1+2+4 path (_ratio_back.h5 was source): write to _ratio_back.h5 and set attribute there.
            # 0+1+4 path (_back.h5 was source): write to _back.h5 only.
            if (h5_save):
                h5_time_start = timer()
                if _two_ch_source == 'ratio':
                    h5(acceptor,'acceptor',work_out_path+'_ratio_back.h5',frange)
                    h5(donor,'donor',work_out_path+'_ratio_back.h5',frange)
                    with h5py.File(work_out_path+'_ratio_back.h5','a') as _f:
                        _f.attrs['bleach_corrected'] = True
                    h5_time_end = timer()
                    if (verbose):
                        print("Saving bleach corrected Acceptor and Donor stacks in " + work_out_path+'_ratio_back.h5' + ' [Time: ' + str(int(h5_time_end - h5_time_start) + 1) + " second(s)]")
                else:
                    h5(acceptor,'acceptor',work_out_path+'_back.h5',frange)
                    h5(donor,'donor',work_out_path+'_back.h5',frange)
                    h5_time_end = timer()
                    if (verbose):
                        print("Saving bleach corrected Acceptor and Donor stacks in " + work_out_path+'_back.h5' + ' [Time: ' + str(int(h5_time_end - h5_time_start) + 1) + " second(s)]")
                with h5py.File(work_out_path+'_back.h5','a') as _f:
                    _f.attrs['bleach_corrected'] = True
            if (tiff_save):
                tiff_time_start = timer()
                tiff(acceptor, work_out_path + '_acceptor_back_bleach.tif')
                tiff(donor, work_out_path + '_donor_back_bleach.tif')
                tiff_time_end = timer()
                if (verbose):
                    print("Saving bleach corrected Acceptor TIFF in " + work_out_path + '_acceptor_back_bleach.tif' + ' [Time: ' + str(int(tiff_time_end - tiff_time_start)+1) + " second(s)]")
                    print("Saving bleach corrected Donor TIFF in " + work_out_path + '_donor_back_bleach.tif')
            # Recalculate and save ratio from bleach corrected channels (only when _ratio_back.h5 exists)
            if _two_ch_source == 'ratio':
                ratio, ratio_raw = ratio_calc(acceptor,donor)
                if (h5_save):
                    h5_time_start = timer()
                    h5(ratio,'ratio',work_out_path+'_ratio_back.h5',frange)
                    h5(ratio_raw,'ratio_raw',work_out_path+'_ratio_back.h5',frange)
                    h5_time_end = timer()
                    if (verbose):
                        print("Saving bleach corrected Ratio stack in " + work_out_path+'_ratio_back.h5' + ' [Time: ' + str(int(h5_time_end - h5_time_start) + 1) + " second(s)]")
                if (tiff_save):
                    tiff_time_start = timer()
                    tiff(ratio, work_out_path + '_ratio_back_bleach.tif')
                    tiff_time_end = timer()
                    if (verbose):
                        print("Saving bleach corrected Ratio TIFF in " + work_out_path + '_ratio_back_bleach.tif' + ' [Time: ' + str(int(tiff_time_end - tiff_time_start)+1) + " second(s)]")


def ratio(verbose,logger,work_out_path,crop,res,register,union,h5_save,tiff_save,frange):
    # Start time
    time_start = timer()

    # Input background subtracted image stack
    try:
        with h5py.File(work_out_path + '_back.h5', 'r') as f:
            try:
                acceptor = np.array(f['acceptor'])
                acceptor_frange = np.array(f.attrs['acceptor_frange'])
            except:
                raise ImportError("Acceptor stack background not processed")
            try:
                donor = np.array(f['donor'])
                donor_frange = np.array(f.attrs['donor_frange'])
            except:
                raise ImportError("Donor stack background not processed")
    except ImportError:
        raise
    except:
        raise ImportError(work_out_path + "_back.h5 not found")

    # Find frame dimensions and intersection between processed frames and input frames
    Ydim, Xdim = acceptor.shape[1:]
    brange = np.intersect1d(frange,acceptor_frange,return_indices=True)[2]

    # Set default values for crop
    if (crop[2] == 0):
        crop[2] = Xdim
    if (crop[3] == 0):
        crop[3] = Ydim

    # Testing input values
    test.assert_array_equal (acceptor_frange,donor_frange), "Acceptor and Donor stacks have different frame numbers"
    assert (sum(~np.isin(frange,acceptor_frange)) == 0), "background subtracted stacks have not been processed for all frame values"
    assert (crop[2] >= crop[0]), "crop[2] must be greater than crop[0]"
    assert (crop[3] >= crop[1]), "crop[3] must be greater than crop[1]"
    assert (crop[0] >= 0), "crop[0] must be >= 0"
    assert (crop[2] <= Xdim), "crop[2] must be <= than the width of the image"
    assert (crop[1] >= 0), "crop[1] must be >= 0"
    assert (crop[3] <= Ydim), "crop[3] must be <= than the height of the image"

    # Image crop
    acceptorc = acceptor[:,crop[1]:crop[3],crop[0]:crop[2]]
    donorc = donor[:,crop[1]:crop[3],crop[0]:crop[2]]

    # Search for saved ratio images
    try:
        # Input files into dictionaries
        with h5py.File(work_out_path + '_ratio_back.h5', 'r') as f2:
            ratio_frange = np.array(f2.attrs['ratio_frange'])
            acceptori = dict(list(zip(ratio_frange, np.array(f2['acceptori']))))
            donori = dict(list(zip(ratio_frange, np.array(f2['donori']))))
    except:
        # Initialize empty dictionaries for intensities
        acceptori, donori = {},{}

    # Initialize empty dictionaries for pixel counts
    acceptornz, donornz = {},{}

    # Set up constants for loop
    mult = np.float32(255)/np.float32(res)
    ires = 100/np.float32(res)
    ipix = 100/(Xdim*Ydim)

    # Loop through frames
    for count,frame in list(zip(frange,brange)):
        if (verbose):
            print ("(Ratio Processing) Frame Number: " + str(count+1))

        # Image registration for donor channel
        if (register):
            trans = ird.translation(acceptorc[frame,:,:], donorc[frame,:,:])
            tvec = trans["tvec"].round(4)
            donorc[frame,:,:] = np.round(ird.transform_img(donorc[frame,:,:], tvec=tvec))

        # Thresholding
        acceptors = np.uint8(np.float32(acceptorc[frame, :, :]) * mult)
        donors = np.uint8(np.float32(donorc[frame, :, :]) * mult)

        # Check for max image intensity
        if np.uint32(np.amax(acceptors)) + np.uint32(np.amax(donors)) > 70:
            # Otsu thresholding for normal intensity images
            _, A_thresh = cv2.threshold(acceptors, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            _, B_thresh = cv2.threshold(donors, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            # Simple thresholding for low intensity images
            _, A_thresh = cv2.threshold(acceptors, 3, 255, cv2.THRESH_BINARY)
            _, B_thresh = cv2.threshold(donors, 3, 255, cv2.THRESH_BINARY)

        # Setting values below threshold to zero
        acceptorc[frame,:,:] *= np.uint16(A_thresh/255)
        donorc[frame,:,:] *= np.uint16(B_thresh/255)

        # Consider only foreground pixel intensity overlapping between donor and acceptor channels to ensure channels overlap perfectly
        if (union):
            # Create mask for overlapping pixels
            C = np.multiply(A_thresh, B_thresh)
            C[C > 0] = 1

            # Set non-overlapping pixels to zero
            acceptorc[frame,:,:] *= C
            donorc[frame,:,:] *= C

        # Count number of non-zero pixels by total pixels per frame
        acceptornz[count] = np.count_nonzero(A_thresh)*ipix
        donornz[count] = np.count_nonzero(B_thresh)*ipix

        # Find the ratio of the median non-zero intensity pixels and the bit depth per frame for the acceptor stack
        if (np.amax(acceptorc[frame,:,:]) > 0.0):
            acceptori[count] = ndimage.median(acceptorc[frame,:,:], labels = A_thresh/255)*ires
        else:
            acceptori[count] = 0

        # Find the ratio of the median non-zero intensity pixels and the bit depth per frame for the donor stack
        if (np.amax(donorc[frame,:,:])> 0.0):
            donori[count] = ndimage.median(donorc[frame,:,:], labels = B_thresh/255)*ires
        else:
            donori[count] = 0

    # End time
    time_end = timer()
    time_elapsed = str(int(time_end - time_start)+1)
    if (verbose):
        print(("(Ratio Processing) Time: " + time_elapsed + " second(s)"))

    # Update log file to save stack metrics
    print_range = [x + 1 for x in frange]
    if (max(np.ediff1d(frange,to_begin=frange[0])) > 1):
        logger.info('(Ratio Processing) ' + 'frames: ' + ",".join(list(map(str, print_range))) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))
    else:
        logger.info('(Ratio Processing) ' + 'frames: ' + str(print_range[0]) + '-' + str(print_range[-1]) + ', time: ' + time_elapsed + ' sec, save: ' + str(h5_save))


    # Create plot to showcase median intensity over frame number and the number of foreground pixels per channel (NON-bleach corrected)
    time_evolution(acceptori,donori,work_out_path,'_intensity_nonbleach.png','Median Intensity/Bit Depth',h5_save)
    time_evolution(acceptornz,donornz,work_out_path,'_pixelcount.png','Foreground/Total Image Pixels',h5_save)

    # Calculate 8-bit ratio image with NON-bleach corrected donor and acceptor channels
    if (h5_save or tiff_save):
        # Calculate ratio stack
        ratio, ratio_raw = ratio_calc(acceptorc,donorc)

        # Save processed images, non-zero pixel count, median intensity and ratio processed images in HDF5 format
        if (h5_save):
            acceptori_brange = np.array([acceptori[a] for a in brange])
            donori_brange = np.array([donori[a] for a in brange])

            h5_time_start = timer()
            h5(acceptorc[brange,:,:],'acceptor',work_out_path+'_ratio_back.h5',frange)
            h5(donorc[brange,:,:],'donor',work_out_path+'_ratio_back.h5',frange)
            h5(acceptori_brange, 'acceptori', work_out_path + '_ratio_back.h5', frange)
            h5(donori_brange, 'donori', work_out_path + '_ratio_back.h5', frange)
            h5(ratio[brange,:,:], 'ratio', work_out_path + '_ratio_back.h5',frange)
            h5(ratio_raw[brange,:,:], 'ratio_raw', work_out_path + '_ratio_back.h5',frange)
            h5_time_end = timer()

            if (verbose):
                print(("Saving Acceptor, Donor and Ratio stacks in " + work_out_path+'_ratio_back.h5' + ' [Time: ' + str(int(h5_time_end - h5_time_start) + 1) + " second(s)]"))
    
        # Save NON-bleach corrected ratio image as TIFF
        if (tiff_save):
            tiff_time_start = timer()
            tiff(ratio, work_out_path + '_ratio_back.tif')
            tiff_time_end = timer()

            if (verbose):
                print(("Saving unbleached Ratio TIFF stack in " + work_out_path + '_ratio_back.tif' + ' [Time: ' + str(int(tiff_time_end - tiff_time_start)+1) + " second(s)]"))