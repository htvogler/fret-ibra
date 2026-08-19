# Tutorial
The acceptor and donor image stacks should first be visualised in a package like ImageJ to extract general parameters like the range of frames of interest, region of interest, bit-depth, resolution, and the presence of shading/noise. These parameters are then set in the config file (.cfg) which also includes the path to the image stack files and other options for each module. A log file is generated with details of the input parameters after each module run.

The *Config_tutorial.cfg* file in */ibra* is used to demonstrate the functionality of this toolkit. Before processing, the donor and acceptor channel image stacks should be renamed using the following format
```txt
Acceptor *file_identifier*_acceptor.tif
Donor    *file_idenfitier*_donor.tif
```
If the images to be processed are single channel images, please add only the **acceptor** suffix to the filenames to process them. Note that noisy/confocal images might benefit from being preprocessed with a narrow kernal gaussian filter.

First, the *input_path* (**absolute path**), *filename*, and *extension* parameters need to be set.
```txt
input_path = ./examples/stack 
filename = Test
extension = tif
```

The range of frames to be processed is then set with the parameter *frames*. Colon-separated values denote continuous frames, while comma-separated values denote manually selected individual frames. Furthermore, the *bit_depth* must be set to either 8, 12, or 16.
```txt
frames = 1:6
bit_depth = 12
```

Finally, the user has the choice to turn on the parallel option. Note that some newer Mac distributions are not compatible with the parallelization module used in this package.
```txt
parallel = 1
```

## Modules
The user then has the option to run one of four modules. These modules are designed to run sequentially. Option 3 runs options 0-2 in batch mode. The workflow is as follows:
* 0 -> Background subtraction (acceptor channel)
* 1 -> Background subtraction (donor channel)
* 2 -> Ratiometric processing
* 3 -> Background subtraction (both channels) + Ratiometric processing (optional)
* 4 -> Bleach correction (optional)

## Background subtraction
The *background subtraction* modules (0 or 1) is run first.
```txt
option = 0
```

The background modules' parameters include *nwindow* (the number of tiles along the long axis of the image that the frame will be divided into) and the acceptor or donor channel *eps* values (depending on whether *option* is set to 0 or 1) for the DBSCAN clustering algorithm. *nwindow* **should be a factor of the image resolution** (default: 40 for 1280X960). If the *nwindow* parameter provided is unsuited to the image resolution, an error will be returned with a suggested initial value.

Note, that the higher the *eps* value, the larger the number of pixels that are considered foreground. Very high *eps* values can thus label background pixels as foreground, reducing the effectiveness of the background subtraction algorithm (default: 0.01). 
```txt
nwindow = 40
eps = 0.01
```

An optional *declutter_radius* parameter (default: 0, disabled) can be set when the sample sits on a static device with its own thin, bright, high-contrast features — e.g. reflective or fluorescent microchannel walls — that are indistinguishable from real signal at the tile level *eps* clustering operates on, but are consistently narrower than it. Setting *declutter_radius* to a pixel radius between the artifact's width and the real signal's width runs a grayscale morphological opening on every frame before clustering, removing the thin artifact while leaving the real signal intact. Leave it at 0 unless you see this specific symptom (thin structured lines surviving background subtraction) — it has no effect on ordinary samples.
```txt
declutter_radius = 0
```

Picking that radius by hand means finding one "representative" frame and measuring it — tedious and a little arbitrary on a stack of thousands of frames, and a single frame can be misleading (e.g. if the real signal happens to be unusually thin or split into separate pieces in that one frame). Setting *declutter_auto* to 1 instead estimates the radius automatically: it samples 30 frames spread across the whole run, measures the local width at every bright-ridge skeleton point in each (regardless of which connected component it belongs to, so a frame where the real signal happens to be fragmented into separate pieces doesn't throw it off), and looks for a natural two-cluster split (via Otsu's method) between narrow artifacts and wide real signal. If the two clusters aren't clearly separated — e.g. there's no such artifact in this sample at all — it logs why and leaves decluttering disabled (or falls back to *declutter_radius* if you set one) rather than guessing. Check the run's log file for the estimated radius and the numbers behind it.
```txt
declutter_auto = 0
```

An optional *background_method* parameter (default: `dbscan`) selects the algorithm used to estimate the background. The default, per-tile DBSCAN clustering described above, assumes nothing about how wide real signal is — correct for samples where it can be wider than one tile (root cross-sections, bulk cytoplasmic/membrane signal). Setting it to `tophat` instead estimates the background as a single large grayscale morphological opening (size *tophat_size*, in px) applied directly to the frame: a structuring element wider than the real signal can never fit inside it, so the background estimate at any point is always built from genuinely surrounding pixels, never from the signal itself. This makes it immune to two failure modes the tile/DBSCAN method has specifically for a *thin*, persistent structure (e.g. a growing tube): a tile partially covered by real signal getting its own inflated median used as background, and DBSCAN correctly telling signal and background apart statistically while the interpolated background value substituted for the signal still nearly matches its own brightness once that margin has eroded (e.g. from photobleaching over a long timelapse — neither failure depends on tile size or eps, and no amount of retuning either one fixes it). *tophat_size* must be set (and comfortably larger than the real signal's own width — measure it the same way as *declutter_radius*, e.g. via a top-hat + distance-transform check on one frame) whenever *background_method* is `tophat`. This is independent of *declutter_radius*/*declutter_auto* above — decluttering removes things *narrower* than real signal (a device artifact), this removes everything *wider* than it (the background trend); use either alone or both together depending on what the sample needs. Only correct for signal narrower than *tophat_size* — leave at `dbscan` unless your sample is a thin, persistent structure and you've confirmed (by comparing outputs) that DBSCAN is corrupting it.
```txt
background_method = dbscan
tophat_size = 0
```

The background subtraction module can then be run with multiple options including an output HDF5 file (-s) (necessary for further processing), a video animation of per-frame metrics (-a) and a TIFF output file (-t). Option (-e) indicates that all output options are switched on.
```bash
./ibra.py -c Config_tutorial.cfg -a -t -s -v
```
Once this module is run, the video animation can be used to optimize the *eps* values visually. As a general rule of thumb, the lower the *eps* value, the better the background subtraction. *eps* values that are too low, result in non-convergence of the clustering algorithm for a particular frame, which is recorded in the log file. 

## Ratiometric processing
Once both donor (*eps* of 0.01) and acceptor channels (*eps* of 0.012) have been processed, the *ratio* processing module should be run. The options include cropping (*crop*) the original image (by the top left and bottom right corner comma-separated coordinates) to the region of interest to speed up processing time. The default for *crop* is (0,0,0,0), which indicates that no cropping is performed. This module also includes boolean parameters for turning on image registration (*register*) and overlap correction (*union*). For best results, *register* and *union* should be set to 0 while the donor and acceptor channels are tested with suboptimal values of *eps*, and should only be set to 1 after the optimal values for *eps* are found.
```txt
crop = 0,0,0,0
register = 0
union = 0
```

The ratiometric processing module can then be used by setting *option* to 2 in the config file and re-running the package. This generates two plots which can be used along with the per-frame metrics animation generated by the background subtraction module to assess if the *eps* value must be redefined for specific frames of either channel.

### Ratio of the number of foreground pixels and the total number of pixels per frame
![Pixel Count](images/Test_pixelcount.png)

### Ratio of the median intensity of foreground pixels and image bit depth per frame
![Intensity](images/Test_intensity_nonbleach.png)

## Correcting individual frames
The median intensity per bit depth and ratio of foreground pixels to total pixels plots clearly show that frame 3 in the acceptor stack is a significant outlier with almost no foreground pixels. This can be corroborated with the background subtraction animation. If there is no experimental justification for this outlier, this frame can be corrected individually by re-running the background subtraction algorithm on frame 3 on the acceptor channel (*option* = 0), with a lower *eps* value, followed by re-running the ratiometric processing module (*option* = 2).
```txt
option = 0
frames = 3

nwindow = 40
eps = 0.009
```

## Bleach correction
Once all unexpected outliers have been corrected, the bleach correction module can be (optionally) run. The median intensity of the foreground in the donor and acceptor image stacks are used to estimate the range of frames (colon-separated) between which the bleaching effect can be fit. The type of fit: linear (regularized), exponential or loess regression must also be stated.
```txt
acceptor_bleach_range = 1:6
donor_bleach_range = 1:6
fit = linear
```

## GUI
To improve the usability of the toolkit, a simple GUI can be used to fill in the configuration parameters and run the package without directly accessing the config file.
```bash
./ibra.py -g
```
The GUI parameters are identical to those in the config file with the addition of an option use an existing config file. The desired configuration can be run using the *Run* button at the bottom of the GUI. When the *Run* button is pressed, a config file with the chosen set of parameters is saved in */ibra*. This saved config file can later be directly run using the *Config Filename* field along with the desired *Output Options*.

### GUI default setting
![GUI](images/GUI_example.png)

It should be noted that even when using the GUI, progress and error messages will only be displayed in the run terminal.
