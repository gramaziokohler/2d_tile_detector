# Perception: OpenCV-based 2D tile detector

[![DOI](https://zenodo.org/badge/611631118.svg)](https://zenodo.org/badge/latestdoi/611631118)


## Required software
1. FLIR's `SpinView`. From [here](https://www.flir.com/support-center/iis/machine-vision/downloads/spinnaker-sdk-and-firmware-download/)
2. MATRIX VISION `mvBlueCOUGAR` suite. From [here](https://www.matrix-vision.com/en/downloads/drivers-software/mvbluecougar-gigabit-ethernet-dual-gigabit-ethernet-10gige-ethernet/windows-7-8-1-10)

## Required hardware
1. A network adapter which supports jumbo frames.
2. A camera, duh..

# Installation
```
conda env create -f environment.yml
```

# Calibration

## Set Image Format to RBG8
Make sure the camera's image format is `RGB8` (default seems to be `BayerRG8`):
1. Start `mvDeviceConfigure(x64)`
2. Double-click the desired device from the list. This will open an instance of the `wxPropView` application for the specific camera.
3. Stop image acquisition by clicking on `Acquire`
4. Go to Setting->Base->Camera->GenICam->ImageFormatControl and change PixelFormat to `BGR8`
5. Hit Ctrl+S to save
6. Close the settings application

![image_format](misc/flir_bgr.PNG)

## Measure Board Properties

The width and height of the board are the number of squares across the width and height of the board.
Only the inner corners which connect two squares are to be counted. These are market with red circles.
![board_count](misc/calib_count.PNG)

Therefore, the board in the picture has 7 squares across one dimension and 4 across the other (are landscape and portrait interchangeable?)

The size of a square is simply the real-life size of a single square measured in centimeters.

## Running the Calibration
Use the following arguments:
`--url "C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti"` - The path to the GenTL producer endpoint.
`-m Blackfly S BFS-PGE-31S4C"`- use the connected "Blackfly S BFS-PGE-31S4C"
`--width 7` - number of inner corners across the board's width 
`--height 4`- number of inner corners across the board's height
`--square_size 3.5` - square size in cm

> The calibration script will append the current date and time to the file's name!
```commandline
python .\calibration.py --url "C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti" --width 7 --height 4 -m "Blackfly S BFS-PGE-19S4C" --square_size 3.5 --save_file "C:\Users\ckasirer\Downloads\calibration.cal"
```

> The camera feed should now show up. If it doesn't, check out the troubleshooting section to make sure the camera is properly detectable.

Press `c` to trigger the calibration. 
Check the terminal output, it should state:
```commandline
Image captured, object points found
```

If the following message appears:
```commandline
Image captured but NO object points found
```
Try moving the checkerboard around a bit, and make sure that the count and dimensions of the squares is correct.

To finish, press `q`. The calibration program will save the calibration data to the provided output file path. You should see the message:
```commandline
Calibration is finished. RMS: x.xxxxxxxxxxxxxxx
```

The result file should look something like this:
```
%YAML:1.0
---
K: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1.2324533133958817e+03, 0., 1.0888207398469019e+03, 0.,
       1.2524057141123801e+03, 6.1500193355283000e+02, 0., 0., 1. ]
D: !!opencv-matrix
   rows: 1
   cols: 5
   dt: d
   data: [ -1.6552682272717364e-01, -7.4027753806342909e-03,
       1.9001888186245344e-03, -3.7763582719531778e-02,
       1.4750392962499670e-02 ]
R: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ 4.6140803507142211e-03, 1.0347721391158395e-03,
       -3.5433204829117204e-01 ]
T: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ -2.4730677451507656e+01, -5.8149409719023204e+00,
       3.5763223775899661e+01 ]
NK: !!opencv-matrix
   rows: 3
   cols: 3
   dt: d
   data: [ 1.0977514648437500e+03, 0., 1.0396554652077612e+03, 0.,
       1.0781127929687500e+03, 6.1744659614560078e+02, 0., 0., 1. ]
ROI: !!opencv-matrix
   rows: 4
   cols: 1
   dt: d
   data: [ 39., 67., 1517., 1106. ]
```

## Troubleshooting

### Interface not responsive / Camera missing from device list

It is occasionally the case that `mvDeviceConfigure` or `wxPropView` are not responsive and/or all/some of the devices are missing from the device list interface.
Restarting the switch and re-detecting the cameras with FLIR's `SpinView` app has shown remarkable success rates restoring the camera interface to functioning state.

Shake the tree by following all or some of these steps as you see fit:
1. Power-cycle the PoE switch
2. Reconnect the (USB) NIC to your station
3. Open the `SpinView` app and search for any devices which appear with a red exclamation mark icon.
   1. Double click the failed camera and let `SpinView` re-configure it

Now try again in `mvDeviceConfigure`, all connected devices should be listed.
