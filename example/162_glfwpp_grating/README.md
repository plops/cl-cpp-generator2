display a grating on the video projector.

eventually, i want to build a 3d scanner


|       |   |                                                                   |
| gen01 |   | display vertical and horizontal binary gratings, optional inverse |
| gen02 |   | add some kind of barcode to indicate frame id                     |
|       |   |                                                                   |


- display a binary pattern of stripes for structured illumination 3d reconstruction
- a barcode with extra bits for error correction is displayed to communicate the frame id
- the images are intended to be displayed with a video projector and
  captured with sufficiently fast slow-motion camera (e.g. display
  with 30Hz and capture with iphone at 240Hz)
  
- frame to frame timing is measured and mean and standard deviation is
  shown. high standard deviation indicates problems with the graphics
  card (e.g. busy or the selected framerates don't match).
  
- the barcode is shown as vertical stripes. on my laptop i found
  screen tearing at the top of the screen, even though i enable
  vsync. that means the top of the screen always shows one earlier
  frame.

- if more than one screen is connected to the computer the screen tear
  may run accross the screen. it may help to use `xrandr` with the
  option `--primary` on the display that is going to be captured.
