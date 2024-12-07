display a grating on the video projector.

eventually, i want to build a 3d scanner


|       |   |                                                                   |
| gen01 |   | display vertical and horizontal binary gratings, optional inverse |
| gen02 |   | add some kind of barcode to indicate frame id                     |
|       |   |                                                                   |


- display a binary pattern of stripes for structured illumination 3d
  reconstruction
- a barcode with extra bits for error correction is displayed to
  communicate the frame id
- the images are intended to be displayed with a video projector and
  captured with sufficiently fast slow-motion camera (e.g. display
  with 30Hz and capture with iphone at 240Hz)
  
- frame to frame timing of the projector is measured and mean and
  standard deviation is shown. high standard deviation indicates
  problems with the graphics card (e.g. gpu busy or the selected
  framerates of gpu and projector don't match).
  
- the barcode is shown as vertical stripes. on my laptop i found
  screen tearing at the top of the screen, even though i enable
  vsync. that means the top of the screen always shows one earlier
  frame.

- if more than one screen is connected to the computer the screen tear
  may run accross the screen. it may help to use `xrandr` with the
  option `--primary` on the display that is going to be captured.

- observations with the iphones slow-motion capture indicate that when
  my lcd based projector runs at 60Hz, there is insufficient time to
  switch from black to white (or the reverse).

- when the projector is updated at 30Hz, some of the iphone
  slow-motion frames are properly illuminated while many show the lcd
  in a transitioning state. the 3d reconstruction software shall
  remove these frames. the update pattern seems to run over the
  display in a vertical motion (but i'm not sure). maybe it will be
  usefull to have a large white plane next to the 3d target that will
  alternatively be black or bright. only when this area is
  sufficiently uniform a frame ma be considered 'properly
  illuminated'.
