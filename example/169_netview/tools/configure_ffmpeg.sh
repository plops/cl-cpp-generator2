#!/bin/bash
# --enable-libaom av1 decode 
#   --enable-libdav1d        enable AV1 decoding via libdav1d [no]
#   --enable-libvpx vp9 decode
  # --disable-safe-bitstream-reader 
  #                          disable buffer boundary checking in bitreaders
  #                          (faster, but may crash)
# --disable-asm
#--disable-avformat
#--disable-avcodec 

# ./configure \
# --enable-gpl \
# --enable-version3 \
# --enable-nonfree \
# --enable-static \
# --disable-runtime-cpudetect \
# --disable-autodetect \
# --disable-programs \
# 	    --disable-doc \
# 	    --disable-iconv \
#   --disable-avdevice \
#   --disable-swresample \
#   --disable-swscale \
#   --disable-postproc \
#   --disable-avfilter \
#   --disable-muxers \
#   --disable-bsfs \
#   --disable-protocols \
#   --disable-everything  \
#   --enable-decoder=vp9,h264,av1,av1_decoder

  ./configure \
--disable-everything \
      --enable-gpl \
--enable-version3 \
--enable-nonfree \
      --enable-static \
      --disable-shared \
--disable-runtime-cpudetect \
--disable-autodetect \
--disable-programs \
	    --disable-doc \
	    --disable-iconv \
  --disable-avdevice \
  --disable-swresample \
  --disable-swscale \
  --disable-postproc \
  --disable-avfilter \
  --disable-w32threads \
  --disable-os2threads \
  --disable-network    \
  --disable-dwt \
  --disable-error-resilience \
  --disable-lsp      \
  --disable-faan     \
  --disable-pixelutils \
  --disable-encoders   \
  --enable-decoder=vp9,h264,av1,av1_decoder \
  --enable-hwaccel=av1_vaapi,h264_vaapi,vp9_vaapi \
  --disable-muxers \
  --enable-demuxer=matroska,av1,h264 \
  --enable-parser=h264,vp9,av1 \
  --enable-bsf=av1_frame_merge,av1_frame_split,av1_metadata,h264_metadata,h264_mp4toannexb,h264_redundant_pps,vp9_metadata,vp9_raw_reorder,vp9_superframe,vp9_superframe_split \
  --disable-protocols \
    --disable-indevs \
  --disable-outdevs  \
  --disable-devices  \
  --disable-filters  \
     --disable-xlib  \
  --enable-vaapi   \
  --enable-vulkan  \
  --enable-lto \
  --disable-debug  \
   --enable-extra-warnings 
 
