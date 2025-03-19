#!/bin/bash
# --enable-libaom av1 decode 
#   --enable-libdav1d        enable AV1 decoding via libdav1d [no]
#   --enable-libvpx vp9 decode
  # --disable-safe-bitstream-reader 
  #                          disable buffer boundary checking in bitreaders
  #                          (faster, but may crash)
# --disable-asm

./configure \
--enable-gpl \
--enable-version3 \
--enable-nonfree \
--enable-static \
--disable-runtime-cpudetect \
--disable-all \
--disable-autodetect \
--disable-programs \
--disable-doc \
  --disable-avdevice \
  --disable-avcodec  \
  --disable-avformat \
  --disable-swresample \
  --disable-swscale \
  --disable-postproc \
  --disable-avfilter \
  --disable-pthreads \
  --disable-w32threads \
  --disable-os2threads \
  --disable-network    \
  --disable-dwt \
  --disable-error-resilience \
  --disable-lsp      \
  --disable-faan     \
  --disable-pixelutils \
  --disable-everything \
  --disable-encoders   \
  --enable-decoder=vp9,h264,av1 \
  --enable-hwaccel=av1_vaapi,h264_vaapi,vp9_vaapi \
  --disable-muxers \
  --enable-demuxer=matroska,mpegtsraw,webm_dash_manifest,av1,h264,m4v,mpegts,mpegtsraw \
  --enable-parser=h264,vp9,av1 \
  --enable-bsf=av1_frame_merge,av1_frame_split,av1_metadata,h264_metadata,h264_mp4toannexb,h264_redundant_pps,vp9_metadata,vp9_raw_reorder,vp9_superframe,vp9_superframe_split \
  --disable-protocols \
    --disable-indevs \
  --disable-outdevs  \
  --disable-devices  \
  --disable-filters  \
     --disable-xlib  \ 
  --disable-zlib    \
  --enable-vaapi   \
  --enable-vulkan  \
  --arch=znver3 \
  --cpu=znver3
  --enable-lto \
    --enable-hardcoded-tables use hardcoded tables instead of runtime generation
  --disable-debug  \
   --enable-extra-warnings 
