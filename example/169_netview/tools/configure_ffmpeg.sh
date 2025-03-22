#!/bin/bash

 FORMATS="h264,vp9,av1"
 DEMUXERS="matroska,av1,h264"
 FILTERS="av1_frame_merge,av1_frame_split,av1_metadata,h264_metadata,h264_mp4toannexb,h264_redundant_pps,vp9_metadata,vp9_raw_reorder,vp9_superframe,vp9_superframe_split"

#FORMATS="h264"
#DEMUXERS="h264"
#FILTERS="h264_metadata,h264_mp4toannexb,h264_redundant_pps"

#    --enable-lto

EXTRA_OPTIONS=" \
  --disable-asm \
  --disable-vaapi \
  --disable-vulkan \
  --disable-runtime-cpudetect \
  --disable-autodetect"

SELECTED_FORMATS=" \
  --disable-everything \
  --enable-protocol=file \
  --enable-decoder=$FORMATS \
  --enable-demuxer=$DEMUXERS \
  --enable-parser=$FORMATS \
  --enable-bsf=$FILTERS"

#SELECTED_FORMATS=""

NO_DISTRIB="--enable-gpl --enable-version3 --enable-nonfree"
#  --disable-programs

./configure \
  $NO_DISTRIB \
  $EXTRA_OPTIONS \
  --disable-autodetect \
  --enable-static \
  --disable-shared \
	--disable-doc \
	--disable-iconv \
  --disable-postproc \
  --disable-w32threads \
  --disable-os2threads \
  --disable-network    \
  --disable-dwt \
  --disable-error-resilience \
  --disable-lsp      \
  --disable-faan     \
  --disable-pixelutils \
  --disable-encoders   \
  --disable-hwaccels \
  --disable-muxers \
  --disable-indevs \
  --disable-outdevs  \
  --disable-devices  \
  --disable-filters  \
  --disable-xlib  \
  --disable-debug  \
  --prefix=/home/martin/ffmpeg \
  --disable-avfilter  \
  --disable-swresample  \
  --disable-swscale \
  --disable-postproc \
  $SELECTED_FORMATS


