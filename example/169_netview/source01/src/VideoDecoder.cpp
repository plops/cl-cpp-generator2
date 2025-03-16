//
// Created by martin on 3/16/25.
//

#include "VideoDecoder.h"

#include <avcpp/av.h>
#include <format.h>

#include <iostream>

using namespace std;

bool VideoDecoder::initialize(const string& uri) {
  cout << "Initializing video decoder " << endl;
  auto version = avformat_version();
  auto versionStr =
      format("libavformat: {}.{}.{}", AV_VERSION_MAJOR(version),
             AV_VERSION_MINOR(version), AV_VERSION_MICRO(version));

  cout << versionStr << endl;

  av::init();
  av::setFFmpegLoggingLevel(AV_LOG_DEBUG);
  ctx = make_unique<av::FormatContext>();
  ctx->openInput(uri, ec);
  if (ec) {
    cerr << "Error opening input file " << uri << " " << ec.message() << endl;
    return false;
  }
  ctx->findStreamInfo(ec);
  if (ec) {
    cerr << "Error finding stream information " << ec.message() << endl;
  }
  isInitialized = true;
  return true;
}