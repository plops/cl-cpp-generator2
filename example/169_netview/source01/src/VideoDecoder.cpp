//
// Created by martin on 3/16/25.
//

#include "VideoDecoder.h"

#include <avcpp/av.h>
#include <format.h>

#include <iostream>

using namespace std;

void VideoDecoder::initialize() {
  cout << "Initializing video decoder " << endl;
  auto version = avformat_version();
  auto versionStr =
      format("libavformat: {}.{}.{}", AV_VERSION_MAJOR(version),
             AV_VERSION_MINOR(version), AV_VERSION_MICRO(version));

  cout << versionStr << endl;

  av::init();
  m_isInitialized = true;
}