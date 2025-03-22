# capnp compile -oc++ video.capnp

@0xb1039f6a7c611b12;

interface VideoArchive {
    getVideoList @0 () -> (videoList :VideoList);
    getVideoInfo @1 (filePath :Text) -> (videoInfo :VideoInfo);
}

struct Rational {
    top @0 :Int32;
    bottom @1 :Int32;
}

struct KeyFrame {
    timePosition @0 :UInt64;
    timebase @1 :Rational;
    durationSincePreviousKeyframe @2 :Float32;
    packetIndex @3 :UInt64;
    packetSize @4 :UInt64;
    frameSize @5 :UInt64;
    frameWidth @6 :Int32;
    frameHeight @7 :Int32;
    quality @8 :Int32;
    bitsPerPixel @9 :Int32;
    rawSize @10 :Int32;
}

struct VideoInfo {
  id @0 :UInt64;       # Unique ID for each video (e.g., hash of path)
  filePath @1 :Text;    # Relative or absolute path on the server
  fileName @2 :Text;   # filename
  fileSize @3 :UInt64;  # File size in bytes
  duration @4 :Float64; # Duration in seconds (obtained during server-side indexing)
  width @5 :UInt32;     # Video width
  height @6 :UInt32;    # Video height
  bitrate @7 :UInt64;
  keyFrames @8 :List(KeyFrame);
}

struct Video {
    name @0 :Text;
    sizeBytes @1 :UInt64;
}

struct VideoList {
    videos @0 :List(Video);
}

