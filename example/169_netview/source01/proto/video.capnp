# capnp compile -oc++ video.capnp

@0xb1039f6a7c611b12;

interface VideoArchive {
    getVideoList @0 () -> (videoList :VideoList);
}

struct Video {
    name @0 :Text;
    sizeBytes @1 :UInt64;
}

struct VideoList {
    videos @0 :List(Video);
}

