# capnp compile -oc++ video.capnp

@0xb1039f6a7c611b12;

interface Tube {

list @0 () -> (videos :Videos);

struct Video {
    name @0 :Text;
    size @1 :UInt64;
}

struct Videos {
    video @0 :List(Video);
}

}