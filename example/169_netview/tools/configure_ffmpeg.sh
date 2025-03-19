Usage: configure [options]
Options: [defaults in brackets after descriptions]

Help options:
  --help                   print this message
  --quiet                  Suppress showing informative output
  --list-decoders          show all available decoders
  --list-encoders          show all available encoders
  --list-hwaccels          show all available hardware accelerators
  --list-demuxers          show all available demuxers
  --list-muxers            show all available muxers
  --list-parsers           show all available parsers
  --list-protocols         show all available protocols
  --list-bsfs              show all available bitstream filters
  --list-indevs            show all available input devices
  --list-outdevs           show all available output devices
  --list-filters           show all available filters

Standard options:
  --logfile=FILE           log tests and output to FILE [ffbuild/config.log]
  --disable-logging        do not log configure debug information
  --fatal-warnings         fail if any configure warning is generated
  --prefix=PREFIX          install in PREFIX [/usr/local]
  --bindir=DIR             install binaries in DIR [PREFIX/bin]
  --datadir=DIR            install data files in DIR [PREFIX/share/ffmpeg]
  --docdir=DIR             install documentation in DIR [PREFIX/share/doc/ffmpeg]
  --libdir=DIR             install libs in DIR [PREFIX/lib]
  --shlibdir=DIR           install shared libs in DIR [LIBDIR]
  --incdir=DIR             install includes in DIR [PREFIX/include]
  --mandir=DIR             install man page in DIR [PREFIX/share/man]
  --pkgconfigdir=DIR       install pkg-config files in DIR [LIBDIR/pkgconfig]
  --enable-rpath           use rpath to allow installing libraries in paths
                           not part of the dynamic linker search path
                           use rpath when linking programs (USE WITH CARE)
  --install-name-dir=DIR   Darwin directory name for installed targets

Licensing options:
  --enable-gpl             allow use of GPL code, the resulting libs
                           and binaries will be under GPL [no]
  --enable-version3        upgrade (L)GPL to version 3 [no]
  --enable-nonfree         allow use of nonfree code, the resulting libs
                           and binaries will be unredistributable [no]

Configuration options:
  --disable-static         do not build static libraries [no]
  --enable-shared          build shared libraries [no]
  --enable-small           optimize for size instead of speed
  --disable-runtime-cpudetect disable detecting CPU capabilities at runtime (smaller binary)
  --enable-gray            enable full grayscale support (slower color)
  --disable-swscale-alpha  disable alpha channel support in swscale
  --disable-all            disable building components, libraries and programs
  --disable-autodetect     disable automatically detected external libraries [no]

Program options:
  --disable-programs       do not build command line programs
  --disable-ffmpeg         disable ffmpeg build
  --disable-ffplay         disable ffplay build
  --disable-ffprobe        disable ffprobe build

Documentation options:
  --disable-doc            do not build documentation
  --disable-htmlpages      do not build HTML documentation pages
  --disable-manpages       do not build man documentation pages
  --disable-podpages       do not build POD documentation pages
  --disable-txtpages       do not build text documentation pages

Component options:
  --disable-avdevice       disable libavdevice build
  --disable-avcodec        disable libavcodec build
  --disable-avformat       disable libavformat build
  --disable-swresample     disable libswresample build
  --disable-swscale        disable libswscale build
  --disable-postproc       disable libpostproc build
  --disable-avfilter       disable libavfilter build
  --disable-pthreads       disable pthreads [autodetect]
  --disable-w32threads     disable Win32 threads [autodetect]
  --disable-os2threads     disable OS/2 threads [autodetect]
  --disable-network        disable network support [no]
  --disable-dwt            disable DWT code
  --disable-error-resilience disable error resilience code
  --disable-lsp            disable LSP code
  --disable-faan           disable floating point AAN (I)DCT code
  --disable-pixelutils     disable pixel utils in libavutil

Individual component options:
  --disable-everything     disable all components listed below
  --disable-encoder=NAME   disable encoder NAME
  --enable-encoder=NAME    enable encoder NAME
  --disable-encoders       disable all encoders
  --disable-decoder=NAME   disable decoder NAME
  --enable-decoder=NAME    enable decoder NAME
  --disable-decoders       disable all decoders
  --disable-hwaccel=NAME   disable hwaccel NAME
  --enable-hwaccel=NAME    enable hwaccel NAME
  --disable-hwaccels       disable all hwaccels
  --disable-muxer=NAME     disable muxer NAME
  --enable-muxer=NAME      enable muxer NAME
  --disable-muxers         disable all muxers
  --disable-demuxer=NAME   disable demuxer NAME
  --enable-demuxer=NAME    enable demuxer NAME
  --disable-demuxers       disable all demuxers
  --enable-parser=NAME     enable parser NAME
  --disable-parser=NAME    disable parser NAME
  --disable-parsers        disable all parsers
  --enable-bsf=NAME        enable bitstream filter NAME
  --disable-bsf=NAME       disable bitstream filter NAME
  --disable-bsfs           disable all bitstream filters
  --enable-protocol=NAME   enable protocol NAME
  --disable-protocol=NAME  disable protocol NAME
  --disable-protocols      disable all protocols
  --enable-indev=NAME      enable input device NAME
  --disable-indev=NAME     disable input device NAME
  --disable-indevs         disable input devices
  --enable-outdev=NAME     enable output device NAME
  --disable-outdev=NAME    disable output device NAME
  --disable-outdevs        disable output devices
  --disable-devices        disable all devices
  --enable-filter=NAME     enable filter NAME
  --disable-filter=NAME    disable filter NAME
  --disable-filters        disable all filters

External library support:

  Using any of the following switches will allow FFmpeg to link to the
  corresponding external library. All the components depending on that library
  will become enabled, if all their other dependencies are met and they are not
  explicitly disabled. E.g. --enable-libopus will enable linking to
  libopus and allow the libopus encoder to be built, unless it is
  specifically disabled with --disable-encoder=libopus.

  Note that only the system libraries are auto-detected. All the other external
  libraries must be explicitly enabled.

  Also note that the following help text describes the purpose of the libraries
  themselves, not all their features will necessarily be usable by FFmpeg.

  --disable-alsa           disable ALSA support [autodetect]
  --disable-appkit         disable Apple AppKit framework [autodetect]
  --disable-avfoundation   disable Apple AVFoundation framework [autodetect]
  --enable-avisynth        enable reading of AviSynth script files [no]
  --disable-bzlib          disable bzlib [autodetect]
  --disable-coreimage      disable Apple CoreImage framework [autodetect]
  --enable-chromaprint     enable audio fingerprinting with chromaprint [no]
  --enable-frei0r          enable frei0r video filtering [no]
  --enable-gcrypt          enable gcrypt, needed for rtmp(t)e support
                           if openssl, librtmp or gmp is not used [no]
  --enable-gmp             enable gmp, needed for rtmp(t)e support
                           if openssl or librtmp is not used [no]
  --enable-gnutls          enable gnutls, needed for https support
                           if openssl, libtls or mbedtls is not used [no]
  --disable-iconv          disable iconv [autodetect]
  --enable-jni             enable JNI support [no]
  --enable-ladspa          enable LADSPA audio filtering [no]
  --enable-lcms2           enable ICC profile support via LittleCMS 2 [no]
  --enable-libaom          enable AV1 video encoding/decoding via libaom [no]
  --enable-libaribb24      enable ARIB text and caption decoding via libaribb24 [no]
  --enable-libaribcaption  enable ARIB text and caption decoding via libaribcaption [no]
  --enable-libass          enable libass subtitles rendering,
                           needed for subtitles and ass filter [no]
  --enable-libbluray       enable BluRay reading using libbluray [no]
  --enable-libbs2b         enable bs2b DSP library [no]
  --enable-libcaca         enable textual display using libcaca [no]
  --enable-libcelt         enable CELT decoding via libcelt [no]
  --enable-libcdio         enable audio CD grabbing with libcdio [no]
  --enable-libcodec2       enable codec2 en/decoding using libcodec2 [no]
  --enable-libdav1d        enable AV1 decoding via libdav1d [no]
  --enable-libdavs2        enable AVS2 decoding via libdavs2 [no]
  --enable-libdc1394       enable IIDC-1394 grabbing using libdc1394
                           and libraw1394 [no]
  --enable-libfdk-aac      enable AAC de/encoding via libfdk-aac [no]
  --enable-libflite        enable flite (voice synthesis) support via libflite [no]
  --enable-libfontconfig   enable libfontconfig, useful for drawtext filter [no]
  --enable-libfreetype     enable libfreetype, needed for drawtext filter [no]
  --enable-libfribidi      enable libfribidi, improves drawtext filter [no]
  --enable-libharfbuzz     enable libharfbuzz, needed for drawtext filter [no]
  --enable-libglslang      enable GLSL->SPIRV compilation via libglslang [no]
  --enable-libgme          enable Game Music Emu via libgme [no]
  --enable-libgsm          enable GSM de/encoding via libgsm [no]
  --enable-libiec61883     enable iec61883 via libiec61883 [no]
  --enable-libilbc         enable iLBC de/encoding via libilbc [no]
  --enable-libjack         enable JACK audio sound server [no]
  --enable-libjxl          enable JPEG XL de/encoding via libjxl [no]
  --enable-libklvanc       enable Kernel Labs VANC processing [no]
  --enable-libkvazaar      enable HEVC encoding via libkvazaar [no]
  --enable-liblensfun      enable lensfun lens correction [no]
  --enable-libmodplug      enable ModPlug via libmodplug [no]
  --enable-libmp3lame      enable MP3 encoding via libmp3lame [no]
  --enable-libopencore-amrnb enable AMR-NB de/encoding via libopencore-amrnb [no]
  --enable-libopencore-amrwb enable AMR-WB decoding via libopencore-amrwb [no]
  --enable-libopencv       enable video filtering via libopencv [no]
  --enable-libopenh264     enable H.264 encoding via OpenH264 [no]
  --enable-libopenjpeg     enable JPEG 2000 de/encoding via OpenJPEG [no]
  --enable-libopenmpt      enable decoding tracked files via libopenmpt [no]
  --enable-libopenvino     enable OpenVINO as a DNN module backend
                           for DNN based filters like dnn_processing [no]
  --enable-libopus         enable Opus de/encoding via libopus [no]
  --enable-libplacebo      enable libplacebo library [no]
  --enable-libpulse        enable Pulseaudio input via libpulse [no]
  --enable-librabbitmq     enable RabbitMQ library [no]
  --enable-librav1e        enable AV1 encoding via rav1e [no]
  --enable-librist         enable RIST via librist [no]
  --enable-librsvg         enable SVG rasterization via librsvg [no]
  --enable-librubberband   enable rubberband needed for rubberband filter [no]
  --enable-librtmp         enable RTMP[E] support via librtmp [no]
  --enable-libshaderc      enable GLSL->SPIRV compilation via libshaderc [no]
  --enable-libshine        enable fixed-point MP3 encoding via libshine [no]
  --enable-libsmbclient    enable Samba protocol via libsmbclient [no]
  --enable-libsnappy       enable Snappy compression, needed for hap encoding [no]
  --enable-libsoxr         enable Include libsoxr resampling [no]
  --enable-libspeex        enable Speex de/encoding via libspeex [no]
  --enable-libsrt          enable Haivision SRT protocol via libsrt [no]
  --enable-libssh          enable SFTP protocol via libssh [no]
  --enable-libsvtav1       enable AV1 encoding via SVT [no]
  --enable-libtensorflow   enable TensorFlow as a DNN module backend
                           for DNN based filters like sr [no]
  --enable-libtesseract    enable Tesseract, needed for ocr filter [no]
  --enable-libtheora       enable Theora encoding via libtheora [no]
  --enable-libtls          enable LibreSSL (via libtls), needed for https support
                           if openssl, gnutls or mbedtls is not used [no]
  --enable-libtwolame      enable MP2 encoding via libtwolame [no]
  --enable-libuavs3d       enable AVS3 decoding via libuavs3d [no]
  --enable-libv4l2         enable libv4l2/v4l-utils [no]
  --enable-libvidstab      enable video stabilization using vid.stab [no]
  --enable-libvmaf         enable vmaf filter via libvmaf [no]
  --enable-libvo-amrwbenc  enable AMR-WB encoding via libvo-amrwbenc [no]
  --enable-libvorbis       enable Vorbis en/decoding via libvorbis,
                           native implementation exists [no]
  --enable-libvpx          enable VP8 and VP9 de/encoding via libvpx [no]
  --enable-libwebp         enable WebP encoding via libwebp [no]
  --enable-libx264         enable H.264 encoding via x264 [no]
  --enable-libx265         enable HEVC encoding via x265 [no]
  --enable-libxavs         enable AVS encoding via xavs [no]
  --enable-libxavs2        enable AVS2 encoding via xavs2 [no]
  --enable-libxcb          enable X11 grabbing using XCB [autodetect]
  --enable-libxcb-shm      enable X11 grabbing shm communication [autodetect]
  --enable-libxcb-xfixes   enable X11 grabbing mouse rendering [autodetect]
  --enable-libxcb-shape    enable X11 grabbing shape rendering [autodetect]
  --enable-libxvid         enable Xvid encoding via xvidcore,
                           native MPEG-4/Xvid encoder exists [no]
  --enable-libxml2         enable XML parsing using the C library libxml2, needed
                           for dash and imf demuxing support [no]
  --enable-libzimg         enable z.lib, needed for zscale filter [no]
  --enable-libzmq          enable message passing via libzmq [no]
  --enable-libzvbi         enable teletext support via libzvbi [no]
  --enable-lv2             enable LV2 audio filtering [no]
  --disable-lzma           disable lzma [autodetect]
  --enable-decklink        enable Blackmagic DeckLink I/O support [no]
  --enable-mbedtls         enable mbedTLS, needed for https support
                           if openssl, gnutls or libtls is not used [no]
  --enable-mediacodec      enable Android MediaCodec support [no]
  --enable-mediafoundation enable encoding via MediaFoundation [auto]
  --disable-metal          disable Apple Metal framework [autodetect]
  --enable-libmysofa       enable libmysofa, needed for sofalizer filter [no]
  --enable-openal          enable OpenAL 1.1 capture support [no]
  --enable-opencl          enable OpenCL processing [no]
  --enable-opengl          enable OpenGL rendering [no]
  --enable-openssl         enable openssl, needed for https support
                           if gnutls, libtls or mbedtls is not used [no]
  --enable-pocketsphinx    enable PocketSphinx, needed for asr filter [no]
  --disable-sndio          disable sndio support [autodetect]
  --disable-schannel       disable SChannel SSP, needed for TLS support on
                           Windows if openssl and gnutls are not used [autodetect]
  --disable-sdl2           disable sdl2 [autodetect]
  --disable-securetransport disable Secure Transport, needed for TLS support
                           on OSX if openssl and gnutls are not used [autodetect]
  --enable-vapoursynth     enable VapourSynth demuxer [no]
  --disable-xlib           disable xlib [autodetect]
  --disable-zlib           disable zlib [autodetect]

  The following libraries provide various hardware acceleration features:
  --disable-amf            disable AMF video encoding code [autodetect]
  --disable-audiotoolbox   disable Apple AudioToolbox code [autodetect]
  --enable-cuda-nvcc       enable Nvidia CUDA compiler [no]
  --disable-cuda-llvm      disable CUDA compilation using clang [autodetect]
  --disable-cuvid          disable Nvidia CUVID support [autodetect]
  --disable-d3d11va        disable Microsoft Direct3D 11 video acceleration code [autodetect]
  --disable-dxva2          disable Microsoft DirectX 9 video acceleration code [autodetect]
  --disable-ffnvcodec      disable dynamically linked Nvidia code [autodetect]
  --enable-libdrm          enable DRM code (Linux) [no]
  --enable-libmfx          enable Intel MediaSDK (AKA Quick Sync Video) code via libmfx [no]
  --enable-libvpl          enable Intel oneVPL code via libvpl if libmfx is not used [no]
  --enable-libnpp          enable Nvidia Performance Primitives-based code [no]
  --enable-mmal            enable Broadcom Multi-Media Abstraction Layer (Raspberry Pi) via MMAL [no]
  --disable-nvdec          disable Nvidia video decoding acceleration (via hwaccel) [autodetect]
  --disable-nvenc          disable Nvidia video encoding code [autodetect]
  --enable-omx             enable OpenMAX IL code [no]
  --enable-omx-rpi         enable OpenMAX IL code for Raspberry Pi [no]
  --enable-rkmpp           enable Rockchip Media Process Platform code [no]
  --disable-v4l2-m2m       disable V4L2 mem2mem code [autodetect]
  --disable-vaapi          disable Video Acceleration API (mainly Unix/Intel) code [autodetect]
  --disable-vdpau          disable Nvidia Video Decode and Presentation API for Unix code [autodetect]
  --disable-videotoolbox   disable VideoToolbox code [autodetect]
  --disable-vulkan         disable Vulkan code [autodetect]

Toolchain options:
  --arch=ARCH              select architecture []
  --cpu=CPU                select the minimum required CPU (affects
                           instruction selection, may crash on older CPUs)
  --cross-prefix=PREFIX    use PREFIX for compilation tools []
  --progs-suffix=SUFFIX    program name suffix []
  --enable-cross-compile   assume a cross-compiler is used
  --sysroot=PATH           root of cross-build tree
  --sysinclude=PATH        location of cross-build system headers
  --target-os=OS           compiler targets OS []
  --target-exec=CMD        command to run executables on target
  --target-path=DIR        path to view of build directory on target
  --target-samples=DIR     path to samples directory on target
  --tempprefix=PATH        force fixed dir/prefix instead of mktemp for checks
  --toolchain=NAME         set tool defaults according to NAME
                           (gcc-asan, clang-asan, gcc-msan, clang-msan,
                           gcc-tsan, clang-tsan, gcc-usan, clang-usan,
                           valgrind-massif, valgrind-memcheck,
                           msvc, icl, gcov, llvm-cov, hardened)
  --nm=NM                  use nm tool NM [nm -g]
  --ar=AR                  use archive tool AR [ar]
  --as=AS                  use assembler AS []
  --ln_s=LN_S              use symbolic link tool LN_S [ln -s -f]
  --strip=STRIP            use strip tool STRIP [strip]
  --windres=WINDRES        use windows resource compiler WINDRES [windres]
  --x86asmexe=EXE          use nasm-compatible assembler EXE [nasm]
  --cc=CC                  use C compiler CC [gcc]
  --cxx=CXX                use C compiler CXX [g++]
  --objcc=OCC              use ObjC compiler OCC [gcc]
  --dep-cc=DEPCC           use dependency generator DEPCC [gcc]
  --nvcc=NVCC              use Nvidia CUDA compiler NVCC or clang []
  --ld=LD                  use linker LD []
  --metalcc=METALCC        use metal compiler METALCC [xcrun -sdk macosx metal]
  --metallib=METALLIB      use metal linker METALLIB [xcrun -sdk macosx metallib]
  --pkg-config=PKGCONFIG   use pkg-config tool PKGCONFIG [pkg-config]
  --pkg-config-flags=FLAGS pass additional flags to pkgconf []
  --ranlib=RANLIB          use ranlib RANLIB [ranlib]
  --doxygen=DOXYGEN        use DOXYGEN to generate API doc [doxygen]
  --host-cc=HOSTCC         use host C compiler HOSTCC
  --host-cflags=HCFLAGS    use HCFLAGS when compiling for host
  --host-cppflags=HCPPFLAGS use HCPPFLAGS when compiling for host
  --host-ld=HOSTLD         use host linker HOSTLD
  --host-ldflags=HLDFLAGS  use HLDFLAGS when linking for host
  --host-extralibs=HLIBS   use libs HLIBS when linking for host
  --host-os=OS             compiler host OS []
  --extra-cflags=ECFLAGS   add ECFLAGS to CFLAGS []
  --extra-cxxflags=ECFLAGS add ECFLAGS to CXXFLAGS []
  --extra-objcflags=FLAGS  add FLAGS to OBJCFLAGS []
  --extra-ldflags=ELDFLAGS add ELDFLAGS to LDFLAGS []
  --extra-ldexeflags=ELDFLAGS add ELDFLAGS to LDEXEFLAGS []
  --extra-ldsoflags=ELDFLAGS add ELDFLAGS to LDSOFLAGS []
  --extra-libs=ELIBS       add ELIBS []
  --extra-version=STRING   version string suffix []
  --optflags=OPTFLAGS      override optimization-related compiler flags
  --nvccflags=NVCCFLAGS    override nvcc flags []
  --build-suffix=SUFFIX    library name suffix []
  --enable-pic             build position-independent code
  --enable-thumb           compile for Thumb instruction set
  --enable-lto[=arg]       use link-time optimization
  --env="ENV=override"     override the environment variables

Advanced options (experts only):
  --malloc-prefix=PREFIX   prefix malloc and related names with PREFIX
  --custom-allocator=NAME  use a supported custom allocator
  --disable-symver         disable symbol versioning
  --enable-hardcoded-tables use hardcoded tables instead of runtime generation
  --disable-safe-bitstream-reader
                           disable buffer boundary checking in bitreaders
                           (faster, but may crash)
  --sws-max-filter-size=N  the max filter size swscale uses [256]

Optimization options (experts only):
  --disable-asm            disable all assembly optimizations
  --disable-altivec        disable AltiVec optimizations
  --disable-vsx            disable VSX optimizations
  --disable-power8         disable POWER8 optimizations
  --disable-amd3dnow       disable 3DNow! optimizations
  --disable-amd3dnowext    disable 3DNow! extended optimizations
  --disable-mmx            disable MMX optimizations
  --disable-mmxext         disable MMXEXT optimizations
  --disable-sse            disable SSE optimizations
  --disable-sse2           disable SSE2 optimizations
  --disable-sse3           disable SSE3 optimizations
  --disable-ssse3          disable SSSE3 optimizations
  --disable-sse4           disable SSE4 optimizations
  --disable-sse42          disable SSE4.2 optimizations
  --disable-avx            disable AVX optimizations
  --disable-xop            disable XOP optimizations
  --disable-fma3           disable FMA3 optimizations
  --disable-fma4           disable FMA4 optimizations
  --disable-avx2           disable AVX2 optimizations
  --disable-avx512         disable AVX-512 optimizations
  --disable-avx512icl      disable AVX-512ICL optimizations
  --disable-aesni          disable AESNI optimizations
  --disable-armv5te        disable armv5te optimizations
  --disable-armv6          disable armv6 optimizations
  --disable-armv6t2        disable armv6t2 optimizations
  --disable-vfp            disable VFP optimizations
  --disable-neon           disable NEON optimizations
  --disable-dotprod        disable DOTPROD optimizations
  --disable-i8mm           disable I8MM optimizations
  --disable-inline-asm     disable use of inline assembly
  --disable-x86asm         disable use of standalone x86 assembly
  --disable-mipsdsp        disable MIPS DSP ASE R1 optimizations
  --disable-mipsdspr2      disable MIPS DSP ASE R2 optimizations
  --disable-msa            disable MSA optimizations
  --disable-mipsfpu        disable floating point MIPS optimizations
  --disable-mmi            disable Loongson MMI optimizations
  --disable-lsx            disable Loongson LSX optimizations
  --disable-lasx           disable Loongson LASX optimizations
  --disable-rvv            disable RISC-V Vector optimizations
  --disable-fast-unaligned consider unaligned accesses slow

Developer options (useful when working on FFmpeg itself):
  --disable-debug          disable debugging symbols
  --enable-debug=LEVEL     set the debug level []
  --disable-optimizations  disable compiler optimizations
  --enable-extra-warnings  enable more compiler warnings
  --disable-stripping      disable stripping of executables and shared libraries
  --assert-level=level     0(default), 1 or 2, amount of assertion testing,
                           2 causes a slowdown at runtime.
  --enable-memory-poisoning fill heap uninitialized allocated space with arbitrary data
  --valgrind=VALGRIND      run "make fate" tests through valgrind to detect memory
                           leaks and errors, using the specified valgrind binary.
                           Cannot be combined with --target-exec
  --enable-ftrapv          Trap arithmetic overflows
  --samples=PATH           location of test samples for FATE, if not set use
                           $FATE_SAMPLES at make invocation time.
  --enable-neon-clobber-test check NEON registers for clobbering (should be
                           used only for debugging purposes)
  --enable-xmm-clobber-test check XMM registers for clobbering (Win64-only;
                           should be used only for debugging purposes)
  --enable-random          randomly enable/disable components
  --disable-random
  --enable-random=LIST     randomly enable/disable specific components or
  --disable-random=LIST    component groups. LIST is a comma-separated list
                           of NAME[:PROB] entries where NAME is a component
                           (group) and PROB the probability associated with
                           NAME (default 0.5).
  --random-seed=VALUE      seed value for --enable/disable-random
  --disable-valgrind-backtrace do not print a backtrace under Valgrind
                           (only applies to --disable-optimizations builds)
  --enable-ossfuzz         Enable building fuzzer tool
  --libfuzzer=PATH         path to libfuzzer
  --ignore-tests=TESTS     comma-separated list (without "fate-" prefix
                           in the name) of tests whose result is ignored
  --enable-linux-perf      enable Linux Performance Monitor API
  --enable-macos-kperf     enable macOS kperf (private) API
  --disable-large-tests    disable tests that use a large amount of memory
  --disable-ptx-compression don't compress CUDA PTX code even when possible

NOTE: Object files are built at the place where configure is launched.
aac                     gsm_ms                  pcm_u24le
aac_at                  gsm_ms_at               pcm_u32be
aac_fixed               h261                    pcm_u32le
aac_latm                h263                    pcm_u8
aasc                    h263_v4l2m2m            pcm_vidc
ac3                     h263i                   pcx
ac3_at                  h263p                   pdv
ac3_fixed               h264                    pfm
acelp_kelvin            h264_crystalhd          pgm
adpcm_4xm               h264_cuvid              pgmyuv
adpcm_adx               h264_mediacodec         pgssub
adpcm_afc               h264_mmal               pgx
adpcm_agm               h264_qsv                phm
adpcm_aica              h264_rkmpp              photocd
adpcm_argo              h264_v4l2m2m            pictor
adpcm_ct                hap                     pixlet
adpcm_dtk               hca                     pjs
adpcm_ea                hcom                    png
adpcm_ea_maxis_xa       hdr                     ppm
adpcm_ea_r1             hevc                    prores
adpcm_ea_r2             hevc_cuvid              prosumer
adpcm_ea_r3             hevc_mediacodec         psd
adpcm_ea_xas            hevc_qsv                ptx
adpcm_g722              hevc_rkmpp              qcelp
adpcm_g726              hevc_v4l2m2m            qdm2
adpcm_g726le            hnm4_video              qdm2_at
adpcm_ima_acorn         hq_hqa                  qdmc
adpcm_ima_alp           hqx                     qdmc_at
adpcm_ima_amv           huffyuv                 qdraw
adpcm_ima_apc           hymt                    qoi
adpcm_ima_apm           iac                     qpeg
adpcm_ima_cunning       idcin                   qtrle
adpcm_ima_dat4          idf                     r10k
adpcm_ima_dk3           iff_ilbm                r210
adpcm_ima_dk4           ilbc                    ra_144
adpcm_ima_ea_eacs       ilbc_at                 ra_288
adpcm_ima_ea_sead       imc                     ralf
adpcm_ima_iss           imm4                    rasc
adpcm_ima_moflex        imm5                    rawvideo
adpcm_ima_mtf           indeo2                  realtext
adpcm_ima_oki           indeo3                  rka
adpcm_ima_qt            indeo4                  rl2
adpcm_ima_qt_at         indeo5                  roq
adpcm_ima_rad           interplay_acm           roq_dpcm
adpcm_ima_smjpeg        interplay_dpcm          rpza
adpcm_ima_ssi           interplay_video         rscc
adpcm_ima_wav           ipu                     rtv1
adpcm_ima_ws            jacosub                 rv10
adpcm_ms                jpeg2000                rv20
adpcm_mtaf              jpegls                  rv30
adpcm_psx               jv                      rv40
adpcm_sbpro_2           kgv1                    s302m
adpcm_sbpro_3           kmvc                    sami
adpcm_sbpro_4           lagarith                sanm
adpcm_swf               libaom_av1              sbc
adpcm_thp               libaribb24              scpr
adpcm_thp_le            libaribcaption          screenpresso
adpcm_vima              libcelt                 sdx2_dpcm
adpcm_xa                libcodec2               sga
adpcm_xmd               libdav1d                sgi
adpcm_yamaha            libdavs2                sgirle
adpcm_zork              libfdk_aac              sheervideo
agm                     libgsm                  shorten
aic                     libgsm_ms               simbiosis_imx
alac                    libilbc                 sipr
alac_at                 libjxl                  siren
alias_pix               libopencore_amrnb       smackaud
als                     libopencore_amrwb       smacker
amr_nb_at               libopenh264             smc
amrnb                   libopus                 smvjpeg
amrwb                   librsvg                 snow
amv                     libspeex                sol_dpcm
anm                     libuavs3d               sonic
ansi                    libvorbis               sp5x
anull                   libvpx_vp8              speedhq
apac                    libvpx_vp9              speex
ape                     libzvbi_teletext        srgc
apng                    loco                    srt
aptx                    lscr                    ssa
aptx_hd                 m101                    stl
arbc                    mace3                   subrip
argo                    mace6                   subviewer
ass                     magicyuv                subviewer1
asv1                    mdec                    sunrast
asv2                    media100                svq1
atrac1                  metasound               svq3
atrac3                  microdvd                tak
atrac3al                mimic                   targa
atrac3p                 misc4                   targa_y216
atrac3pal               mjpeg                   tdsc
atrac9                  mjpeg_cuvid             text
aura                    mjpeg_qsv               theora
aura2                   mjpegb                  thp
av1                     mlp                     tiertexseqvideo
av1_cuvid               mmvideo                 tiff
av1_mediacodec          mobiclip                tmv
av1_qsv                 motionpixels            truehd
avrn                    movtext                 truemotion1
avrp                    mp1                     truemotion2
avs                     mp1_at                  truemotion2rt
avui                    mp1float                truespeech
ayuv                    mp2                     tscc
bethsoftvid             mp2_at                  tscc2
bfi                     mp2float                tta
bink                    mp3                     twinvq
binkaudio_dct           mp3_at                  txd
binkaudio_rdft          mp3adu                  ulti
bintext                 mp3adufloat             utvideo
bitpacked               mp3float                v210
bmp                     mp3on4                  v210x
bmv_audio               mp3on4float             v308
bmv_video               mpc7                    v408
bonk                    mpc8                    v410
brender_pix             mpeg1_cuvid             vb
c93                     mpeg1_v4l2m2m           vble
cavs                    mpeg1video              vbn
cbd2_dpcm               mpeg2_crystalhd         vc1
ccaption                mpeg2_cuvid             vc1_crystalhd
cdgraphics              mpeg2_mediacodec        vc1_cuvid
cdtoons                 mpeg2_mmal              vc1_mmal
cdxl                    mpeg2_qsv               vc1_qsv
cfhd                    mpeg2_v4l2m2m           vc1_v4l2m2m
cinepak                 mpeg2video              vc1image
clearvideo              mpeg4                   vcr1
cljr                    mpeg4_crystalhd         vmdaudio
cllc                    mpeg4_cuvid             vmdvideo
comfortnoise            mpeg4_mediacodec        vmix
cook                    mpeg4_mmal              vmnc
cpia                    mpeg4_v4l2m2m           vnull
cri                     mpegvideo               vorbis
cscd                    mpl2                    vp3
cyuv                    msa1                    vp4
dca                     mscc                    vp5
dds                     msmpeg4_crystalhd       vp6
derf_dpcm               msmpeg4v1               vp6a
dfa                     msmpeg4v2               vp6f
dfpwm                   msmpeg4v3               vp7
dirac                   msnsiren                vp8
dnxhd                   msp2                    vp8_cuvid
dolby_e                 msrle                   vp8_mediacodec
dpx                     mss1                    vp8_qsv
dsd_lsbf                mss2                    vp8_rkmpp
dsd_lsbf_planar         msvideo1                vp8_v4l2m2m
dsd_msbf                mszh                    vp9
dsd_msbf_planar         mts2                    vp9_cuvid
dsicinaudio             mv30                    vp9_mediacodec
dsicinvideo             mvc1                    vp9_qsv
dss_sp                  mvc2                    vp9_rkmpp
dst                     mvdv                    vp9_v4l2m2m
dvaudio                 mvha                    vplayer
dvbsub                  mwsc                    vqa
dvdsub                  mxpeg                   vqc
dvvideo                 nellymoser              wady_dpcm
dxa                     notchlc                 wavarc
dxtory                  nuv                     wavpack
dxv                     on2avc                  wbmp
eac3                    opus                    wcmv
eac3_at                 osq                     webp
eacmv                   paf_audio               webvtt
eamad                   paf_video               wmalossless
eatgq                   pam                     wmapro
eatgv                   pbm                     wmav1
eatqi                   pcm_alaw                wmav2
eightbps                pcm_alaw_at             wmavoice
eightsvx_exp            pcm_bluray              wmv1
eightsvx_fib            pcm_dvd                 wmv2
escape124               pcm_f16le               wmv3
escape130               pcm_f24le               wmv3_crystalhd
evrc                    pcm_f32be               wmv3image
exr                     pcm_f32le               wnv1
fastaudio               pcm_f64be               wrapped_avframe
ffv1                    pcm_f64le               ws_snd1
ffvhuff                 pcm_lxf                 xan_dpcm
ffwavesynth             pcm_mulaw               xan_wc3
fic                     pcm_mulaw_at            xan_wc4
fits                    pcm_s16be               xbin
flac                    pcm_s16be_planar        xbm
flashsv                 pcm_s16le               xface
flashsv2                pcm_s16le_planar        xl
flic                    pcm_s24be               xma1
flv                     pcm_s24daud             xma2
fmvc                    pcm_s24le               xpm
fourxm                  pcm_s24le_planar        xsub
fraps                   pcm_s32be               xwd
frwu                    pcm_s32le               y41p
ftr                     pcm_s32le_planar        ylc
g2m                     pcm_s64be               yop
g723_1                  pcm_s64le               yuv4
g729                    pcm_s8                  zero12v
gdv                     pcm_s8_planar           zerocodec
gem                     pcm_sga                 zlib
gif                     pcm_u16be               zmbv

# hw accels
gremlin_dpcm            pcm_u16le
gsm                     pcm_u24be
av1_d3d11va             hevc_vaapi              vc1_d3d11va
av1_d3d11va2            hevc_vdpau              vc1_d3d11va2
av1_dxva2               hevc_videotoolbox       vc1_dxva2
av1_nvdec               hevc_vulkan             vc1_nvdec
av1_vaapi               mjpeg_nvdec             vc1_vaapi
av1_vdpau               mjpeg_vaapi             vc1_vdpau
av1_vulkan              mpeg1_nvdec             vp8_nvdec
h263_vaapi              mpeg1_vdpau             vp8_vaapi
h263_videotoolbox       mpeg1_videotoolbox      vp9_d3d11va
h264_d3d11va            mpeg2_d3d11va           vp9_d3d11va2
h264_d3d11va2           mpeg2_d3d11va2          vp9_dxva2
h264_dxva2              mpeg2_dxva2             vp9_nvdec
h264_nvdec              mpeg2_nvdec             vp9_vaapi
h264_vaapi              mpeg2_vaapi             vp9_vdpau
h264_vdpau              mpeg2_vdpau             vp9_videotoolbox
h264_videotoolbox       mpeg2_videotoolbox      wmv3_d3d11va
h264_vulkan             mpeg4_nvdec             wmv3_d3d11va2
hevc_d3d11va            mpeg4_vaapi             wmv3_dxva2
hevc_d3d11va2           mpeg4_vdpau             wmv3_nvdec
hevc_dxva2              mpeg4_videotoolbox      wmv3_vaapi
hevc_nvdec              prores_videotoolbox     wmv3_vdpau

#demuxers
aa                      idf                     pcm_f64le
aac                     iff                     pcm_mulaw
aax                     ifv                     pcm_s16be
ac3                     ilbc                    pcm_s16le
ac4                     image2                  pcm_s24be
ace                     image2_alias_pix        pcm_s24le
acm                     image2_brender_pix      pcm_s32be
act                     image2pipe              pcm_s32le
adf                     image_bmp_pipe          pcm_s8
adp                     image_cri_pipe          pcm_u16be
ads                     image_dds_pipe          pcm_u16le
adx                     image_dpx_pipe          pcm_u24be
aea                     image_exr_pipe          pcm_u24le
afc                     image_gem_pipe          pcm_u32be
aiff                    image_gif_pipe          pcm_u32le
aix                     image_hdr_pipe          pcm_u8
alp                     image_j2k_pipe          pcm_vidc
amr                     image_jpeg_pipe         pdv
amrnb                   image_jpegls_pipe       pjs
amrwb                   image_jpegxl_pipe       pmp
anm                     image_pam_pipe          pp_bnk
apac                    image_pbm_pipe          pva
apc                     image_pcx_pipe          pvf
ape                     image_pfm_pipe          qcp
apm                     image_pgm_pipe          r3d
apng                    image_pgmyuv_pipe       rawvideo
aptx                    image_pgx_pipe          realtext
aptx_hd                 image_phm_pipe          redspark
aqtitle                 image_photocd_pipe      rka
argo_asf                image_pictor_pipe       rl2
argo_brp                image_png_pipe          rm
argo_cvg                image_ppm_pipe          roq
asf                     image_psd_pipe          rpl
asf_o                   image_qdraw_pipe        rsd
ass                     image_qoi_pipe          rso
ast                     image_sgi_pipe          rtp
au                      image_sunrast_pipe      rtsp
av1                     image_svg_pipe          s337m
avi                     image_tiff_pipe         sami
avisynth                image_vbn_pipe          sap
avr                     image_webp_pipe         sbc
avs                     image_xbm_pipe          sbg
avs2                    image_xpm_pipe          scc
avs3                    image_xwd_pipe          scd
bethsoftvid             imf                     sdns
bfi                     ingenient               sdp
bfstm                   ipmovie                 sdr2
bink                    ipu                     sds
binka                   ircam                   sdx
bintext                 iss                     segafilm
bit                     iv8                     ser
bitpacked               ivf                     sga
bmv                     ivr                     shorten
boa                     jacosub                 siff
bonk                    jpegxl_anim             simbiosis_imx
brstm                   jv                      sln
c93                     kux                     smacker
caf                     kvag                    smjpeg
cavsvideo               laf                     smush
cdg                     libgme                  sol
cdxl                    libmodplug              sox
cine                    libopenmpt              spdif
codec2                  live_flv                srt
codec2raw               lmlm4                   stl
concat                  loas                    str
dash                    lrc                     subviewer
data                    luodat                  subviewer1
daud                    lvf                     sup
dcstr                   lxf                     svag
derf                    m4v                     svs
dfa                     matroska                swf
dfpwm                   mca                     tak
dhav                    mcc                     tedcaptions
dirac                   mgsts                   thp
dnxhd                   microdvd                threedostr
dsf                     mjpeg                   tiertexseq
dsicin                  mjpeg_2000              tmv
dss                     mlp                     truehd
dts                     mlv                     tta
dtshd                   mm                      tty
dv                      mmf                     txd
dvbsub                  mods                    ty
dvbtxt                  moflex                  usm
dxa                     mov                     v210
ea                      mp3                     v210x
ea_cdata                mpc                     vag
eac3                    mpc8                    vapoursynth
epaf                    mpegps                  vc1
evc                     mpegts                  vc1t
ffmetadata              mpegtsraw               vividas
filmstrip               mpegvideo               vivo
fits                    mpjpeg                  vmd
flac                    mpl2                    vobsub
flic                    mpsub                   voc
flv                     msf                     vpk
fourxm                  msnwc_tcp               vplayer
frm                     msp                     vqf
fsb                     mtaf                    vvc
fwse                    mtv                     w64
g722                    musx                    wady
g723_1                  mv                      wav
g726                    mvi                     wavarc
g726le                  mxf                     wc3
g729                    mxg                     webm_dash_manifest
gdv                     nc                      webvtt
genh                    nistsphere              wsaud
gif                     nsp                     wsd
gsm                     nsv                     wsvqa
gxf                     nut                     wtv
h261                    nuv                     wv
h263                    obu                     wve
h264                    ogg                     xa
hca                     oma                     xbin
hcom                    osq                     xmd
hevc                    paf                     xmv
hls                     pcm_alaw                xvag
hnm                     pcm_f32be               xwma
ico                     pcm_f32le               yop
idcin                   pcm_f64be               yuv4mpegpipe

#parsers
aac                     dvdsub                  mpegaudio
aac_latm                evc                     mpegvideo
ac3                     flac                    opus
adx                     ftr                     png
amr                     g723_1                  pnm
av1                     g729                    qoi
avs2                    gif                     rv34
avs3                    gsm                     sbc
bmp                     h261                    sipr
cavsvideo               h263                    tak
cook                    h264                    vc1
cri                     hdr                     vorbis
dca                     hevc                    vp3
dirac                   ipu                     vp8
dnxhd                   jpeg2000                vp9
dolby_e                 jpegxl                  vvc
dpx                     misc4                   webp
dvaudio                 mjpeg                   xbm
dvbsub                  mlp                     xma
dvd_nav                 mpeg4video              xwd

# bsfs

aac_adtstoasc           h264_redundant_pps      pcm_rechunk
av1_frame_merge         hapqa_extract           pgs_frame_merge
av1_frame_split         hevc_metadata           prores_metadata
av1_metadata            hevc_mp4toannexb        remove_extradata
chomp                   imx_dump_header         setts
dca_core                media100_to_mjpegb      text2movsub
dts2pts                 mjpeg2jpeg              trace_headers
dump_extradata          mjpega_dump_header      truehd_core
dv_error_marker         mov2textsub             vp9_metadata
eac3_core               mp3_header_decompress   vp9_raw_reorder
evc_frame_merge         mpeg2_metadata          vp9_superframe
extract_extradata       mpeg4_unpack_bframes    vp9_superframe_split
filter_units            noise                   vvc_metadata
h264_metadata           null                    vvc_mp4toannexb
h264_mp4toannexb        opus_metadata
