cmake_minimum_required(VERSION 3.17)
project(cxxnet_client)

option(NO_LIBS "Pull dependencies directly into project as source files" OFF)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED yes)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)


if (NOT WIN32)
    add_compile_options(
            -Wall
            -Wextra
            -Wconversion
            -Wdouble-promotion
            -Wno-unused-parameter
            -Wno-unused-function
            -Wno-sign-conversion
            -Wvla
            -Wframe-larger-than=5000
            -Wstack-usage=10000
            #-Wshadow
            -Wformat=2
            -fno-common
            -fstack-usage
            -fvisibility=hidden
            -fno-strict-overflow
            -Wno-strict-overflow
    )

endif ()

set(SOURCE_FILES src/main.cpp
        src/VideoArchiveImpl.cpp
        src/VideoDecoder.cpp
        src/VideoDecoder.h
        src/Histogram.h
        src/DurationComputer.cpp
        src/DurationComputer.h
)

# AddressSanitizer
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if (ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif ()

# UndefinedBehaviorSanitizer
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
if (ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif ()

# ThreadSanitizer
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
if (ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif ()

# LeakSanitizer
option(ENABLE_LSAN "Enable LeakSanitizer" OFF)
if (ENABLE_LSAN)
    add_compile_options(-fsanitize=leak)
    add_link_options(-fsanitize=leak)
endif ()


# set compiler flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
            -Og -g -g3 -ggdb -gdwarf-4
    )
else ()
    #    set(CMAKE_UNITY_BUILD true)
    #    set(CMAKE_UNITY_BUILD_BATCH_SIZE 90)
    add_compile_options(
            -Ofast
            -ffunction-sections
            -fdata-sections
            -Wpadded
            -march=znver3
    )
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    add_link_options(
            #  -Wl,-gc-sections,--print-gc-sections,--print-memory-usage,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map
            -Wl,-gc-sections,--print-gc-sections
            #-Wl,-gc-sections
    )
endif ()

include_directories(${CMAKE_SOURCE_DIR}/src)

set(capn /home/martin/src/capnproto/c++/src/capnp)
set(kj /home/martin/src/capnproto/c++/src/kj)
set(capnp_sources_lite
        ${capn}/c++.capnp.c++
        ${capn}/blob.c++
        ${capn}/arena.c++
        ${capn}/layout.c++
        ${capn}/list.c++
        ${capn}/any.c++
        ${capn}/message.c++
        ${capn}/schema.capnp.c++
        ${capn}/stream.capnp.c++
        ${capn}/serialize.c++
        ${capn}/serialize-packed.c++
)
set(capnp_sources_heavy
        ${capn}/schema.c++
        ${capn}/schema-loader.c++
        ${capn}/dynamic.c++
        ${capn}/stringify.c++
)
set(capnp-rpc_sources
        ${capn}/serialize-async.c++
        ${capn}/capability.c++
        ${capn}/membrane.c++
        ${capn}/dynamic-capability.c++
        ${capn}/rpc.c++
        ${capn}/rpc.capnp.c++
        ${capn}/rpc-twoparty.c++
        ${capn}/rpc-twoparty.capnp.c++
        ${capn}/persistent.capnp.c++
        ${capn}/ez-rpc.c++
)
set(kj_sources_lite
        ${kj}/array.c++
        ${kj}/cidr.c++
        ${kj}/list.c++
        ${kj}/common.c++
        ${kj}/debug.c++
        ${kj}/exception.c++
        ${kj}/io.c++
        ${kj}/memory.c++
        ${kj}/mutex.c++
        ${kj}/string.c++
        ${kj}/source-location.c++
        ${kj}/hash.c++
        ${kj}/table.c++
        ${kj}/thread.c++
        ${kj}/main.c++
        ${kj}/arena.c++
        ${kj}/test-helpers.c++
        ${kj}/units.c++
        ${kj}/encoding.c++
)
set(kj_sources_heavy
        ${kj}/refcount.c++
        ${kj}/string-tree.c++
        ${kj}/time.c++
        ${kj}/filesystem.c++
        ${kj}/filesystem-disk-unix.c++
        #${kj}/filesystem-disk-win32.c++
        ${kj}/parse/char.c++
)
set(kj-async_sources
        ${kj}/async.c++
        ${kj}/async-unix.c++
        #${kj}/async-win32.c++
        #${kj}/async-io-win32.c++
        ${kj}/async-io.c++
        ${kj}/async-io-unix.c++
        ${kj}/timer.c++
)


# for i in libavutil libavformat libavcodec; do cd $i; find *|grep \\.o$|awk "{print \"\${"$i"}/\"\$1}" |sed s/\\.o/.c/g;cd ..;done
set(ffmpeg_path /home/martin/src/ffmpeg-6.1.2)
set(libavutil ${ffmpeg_path}/libavutil)
set(libavformat ${ffmpeg_path}/libavformat)
set(libavcodec ${ffmpeg_path}/libavcodec)
set(ffmpeg_sources
        ${libavutil}/adler32.c
        ${libavutil}/aes.c
        ${libavutil}/aes_ctr.c
        ${libavutil}/ambient_viewing_environment.c
        ${libavutil}/audio_fifo.c
        ${libavutil}/avsscanf.c
        ${libavutil}/avstring.c
        ${libavutil}/base64.c
        ${libavutil}/blowfish.c
        ${libavutil}/bprint.c
        ${libavutil}/buffer.c
        ${libavutil}/camellia.c
        ${libavutil}/cast5.c
        ${libavutil}/channel_layout.c
        ${libavutil}/cpu.c
        ${libavutil}/crc.c
        ${libavutil}/csp.c
        ${libavutil}/des.c
        ${libavutil}/detection_bbox.c
        ${libavutil}/dict.c
        ${libavutil}/display.c
        ${libavutil}/dovi_meta.c
        ${libavutil}/downmix_info.c
        ${libavutil}/encryption_info.c
        ${libavutil}/error.c
        ${libavutil}/eval.c
        ${libavutil}/executor.c
        ${libavutil}/fifo.c
        ${libavutil}/file.c
        ${libavutil}/file_open.c
        ${libavutil}/film_grain_params.c
        ${libavutil}/fixed_dsp.c
        ${libavutil}/float_dsp.c
        ${libavutil}/frame.c
        ${libavutil}/hash.c
        ${libavutil}/hdr_dynamic_metadata.c
        ${libavutil}/hdr_dynamic_vivid_metadata.c
        ${libavutil}/hmac.c
        ${libavutil}/hwcontext.c
        ${libavutil}/hwcontext_vaapi.c
        ${libavutil}/hwcontext_vulkan.c
        ${libavutil}/imgutils.c
        ${libavutil}/integer.c
        ${libavutil}/intmath.c
        ${libavutil}/lfg.c
        ${libavutil}/lls.c
        ${libavutil}/log.c
        ${libavutil}/log2_tab.c
        ${libavutil}/lzo.c
        ${libavutil}/mastering_display_metadata.c
        ${libavutil}/mathematics.c
        ${libavutil}/md5.c
        ${libavutil}/mem.c
        ${libavutil}/murmur3.c
        ${libavutil}/opt.c
        ${libavutil}/parseutils.c
        ${libavutil}/pixdesc.c
        ${libavutil}/pixelutils.c
        ${libavutil}/random_seed.c
        ${libavutil}/rational.c
        ${libavutil}/rc4.c
        ${libavutil}/reverse.c
        ${libavutil}/ripemd.c
        ${libavutil}/samplefmt.c
        ${libavutil}/sha.c
        ${libavutil}/sha512.c
        ${libavutil}/slicethread.c
        ${libavutil}/spherical.c
        ${libavutil}/stereo3d.c
        ${libavutil}/tea.c
        ${libavutil}/threadmessage.c
        ${libavutil}/time.c
        ${libavutil}/timecode.c
        ${libavutil}/tree.c
        ${libavutil}/twofish.c
        ${libavutil}/tx.c
        ${libavutil}/tx_double.c
        ${libavutil}/tx_float.c
        ${libavutil}/tx_int32.c
        ${libavutil}/utils.c
        ${libavutil}/uuid.c
        ${libavutil}/version.c
        ${libavutil}/video_enc_params.c
        ${libavutil}/video_hint.c
        ${libavutil}/vulkan.c
        ${libavutil}/xga_font_data.c
        ${libavutil}/xtea.c
        ${libavformat}/allformats.c
        ${libavformat}/av1dec.c
        ${libavformat}/avformat.c
        ${libavformat}/avio.c
        ${libavformat}/aviobuf.c
        ${libavformat}/demux.c
        ${libavformat}/demux_utils.c
        ${libavformat}/dovi_isom.c
        ${libavformat}/dump.c
        ${libavformat}/dv.c
        ${libavformat}/flac_picture.c
        ${libavformat}/format.c
        ${libavformat}/h264dec.c
        ${libavformat}/id3v1.c
        ${libavformat}/id3v2.c
        ${libavformat}/isom_tags.c
        ${libavformat}/matroska.c
        ${libavformat}/matroskadec.c
        ${libavformat}/metadata.c
        ${libavformat}/mux.c
        ${libavformat}/mux_utils.c
        ${libavformat}/oggparsevorbis.c
        ${libavformat}/options.c
        ${libavformat}/os_support.c
        ${libavformat}/protocols.c
        ${libavformat}/qtpalette.c
        ${libavformat}/rawdec.c
        ${libavformat}/replaygain.c
        ${libavformat}/riff.c
        ${libavformat}/riffdec.c
        ${libavformat}/rmsipr.c
        ${libavformat}/sdp.c
        ${libavformat}/seek.c
        ${libavformat}/url.c
        ${libavformat}/utils.c
        ${libavformat}/version.c
        ${libavformat}/vorbiscomment.c
        ${libavcodec}/ac3_parser.c
        ${libavcodec}/adts_parser.c
        ${libavcodec}/allcodecs.c
        ${libavcodec}/atsc_a53.c
        ${libavcodec}/av1_frame_merge_bsf.c
        ${libavcodec}/av1_frame_split_bsf.c
        ${libavcodec}/av1_metadata_bsf.c
        ${libavcodec}/av1_parse.c
        ${libavcodec}/av1_parser.c
        ${libavcodec}/av1dec.c
        ${libavcodec}/avcodec.c
        ${libavcodec}/avdct.c
        ${libavcodec}/avfft.c
        ${libavcodec}/avpacket.c
        ${libavcodec}/bitstream.c
        ${libavcodec}/bitstream_filters.c
        ${libavcodec}/bsf.c
        ${libavcodec}/cabac.c
        ${libavcodec}/cbs.c
        ${libavcodec}/cbs_av1.c
        ${libavcodec}/cbs_bsf.c
        ${libavcodec}/cbs_h2645.c
        ${libavcodec}/cbs_sei.c
        ${libavcodec}/cbs_vp9.c
        ${libavcodec}/codec_desc.c
        ${libavcodec}/codec_par.c
        ${libavcodec}/d3d11va.c
        ${libavcodec}/decode.c
        ${libavcodec}/dirac.c
        ${libavcodec}/dv_profile.c
        ${libavcodec}/encode.c
        ${libavcodec}/get_buffer.c
        ${libavcodec}/golomb.c
        ${libavcodec}/h2645_parse.c
        ${libavcodec}/h2645_sei.c
        ${libavcodec}/h2645_vui.c
        ${libavcodec}/h2645data.c
        ${libavcodec}/h264_cabac.c
        ${libavcodec}/h264_cavlc.c
        ${libavcodec}/h264_direct.c
        ${libavcodec}/h264_levels.c
        ${libavcodec}/h264_loopfilter.c
        ${libavcodec}/h264_mb.c
        ${libavcodec}/h264_metadata_bsf.c
        ${libavcodec}/h264_mp4toannexb_bsf.c
        ${libavcodec}/h264_parse.c
        ${libavcodec}/h264_parser.c
        ${libavcodec}/h264_picture.c
        ${libavcodec}/h264_ps.c
        ${libavcodec}/h264_redundant_pps_bsf.c
        ${libavcodec}/h264_refs.c
        ${libavcodec}/h264_sei.c
        ${libavcodec}/h264_slice.c
        ${libavcodec}/h264chroma.c
        ${libavcodec}/h264data.c
        ${libavcodec}/h264dec.c
        ${libavcodec}/h264dsp.c
        ${libavcodec}/h264idct.c
        ${libavcodec}/h264pred.c
        ${libavcodec}/h264qpel.c
        ${libavcodec}/h274.c
        ${libavcodec}/imgconvert.c
        ${libavcodec}/jni.c
        ${libavcodec}/mathtables.c
        ${libavcodec}/mediacodec.c
        ${libavcodec}/mpeg12framerate.c
        ${libavcodec}/mpeg4audio_sample_rates.c
        ${libavcodec}/options.c
        ${libavcodec}/parser.c
        ${libavcodec}/parsers.c
        ${libavcodec}/profiles.c
        ${libavcodec}/pthread.c
        ${libavcodec}/pthread_frame.c
        ${libavcodec}/pthread_slice.c
        ${libavcodec}/qsv_api.c
        ${libavcodec}/raw.c
        ${libavcodec}/refstruct.c
        ${libavcodec}/startcode.c
        ${libavcodec}/to_upper4.c
        ${libavcodec}/utils.c
        ${libavcodec}/vaapi_av1.c
        ${libavcodec}/vaapi_decode.c
        ${libavcodec}/vaapi_h264.c
        ${libavcodec}/vaapi_vp9.c
        ${libavcodec}/version.c
        ${libavcodec}/videodsp.c
        ${libavcodec}/vlc.c
        ${libavcodec}/vorbis_parser.c
        ${libavcodec}/vp9.c
        ${libavcodec}/vp9_metadata_bsf.c
        ${libavcodec}/vp9_parser.c
        ${libavcodec}/vp9_raw_reorder_bsf.c
        ${libavcodec}/vp9_superframe_bsf.c
        ${libavcodec}/vp9_superframe_split_bsf.c
        ${libavcodec}/vp9block.c
        ${libavcodec}/vp9data.c
        ${libavcodec}/vp9dsp.c
        ${libavcodec}/vp9dsp_10bpp.c
        ${libavcodec}/vp9dsp_12bpp.c
        ${libavcodec}/vp9dsp_8bpp.c
        ${libavcodec}/vp9lpf.c
        ${libavcodec}/vp9mvs.c
        ${libavcodec}/vp9prob.c
        ${libavcodec}/vp9recon.c
        ${libavcodec}/vpx_rac.c
        ${libavcodec}/vulkan.c
        ${libavcodec}/vulkan_video.c
        ${libavcodec}/xiph.c
)


file(GLOB_RECURSE AV_SOURCES "/home/martin/src/avcpp/src/*.cpp")

if (NO_LIBS)
    message("Building with direct avcpp sources (as opposed to linking with libavcpp)")
    set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
    set(CMAKE_PREFIX_PATH "/home/martin/vulkan")

#    find_package(FFmpeg
#            COMPONENTS
#            AVCODEC
#            AVFORMAT AVUTIL
#            AVFILTER SWSCALE SWRESAMPLE
#            REQUIRED)


    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads)
    find_package(CapnProto REQUIRED)
    capnp_generate_cpp(videoSources videoHeaders proto/video.capnp)
    add_executable(
            ${PROJECT_NAME}
            ${SOURCE_FILES}
            ${videoSources}
            ${capnp_sources_lite}
            ${capnp_sources_heavy}
            ${capnp-rpc_sources}
            ${kj_sources_lite}
            ${kj_sources_heavy}
            ${kj-async_sources}
            ${AV_SOURCES}

    )
#    target_link_directories(${PROJECT_NAME} PRIVATE /home/martin/ffmpeg/lib)

    # add_library(ffmpeg ${ffmpeg_sources})
    target_compile_definitions(
            ${PROJECT_NAME}
            PUBLIC
            __STDC_CONSTANT_MACROS
    )
    #    target_compile_definitions(
    #            ffmpeg
    #            PUBLIC
    #            FF_DISABLE_DEPRECATION_WARNINGS
    #            FF_ENABLE_DEPRECATION_WARNINGS
    #    )
    #    target_compile_options(
    #            ${PROJECT_NAME}
    #            PUBLIC
    #    )
    # media-video/ffmpeg-6.1.2 X dav1d encode gpl libdrm opus pulseaudio static-libs threads vaapi vulkan x264
    target_link_directories(${PROJECT_NAME} PRIVATE /home/martin/ffmpeg/lib)
    target_link_libraries(
            ${PROJECT_NAME}
            PRIVATE
            Threads::Threads
            avformat
            avdevice
#            avfilter
            avcodec
#            swscale
#            swresample
            avutil
#            x264
#            pulse
#            pulsecommon-17.0
#            sndfile
#            systemd
#            asyncns
#            FLAC
#            vorbis
#            vorbisenc
#            ogg
#            mpg123
#            mp3lame
#            cap
#            mvec
    )
    target_include_directories(
            ${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR} # for autogenerated capnp header
            /home/martin/vulkan/include
            /home/martin/vulkan/include/avcpp/
    )
    #    target_include_directories(
    #            ffmpeg
    #            PRIVATE
    #            ${ffmpeg_path}
    #            ${ffmpeg_path}/libavutil
    #    )
else ()
    set(CMAKE_PREFIX_PATH /home/martin/vulkan)
    find_package(CapnProto REQUIRED)

    capnp_generate_cpp(videoSources videoHeaders proto/video.capnp)
    # in ~/vulkan//lib64/cmake/avcpp/avcpp-config.cmake change this line
    # include("${CMAKE_CURRENT_LIST_DIR}/../../../lib/cmake/avcpp/avcpp-targets.cmake")
    find_package(avcpp REQUIRED)
    add_executable(
            ${PROJECT_NAME}
            ${SOURCE_FILES}
            ${videoSources}
    )
    target_link_libraries(
            ${PROJECT_NAME}
            PRIVATE
            #            FFmpeg::FFmpeg
            avcpp::avcpp
            CapnProto::capnp-rpc
    )
    target_include_directories(
            ${PROJECT_NAME}
            PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR} # for autogenerated capnp header
            /home/martin/vulkan/include
    )
endif ()






