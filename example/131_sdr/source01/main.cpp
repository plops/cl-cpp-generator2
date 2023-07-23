#include <iostream>
#include <string>
#include <complex>
#include <vector>
#include <algorithm>
#include <functional>
#include <chrono>
#include <SoapySDR/Device.hpp>
#include <SoapySDR/Formats.hpp>
#include <SoapySDR/Errors.hpp>
#include "ArgException.h"
// ./my_project -b $((2**10))

struct Args {
    double sampleRate;
    double frequency;
    int bufferSize;
    int numberBuffers;
};
typedef struct Args Args;

struct Option {
    std::string longOpt;
    std::string shortOpt;
    std::string description;
    std::string defaultValue;
    std::function<void(const std::string &)> handler;
};
typedef struct Option Option;


void printHelp(const std::vector<Option> &options) {
    std::cout << "Usage: ./my_project [OPTIONS]" << std::endl;
    for (const auto &opt: options) {
        std::cout << " " << opt.longOpt << " or " << opt.shortOpt << ": " << opt.description << " default: "
                  << opt.defaultValue << std::endl;
    }
}


Args processArgs(const std::vector<std::string> &args) {
    auto result = Args({.sampleRate=1.00e+7, .frequency=4.330e+8, .bufferSize=512, .numberBuffers=100});
    auto
            options =
            std::vector<Option>(
                    {{.longOpt="--sampleRate", .shortOpt="-r", .description="Sample rate in Hz", .defaultValue=std::to_string(
                            result.sampleRate), .handler=([&result](const std::string &x) {
                        try {
                            result.sampleRate = std::stod(x);


                        } catch (const std::invalid_argument &) {
                            throw ArgException("Invalid value for --sampleRate");
                        }
                    })},
                     {.longOpt="--frequency", .shortOpt="-f", .description="Center frequency in Hz", .defaultValue=std::to_string(
                             result.frequency), .handler=([&result](const std::string &x) {
                         try {
                             result.frequency = std::stod(x);


                         } catch (const std::invalid_argument &) {
                             throw ArgException("Invalid value for --frequency");
                         }
                     })},
                     {.longOpt="--bufferSize", .shortOpt="-b", .description="Buffer Size (number of elements)", .defaultValue=std::to_string(
                             result.bufferSize), .handler=([&result](const std::string &x) {
                         try {
                             result.bufferSize = std::stoi(x);


                         } catch (const std::invalid_argument &) {
                             throw ArgException("Invalid value for --bufferSize");
                         }
                     })},
                     {.longOpt="--numberBuffers", .shortOpt="-n", .description="How many buffers to request", .defaultValue=std::to_string(
                             result.numberBuffers), .handler=([&result](const std::string &x) {
                         try {
                             result.numberBuffers = std::stoi(x);


                         } catch (const std::invalid_argument &) {
                             throw ArgException("Invalid value for --numberBuffers");
                         }
                     })}});
    auto it = args.begin();
    while (it != args.end()) {
        if (*it == "--help" || *it == "-h") {
            printHelp(options);
            exit(0);

        } else {
            // Find matching option

            auto optIt = std::find_if(options.begin(), options.end(), [&it](const Option &opt) {
                return *it == opt.longOpt || *it == opt.shortOpt;
            });
            if (optIt == options.end()) {
                throw ArgException("Unknown argument: " + *it);

            }
            // Move to next item, which should be the value for the option

            it++;
            if (it == args.end()) {
                throw ArgException("Expected value after " + *it);

            }
            optIt->handler(*it);
            // Move to next item, which should be the next option

            it++;


        }
    }

    return result;

}


int main(int argc, char **argv) {
    auto cmdlineArgs = std::vector<std::string>(argv + 1, argv + argc);
    try {
        auto parameters = processArgs(cmdlineArgs);
        auto results = SoapySDR::Device::enumerate();
        for (unsigned long i = 0; i < results.size(); i += 1) {
            std::cout << "found device" << " i='" << i << "' " << std::endl;
        }
        const auto &soapyDeviceArgs = results[0];
        auto sdr = SoapySDR::Device::make(soapyDeviceArgs);
        if (nullptr == sdr) {
            std::cout << "make failed" << std::endl;
            return -1;

        }
        auto direction = SOAPY_SDR_RX;
        auto channel = 0;
        auto antennas = sdr->listAntennas(direction, channel);
        for (const auto &antenna: antennas) {
            std::cout << "listAntennas" << " antenna='" << antenna << "' " << " direction='" << direction << "' "
                      << " channel='" << channel << "' " << std::endl;
        }

        auto gains = sdr->listGains(direction, channel);
        for (const auto &gain: gains) {
            std::cout << "listGains" << " gain='" << gain << "' " << " direction='" << direction << "' " << " channel='"
                      << channel << "' " << std::endl;
        }

        auto ranges = sdr->getFrequencyRange(direction, channel);
        for (const auto &range: ranges) {
            std::cout << "getFrequencyRange" << " range.minimum()='" << range.minimum() << "' " << " range.maximum()='"
                      << range.maximum() << "' " << " direction='" << direction << "' " << " channel='" << channel
                      << "' " << std::endl;
        }

        ([&]() {
            auto fullScale = 0.;
            std::cout << "" << " (sdr)->(getNativeStreamFormat(direction, channel, fullScale))='"
                      << sdr->getNativeStreamFormat(direction, channel, fullScale) << "' " << " fullScale='"
                      << fullScale << "' " << std::endl;

        })();
        sdr->setSampleRate(direction, channel, parameters.sampleRate);
        sdr->setFrequency(direction, channel, parameters.frequency);
        auto rx_stream = sdr->setupStream(direction, SOAPY_SDR_CF32);
        if (nullptr == rx_stream) {
            std::cout << "stream setup failed" << std::endl;
            SoapySDR::Device::unmake(sdr);
            return -1;

        }
        std::cout << "" << " (sdr)->(getStreamMTU(rx_stream))='" << sdr->getStreamMTU(rx_stream) << "' " << std::endl;
        ([&]() {
            auto flags = 0;
            auto timeNs = 0;
            auto numElems = 0;
            sdr->activateStream(rx_stream, flags, timeNs, numElems);

        })();

        // reusable buffer of rx samples

        auto numElems = parameters.bufferSize;
        auto numBytes = parameters.bufferSize * sizeof(std::complex<float>);
        auto buf = std::vector<std::complex<float>>(numElems);
        std::cout << "allocate SOAPY_SDR_CF32 buffer" << " numElems='" << numElems << "' " << " numBytes='" << numBytes
                  << "' " << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        auto expected_ms = (1.00e+3 * numElems) / parameters.sampleRate;
        auto expAvgElapsed_ms = expected_ms;
        auto alpha = 1.00e-2f;
        // choose alpha in [0,1]. for small values old measurements have less impact on the average
// .04 seems to average over 60 values in the history

        for (auto i = 0; i < parameters.numberBuffers; i += 1) {
            auto buffs = std::vector<void *>({buf.data()});
            auto flags = 0;
            auto time_ns = 0LL;
            auto ret = sdr->readStream(rx_stream, buffs.data(), numElems, flags, time_ns, 1.00e+5f);
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<double>(end - start);
            auto elapsed_ms = 1000 * elapsed.count();
            auto expected_ms = (1.00e+3 * ret) / parameters.sampleRate;
            expAvgElapsed_ms = alpha * elapsed_ms + (1.0 - alpha) * expAvgElapsed_ms;

            std::cout << "data block acquisition took" << " i='" << i << "' " << " elapsed_ms='" << elapsed_ms << "' "
                      << " expAvgElapsed_ms='" << expAvgElapsed_ms << "' " << " expected_ms='" << expected_ms << "' "
                      << std::endl;
            start = end;

            if (ret == SOAPY_SDR_TIMEOUT) {
                std::cout << "warning: timeout" << std::endl;

            }
            if (ret == SOAPY_SDR_OVERFLOW) {
                std::cout << "warning: overflow" << std::endl;

            }
            if (ret == SOAPY_SDR_UNDERFLOW) {
                std::cout << "warning: underflow" << std::endl;

            }
            if (ret < 0) {
                std::cout << "readStream failed" << " ret='" << ret << "' " << " SoapySDR::errToStr(ret)='"
                          << SoapySDR::errToStr(ret) << "' " << std::endl;
                sdr->deactivateStream(rx_stream, 0, 0);
                sdr->closeStream(rx_stream);
                SoapySDR::Device::unmake(sdr);

                exit(-1);

            }
            if (ret != numElems) {
                std::cout << "warning: readStream returned unexpected number of elements" << " ret='" << ret << "' "
                          << " flags='" << flags << "' " << " time_ns='" << time_ns << "' " << std::endl;

            }

        }


        sdr->deactivateStream(rx_stream, 0, 0);
        sdr->closeStream(rx_stream);
        SoapySDR::Device::unmake(sdr);


    } catch (const ArgException &e) {
        std::cout << "Error processing command line arguments" << " e.what()='" << e.what() << "' " << std::endl;
        return -1;

    }

    return 0;
}
 
