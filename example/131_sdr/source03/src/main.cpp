#include <iostream>
#include <vector>
#include <algorithm>
#include <deque>
#include <filesystem>
#include <cstdlib>
#include <cmath>
#include "implot.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "GLFW/glfw3.h"
#include "GpsCACodeGenerator.h"
#include "MemoryMappedComplexShortFile.h"
#include "FFTWManager.h"
#include "SdrManager.h"
#include "ProcessMemoryInfo.h"

void glfw_error_callback(int err, const char *desc) {
    std::cout << "GLFW error:" << " err='" << err << "' " << " desc='" << desc << "' " << "\n" << std::flush;
}


bool isDaemonRunning() {
    auto exit_code = system("pidof sdrplay_apiService > /dev/null");
    auto shm_files_exist = true;
    if (!std::filesystem::exists("/dev/shm/Glbl\\sdrSrvRespSema")) {
        std::cout << "file /dev/shm/Glbl\\sdrSrvRespSema does not exist" << "\n" << std::flush;
        return false;

    }
    if (!std::filesystem::exists("/dev/shm/Glbl\\sdrSrvCmdSema")) {
        std::cout << "file /dev/shm/Glbl\\sdrSrvCmdSema does not exist" << "\n" << std::flush;
        return false;

    }
    if (!std::filesystem::exists("/dev/shm/Glbl\\sdrSrvComMtx")) {
        std::cout << "file /dev/shm/Glbl\\sdrSrvComMtx does not exist" << "\n" << std::flush;
        return false;

    }
    if (!std::filesystem::exists("/dev/shm/Glbl\\sdrSrvComShMem")) {
        std::cout << "file /dev/shm/Glbl\\sdrSrvComShMem does not exist" << "\n" << std::flush;
        return false;

    }
    return 0 == WEXITSTATUS(exit_code) && shm_files_exist;

}


void stopDaemon() {
    auto ret = std::system("ps axu|grep 'sdrplay_'|awk '{print $2}'|xargs kill -9");
    std::cout << "stop daemon" << " ret='" << ret << "' " << "\n" << std::flush;

}


void startDaemonIfNotRunning() {
    std::cout << "verify that sdr daemon is running" << "\n" << std::flush;
    if (!isDaemonRunning()) {
        std::cout << "sdrplay daemon is not running. start it" << "\n" << std::flush;
        auto daemon_exit = system("/usr/bin/sdrplay_apiService &");

        std::cout << "return value" << " daemon_exit='" << daemon_exit << "' " << "\n" << std::flush;
        sleep(1);

    }
}


auto DrawMemory = []() {
    static ProcessMemoryInfo memoryInfo;
    static std::deque<int> residentMemoryFifo;
    auto residentMemorySize = memoryInfo.getResidentMemorySize();
    residentMemoryFifo.push_back(residentMemorySize);
    if (size_t(2000) < residentMemoryFifo.size()) {
        residentMemoryFifo.pop_front();

    }

    auto helpx = std::vector<int>(residentMemoryFifo.size());
    auto helpy = std::vector<int>(residentMemoryFifo.size());
    for (size_t i = 0; i < residentMemoryFifo.size(); i += 1) {
        helpx[i] = static_cast<int>(i);


        helpy[i] = residentMemoryFifo[i];


    }

    ImPlot::SetNextAxisLimits(ImAxis_X1, 0, static_cast<int>(residentMemoryFifo.size()));
    ImPlot::SetNextAxisLimits(ImAxis_Y3, *std::min_element(helpy.begin(), helpy.end()),
                              *std::max_element(helpy.begin(), helpy.end()));

    if (ImPlot::BeginPlot("Resident Memory Usage")) {
        ImPlot::PlotLine("Resident Memory", helpx.data(), helpy.data(), static_cast<int>(helpy.size()));
        ImPlot::EndPlot();

    }


};

auto DrawSdrInfo = [](auto sdr) {
    ImGui::Text("sample-rate:");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "%f", sdr->get_sample_rate());

    ImGui::Text("bandwidth:");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "%f", sdr->get_bandwidth());

    ImGui::Text("frequency:");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "%f", sdr->get_frequency());

    ImGui::Text("gain-mode:");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "%s", sdr->get_gain_mode() ? "True" : "False");

};

auto SelectAutoGain = [](auto sdr) {
    static bool automaticGainMode = sdr->get_gain_mode();
    static bool old_automaticGainMode = automaticGainMode;
    ImGui::Checkbox("Automatic Gain Mode", &automaticGainMode);
    if (automaticGainMode != old_automaticGainMode) {
        sdr->set_gain_mode(automaticGainMode);
        old_automaticGainMode = automaticGainMode;


    }

    return automaticGainMode;
};

auto SelectGain = [](auto sdr) {
    static int gainIF = 20;
    if (ImGui::SliderInt("gainIF", &gainIF, 20, 59, "%02d", ImGuiSliderFlags_AlwaysClamp)) {
        sdr->setGainIF(gainIF);

    }

    static int gainRF = 0;
    if (ImGui::SliderInt("gainRF", &gainRF, 0, 3, "%02d", ImGuiSliderFlags_AlwaysClamp)) {
        sdr->setGainRF(gainRF);

    }

    return std::make_pair(gainIF, gainRF);
};

auto SelectStart = [](auto file) {
    static int start = 0;
    static int maxStart = static_cast<int>(file.size() / sizeof(std::complex<short>));
    if (file.ready()) {
        ImGui::SliderInt("Start", &start, 0, maxStart);

    }
    return std::make_pair(start, maxStart);

};

auto SelectWindowSize = []() {
    static size_t windowSizeIndex = 3;
    static size_t old_windowSizeIndex = 3;
    auto windowSizeItemsNum = std::vector<double>(
            {1024, 5456, 8192, 10000, 20000, 32768, 40000, 50000, 65536, 80000, 100000, 140000, 1048576});
    auto windowSize = static_cast<int>(windowSizeItemsNum[windowSizeIndex]);
    auto windowSizeItemsStr = std::vector<std::string>(
            {"1024", "5456", "8192", "10000", "20000", "32768", "40000", "50000", "65536", "80000", "100000", "140000",
             "1048576"});
    if (ImGui::BeginCombo("windowSize", windowSizeItemsStr[windowSizeIndex].c_str())) {
        for (size_t i = 0; i < windowSizeItemsStr.size(); i += 1) {
            auto is_selected = windowSizeIndex == i;
            if (ImGui::Selectable(windowSizeItemsStr[i].c_str(), is_selected)) {
                windowSizeIndex = i;
                windowSize = static_cast<int>(windowSizeItemsNum[i]);


            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();

            }

        }
        ImGui::EndCombo();

    }
    if (old_windowSizeIndex != windowSizeIndex) {
        old_windowSizeIndex = windowSizeIndex;


    }

    return windowSize;
};

auto SetBandwidth = [](auto sdr) {
    static size_t bandwidthIndex = 3;
    static size_t old_bandwidthIndex = 3;
    auto bandwidthItemsNum = std::vector<double>(
            {2.00e+5, 3.00e+5, 6.00e+5, 1.5360e+6, 5.00e+6, 6.00e+6, 7.00e+6, 8.00e+6});
    auto bandwidthItemsStr = std::vector<std::string>(
            {"200000.0d0", "300000.0d0", "600000.0d0", "1536000.0d0", "5000000.0d0", "6000000.0d0", "7000000.0d0",
             "8000000.0d0"});
    if (ImGui::BeginCombo("bandwidth", bandwidthItemsStr[bandwidthIndex].c_str())) {
        for (size_t bandwidthIter = 0; bandwidthIter < bandwidthItemsStr.size(); bandwidthIter += 1) {
            auto is_selected = bandwidthIndex == bandwidthIter;
            if (ImGui::Selectable(bandwidthItemsStr[bandwidthIter].c_str(), is_selected)) {
                bandwidthIndex = bandwidthIter;


            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();

            }

        }
        ImGui::EndCombo();

    }
    if (old_bandwidthIndex != bandwidthIndex) {
        sdr->set_bandwidth(bandwidthItemsNum[bandwidthIndex]);
        old_bandwidthIndex = bandwidthIndex;


    }

};

auto SelectRealtimeDisplay = [](auto file) {
    static bool realtimeDisplay = true;
    if (file.ready()) {
        ImGui::Checkbox("Realtime display", &realtimeDisplay);

    }

    return realtimeDisplay;
};

auto DrawWaveform = [](auto x, auto y1, auto y2) {
    auto windowSize = x.size();
    if (ImPlot::BeginPlot("Waveform (I/Q)")) {
        ImPlot::PlotLine("y1", x.data(), y1.data(), windowSize);
        ImPlot::PlotLine("y2", x.data(), y2.data(), windowSize);
        ImPlot::EndPlot();

    }

};

auto SelectLogScale = []() {
    static bool logScale = true;
    ImGui::Checkbox("Logarithmic Y-axis", &logScale);
    return logScale;

};

auto SelectSatellites = []() {
    static std::array<bool, 32> selectedSatellites = {false};
    if (ImGui::Begin("Satellite")) {
        for (auto i = 0; i < 32; i += 1) {
            ImGui::Checkbox(std::to_string(1 + i).c_str(), &selectedSatellites[i]);
            if (i != (32 - 1)) {
                ImGui::SameLine();

            }
        }
        ImGui::End();

    }


    return selectedSatellites;
};

auto DrawFourier = [](auto sampleRate, auto realtimeDisplay, auto windowSize, auto fftw, auto sdr, auto x, auto y1,
                      auto y2, auto zfifo, auto file, auto start, auto logScale, auto selectedSatellites) {
    auto in = std::vector<std::complex<double>>(windowSize);
    auto gps_freq = 1.575420e+9;
    static double lo_freq = 4.0920e+6;
    static double centerFrequency = realtimeDisplay ? sdr->get_frequency() : (gps_freq - lo_freq);
    auto windowSize2 = windowSize / 2;
    for (auto i = 0; i < windowSize; i += 1) {
        x[i] = centerFrequency + sampleRate * (static_cast<double>(i - windowSize2) / windowSize);


    }
    auto lo_phase = 0.;
    auto lo_rate = (lo_freq / sampleRate) * 4;
    if (realtimeDisplay) {
        for (auto i = 0; i < windowSize; i += 1) {
            auto zs = zfifo[i];
            auto zr = static_cast<double>(zs.real());
            auto zi = static_cast<double>(zs.imag());
            auto z = std::complex<double>(zr, zi);
            in[i] = z;


        }

    }

    auto out = fftw.fft(in, windowSize);
    if (logScale) {
        for (auto i = 0; i < windowSize; i += 1) {
            y1[i] = 10. * log10(std::abs(out[i]) / std::sqrt(windowSize));


        }
    } else {
        for (auto i = 0; i < windowSize; i += 1) {
            y1[i] = (std::abs(out[i]) / std::sqrt(windowSize));


        }
    }
    if (ImPlot::BeginPlot(logScale ? "FFT magnitude (dB)" : "FFT magnitude (linear)")) {
        {
            auto pointsPerPixel = static_cast<int>(static_cast<float>(x.size()) / (ImGui::GetContentRegionAvail().x));

            if (pointsPerPixel <= 1) {
                ImPlot::PlotLine("y1", x.data(), y1.data(), windowSize);

            } else {
                // Calculate upper bound for the number of pixels, preallocate memory for vectors, initialize counter.

                auto pixels = (x.size() + pointsPerPixel + -1) / pointsPerPixel;
                auto x_downsampled = std::vector<double>(pixels);
                auto y_max = std::vector<double>(pixels);
                auto count = 0;
                // Iterate over the data with steps of pointsPerPixel

                for (size_t i = 0; i < x.size(); i += pointsPerPixel) {
                    auto max_val = y1[i];
                    // Iterate over the points under the same pixel

                    for (size_t j = i + 1; j < i + pointsPerPixel && j < x.size(); j++) {
                        max_val = std::max(max_val, y1[j]);


                    }
                    y_max[count] = max_val;

                    x_downsampled[count] = x[i];

                    count++;

                }

                x_downsampled.resize(count);
                y_max.resize(count);
                ImPlot::PlotLine((std::string("y_max_y1_") + "").c_str(), x_downsampled.data(), y_max.data(),
                                 static_cast<int>(x_downsampled.size()));


            }
        }
        // handle user input. clicking into the graph allow tuning the sdr receiver to the specified frequency.

        auto xmouse = ImPlot::GetPlotMousePos().x;

        if (ImPlot::IsPlotHovered() && (ImGui::IsMouseClicked(2) || ImGui::IsMouseDragging(2))) {
            if (realtimeDisplay) {
                centerFrequency = xmouse;

                sdr->set_frequency(centerFrequency);

            } else {
                lo_freq = (xmouse - centerFrequency);


            }

        }

        ImPlot::EndPlot();

    }

    return out;

};

auto DrawCrossCorrelation = [](auto codes, auto out, auto selectedSatellites, auto windowSize, auto fftw,
                               auto sampleRate) {
    auto codesSize = codes[0].size();
    if (static_cast<size_t>(windowSize) == codesSize) {
        if (ImPlot::BeginPlot("Cross-Correlations with PRN sequences")) {
            auto x_corr = std::vector<double>(out.size());
            for (size_t i = 0; i < x_corr.size(); i += 1) {
                x_corr[i] = static_cast<double>(i);


            }

            auto maxSnrDop_vec = std::vector<int>(32);

            auto maxSnrIdx_vec = std::vector<int>(32);

            auto maxSnr_vec = std::vector<double>(32);

#pragma omp parallel for default(none) num_threads(12) shared(selectedSatellites, codes, out, fftw, windowSize, maxSnrDop_vec, maxSnrIdx_vec, maxSnr_vec, sampleRate)
            for (auto code_idx = 0; code_idx < 32; code_idx += 1) {
                auto maxSnrDop = 0;
                auto maxSnrIdx = 0;
                auto maxSnr = 0.;
                auto code = codes[code_idx];
                auto len = out.size();
                auto prod = std::vector<std::complex<double>>(len);
                auto dopStart = static_cast<int>((-5000 * static_cast<int>(len)) / sampleRate);
                auto dopEnd = static_cast<int>((5000 * static_cast<double>(len)) / sampleRate);
                for (int dop = dopStart; dop <= dopEnd; dop++) {
                    for (size_t i = 0; i < out.size(); i += 1) {
                        auto i1 = (i + -dop + len) % len;
                        prod[i] = std::conj(out[i]) * code[i1];


                    }
                    auto corr = fftw.ifft(prod, prod.size());
                    auto corrAbs2 = std::vector<double>(out.size());
                    auto sumPwr = 0.;
                    auto maxPwr = 0.;
                    auto maxPwrIdx = 0;
                    for (auto i = 0; i < static_cast<int>(out.size()); i += 1) {
                        auto v = std::abs(corr[i]);
                        auto pwr = (v * v) / windowSize;
                        if (maxPwr < pwr) {
                            maxPwr = pwr;
                            maxPwrIdx = i;


                        }
                        corrAbs2[i] = pwr;

                        sumPwr += pwr;

                    }
                    auto avgPwr = sumPwr / static_cast<double>(out.size());
                    auto snr = maxPwr / avgPwr;
                    if (maxSnr < snr) {
                        maxSnr = snr;
                        maxSnrDop = dop;
                        maxSnrIdx = maxPwrIdx;


                    }


                }

                maxSnrDop_vec[code_idx] = maxSnrDop;

                maxSnrIdx_vec[code_idx] = maxSnrIdx;

                maxSnr_vec[code_idx] = maxSnr;


            }
            for (auto pnr_idx = 0; pnr_idx < 32; pnr_idx += 1) {
                if (18. < maxSnr_vec[pnr_idx]) {
                    selectedSatellites[pnr_idx] = true;


                } else {
                    selectedSatellites[pnr_idx] = false;


                }
            }

            ImPlot::EndPlot();

        }
    } else {
        ImGui::Text("Don't perform correlation windowSize=%d codesSize=%ld", windowSize, codesSize);
    }

};


void DrawPlot(const MemoryMappedComplexShortFile &file, std::shared_ptr<SdrManager> sdr, FFTWManager &fftw,
              const std::vector<std::vector<std::complex<double>>> &codes, double sampleRate) {
    DrawMemory();
    DrawSdrInfo(sdr);
    auto automaticGainMode = SelectAutoGain(sdr);

    auto [gainIF, gainRf] = SelectGain(sdr);

    auto [start, maxStart] = SelectStart(file);

    auto windowSize = SelectWindowSize();

    SetBandwidth(sdr);
    auto realtimeDisplay = SelectRealtimeDisplay(file);

    auto x = std::vector<double>(windowSize);
    auto y1 = std::vector<double>(windowSize);
    auto y2 = std::vector<double>(windowSize);
    auto zfifo = std::vector<std::complex<short>>(windowSize);
    if (realtimeDisplay) {
        sdr->processFifo([&](const std::deque<std::complex<short>> &fifo) {
            auto n = windowSize;
            for (auto i = 0; i < n; i += 1) {
                auto z = fifo[i];
                zfifo[i] = z;

                x[i] = i;
                y1[i] = z.real();
                y2[i] = z.imag();


            }

        }, windowSize);
    } else {
        if (file.ready()) {
            if ((start + windowSize <= maxStart && 0 < windowSize)) {
                for (auto i = 0; i < windowSize; i += 1) {
                    auto z = file[(start + i)];
                    x[i] = i;
                    y1[i] = z.real();
                    y2[i] = z.imag();


                }
            } else {
                ImGui::Text("window outside of range stored in file start=%d windowSize=%d maxStart=%d", start,
                            windowSize, maxStart);
            }
        } else {
            ImGui::Text("file not ready");
        }
    }
    DrawWaveform(x, y1, y2);

    auto logScale = SelectLogScale();

    auto selectedSatellites = SelectSatellites();

    try {
        auto out = DrawFourier(sampleRate, realtimeDisplay, windowSize, fftw, sdr, x, y1, y2, zfifo, file, start,
                               logScale, selectedSatellites);


    } catch (const std::exception &e) {
        ImGui::Text("Error while processing signal: %s", e.what());
    }
}

auto initGL = []() {
    glfwSetErrorCallback(glfw_error_callback);
    if (0 == glfwInit()) {
        std::cout << "glfw init failed" << "\n" << std::flush;

    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    auto *window = glfwCreateWindow(800, 600, "imgui_dsp", nullptr, nullptr);
    if (nullptr == window) {
        std::cout << "failed to create glfw window" << "\n" << std::flush;

    }
    glfwMakeContextCurrent(window);
    std::cout << "enable vsync" << "\n" << std::flush;
    glfwSwapInterval(1);
    IMGUI_CHECKVERSION();
    std::cout << "create imgui context" << "\n" << std::flush;
    ImGui::CreateContext();
    ImPlot::CreateContext();
    auto &io = ImGui::GetIO();
    io.ConfigFlags = io.ConfigFlags | ImGuiConfigFlags_NavEnableKeyboard;


    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
    glClearColor(0, 0, 0, 1);


    return window;
};

auto initGps = [](double sampleRate, FFTWManager &fftw) {
    // based on Andrew Holme's code http://www.jks.com/gps/SearchFFT.cpp

    const auto caFrequency = 1.0230e+6;
    const auto caStep = caFrequency / sampleRate;
    const auto corrWindowTime_ms = 1.0;
    const auto corrLength = static_cast<int>((corrWindowTime_ms * sampleRate) / 1000);
    const auto caSequenceLength = 1023;
    std::cout << "prepare CA code chips" << " corrLength='" << corrLength << "' " << "\n" << std::flush;
    // the first fft takes always long (even if wisdom is present). as a workaround i just perform a very short fft. then it takes only a few milliseconds. subsequent large ffts are much faster

    auto mini = std::vector<std::complex<double>>(32);
    fftw.fft(mini, 32);

    auto normal = std::vector<std::complex<double>>(corrLength);
    fftw.fft(normal, corrLength);


    auto codes = std::vector<std::vector<std::complex<double>>>(32);
#pragma omp parallel for default(none) num_threads(12) shared(codes, caSequenceLength, corrLength, caStep, fftw)
    for (auto i = 0; i < 32; i += 1) {
        // chatGPT decided to start PRN index with 1. I don't like it but leave it for now.

        auto ca = GpsCACodeGenerator(i + 1);
        auto chips = ca.generate_sequence(caSequenceLength);
        auto code = std::vector<std::complex<double>>(corrLength);
        auto caPhase = 0.;
        auto chipIndex = 0;
        for (auto l = 0; l < corrLength; l += 1) {
            code[l] = chips[(chipIndex % caSequenceLength)] ? 1.0 : -1.0;


            caPhase += caStep;
            if (1 <= caPhase) {
                caPhase -= 1.0;
                chipIndex++;

            }
        }
        auto out = fftw.fft(code, corrLength);

        codes[i] = out;


    }
    return codes;


};

auto initSdr = [](auto sampleRate) {
    auto sdr = std::make_shared<SdrManager>(64512, 1100000, 50000, 5000);
    stopDaemon();
    startDaemonIfNotRunning();
    std::cout << "initialize sdr manager" << "\n" << std::flush;
    sdr->initialize();
    sdr->set_gain_mode(false);
    sdr->setGainIF(20);
    sdr->setGainRF(0);
    sdr->set_frequency(1.575420e+9);
    sdr->set_sample_rate(sampleRate);
    sdr->set_bandwidth(1.5360e+6);
    sdr->startCapture();

    return sdr;
};

auto initFile = [](auto fn) {
    auto file = MemoryMappedComplexShortFile(fn);
    if (file.ready()) {
        std::cout << "first element" << " fn='" << fn << "' " << " file[0].real()='" << file[0].real() << "' " << "\n"
                  << std::flush;

    }
    return file;

};


int main(int argc, char **argv) {
    try {
        auto fftw = FFTWManager(6);
        auto *window = initGL();
        auto sampleRate = 1.00e+7;
        auto codes = initGps(sampleRate, fftw);
        auto sdr = initSdr(sampleRate);
        auto file = initFile("/mnt5/capturedData_L1_rate10MHz_bw5MHz_iq_short.bin");

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            DrawPlot(file, sdr, fftw, codes, sampleRate);
            ImGui::Render();
            auto w = 0;
            auto h = 0;
            glfwGetFramebufferSize(window, &w, &h);
            glViewport(0, 0, w, h);
            glClear(GL_COLOR_BUFFER_BIT);

            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
        }
        sdr->stopCapture();
        sdr->close();

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();


    } catch (const std::runtime_error &e) {
        std::cout << "error 1422:" << " e.what()='" << e.what() << "' " << "\n" << std::flush;
        return -1;
    } catch (const std::exception &e) {
        std::cout << "error 1426:" << " e.what()='" << e.what() << "' " << "\n" << std::flush;
        return -1;
    }
    return 0;
}
 
