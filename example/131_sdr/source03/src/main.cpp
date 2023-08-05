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

void glfw_error_callback (int err, const char* desc)        {
        std::cout<<"GLFW erro:"<<" err='"<<err<<"' "<<" desc='"<<desc<<"' "<<std::endl<<std::flush;
}
 

bool isDaemonRunning ()        {
            auto exit_code  = system("pidof sdrplay_apiService > /dev/null"); 
    auto shm_files_exist  = true; 
    if ( !std::filesystem::exists("/dev/shm/Glbl\\sdrSrvRespSema") ) {
                        std::cout<<"file /dev/shm/Glbl\\sdrSrvRespSema does not exist"<<std::endl<<std::flush;
        return false;
 
} 
    if ( !std::filesystem::exists("/dev/shm/Glbl\\sdrSrvCmdSema") ) {
                        std::cout<<"file /dev/shm/Glbl\\sdrSrvCmdSema does not exist"<<std::endl<<std::flush;
        return false;
 
} 
    if ( !std::filesystem::exists("/dev/shm/Glbl\\sdrSrvComMtx") ) {
                        std::cout<<"file /dev/shm/Glbl\\sdrSrvComMtx does not exist"<<std::endl<<std::flush;
        return false;
 
} 
    if ( !std::filesystem::exists("/dev/shm/Glbl\\sdrSrvComShMem") ) {
                        std::cout<<"file /dev/shm/Glbl\\sdrSrvComShMem does not exist"<<std::endl<<std::flush;
        return false;
 
} 
    return 0==WEXITSTATUS(exit_code)&&shm_files_exist;
 
}
 

void stopDaemon ()        {
        std::system("ps axu|grep 'sdrplay_'|awk '{print $2}'|xargs kill -9");
}
 

void startDaemonIfNotRunning ()        {
        std::cout<<"verify that sdr daemon is running"<<std::endl<<std::flush;
        if ( !isDaemonRunning() ) {
                        std::cout<<"sdrplay daemon is not running. start it"<<std::endl<<std::flush;
                auto daemon_exit  = system("/usr/bin/sdrplay_apiService &"); 
 
        std::cout<<"return value"<<" daemon_exit='"<<daemon_exit<<"' "<<std::endl<<std::flush;
        sleep(1);
 
} 
}
 
 

void DrawPlot (const MemoryMappedComplexShortFile& file, SdrManager& sdr, FFTWManager& fftw, const std::vector<std::vector<std::complex<double>>> & codes)        {
            static ProcessMemoryInfo memoryInfo ; 
    static std::deque<int> residentMemoryFifo ; 
        auto residentMemorySize  = memoryInfo.getResidentMemorySize(); 
    auto sampleIndex  = residentMemoryFifo.size(); 
    residentMemoryFifo.push_back(residentMemorySize);
    if ( 2000<residentMemoryFifo.size() ) {
                        residentMemoryFifo.pop_front();
 
} 
 
            auto helpx  = std::vector<int>(residentMemoryFifo.size()); 
    auto helpy  = std::vector<int>(residentMemoryFifo.size()); 
    for ( auto i = 0;i<residentMemoryFifo.size();i+=1 ) {
                        helpx[i]=i;


                        helpy[i]=residentMemoryFifo[i];


} 
 
        ImPlot::SetNextAxisLimits(ImAxis_X1, 0, residentMemoryFifo.size());
    ImPlot::SetNextAxisLimits(ImAxis_Y3, *std::min_element(helpy.begin(), helpy.end()), *std::max_element(helpy.begin(), helpy.end()));
 
    if ( ImPlot::BeginPlot("Resident Memory Usage", "Sample", "Size (kB)") ) {
                        ImPlot::PlotLine("Resident Memory", helpx.data(), helpy.data(), helpy.size());
        ImPlot::EndPlot();
 
} 
 
 
            ImGui::Text("sample-rate:");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "%f", sdr.get_sample_rate());
 
            ImGui::Text("bandwidth:");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "%f", sdr.get_bandwidth());
 
            ImGui::Text("frequency:");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "%f", sdr.get_frequency());
 
            ImGui::Text("gain-mode:");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "%s", sdr.get_gain_mode() ? "True" : "False");
 
            static bool automaticGainMode  = sdr.get_gain_mode(); 
    static bool old_automaticGainMode  = automaticGainMode; 
    ImGui::Checkbox("Automatic Gain Mode", &automaticGainMode);
    if ( automaticGainMode!=old_automaticGainMode ) {
                        sdr.set_gain_mode(automaticGainMode);
                old_automaticGainMode=automaticGainMode;


 
} 
 
            static int gainIF  = 20; 
    if ( ImGui::SliderInt("gainIF", &gainIF, 20, 59, "%02d", ImGuiSliderFlags_AlwaysClamp) ) {
                        sdr.setGainIF(gainIF);
 
} 
 
            static int gainRF  = 0; 
    if ( ImGui::SliderInt("gainRF", &gainRF, 0, 3, "%02d", ImGuiSliderFlags_AlwaysClamp) ) {
                        sdr.setGainRF(gainRF);
 
} 
 
            static int start  = 0; 
    auto maxStart  = static_cast<int>(file.size()/sizeof(std::complex<short>)); 
    if ( file.ready() ) {
                        ImGui::SliderInt("Start", &start, 0, maxStart);
 
} 
 
            static int windowSizeIndex  = 10; 
    static int old_windowSizeIndex  = 10; 
    auto windowSizeItemsNum  = std::vector<double>({1024, 5456, 8192, 10000, 20000, 32768, 40000, 50000, 65536, 80000, 100000, 140000, 1048576}); 
    auto windowSize  = static_cast<int>(windowSizeItemsNum[windowSizeIndex]); 
    auto windowSizeItemsStr  = std::vector<std::string>({"1024", "5456", "8192", "10000", "20000", "32768", "40000", "50000", "65536", "80000", "100000", "140000", "1048576"}); 
    if ( ImGui::BeginCombo("windowSize", windowSizeItemsStr[windowSizeIndex].c_str()) ) {
                        for ( auto i = 0;i<windowSizeItemsStr.size();i+=1 ) {
                                    auto is_selected  = windowSizeIndex==i; 
            if ( ImGui::Selectable(windowSizeItemsStr[i].c_str(), is_selected) ) {
                                                                windowSizeIndex=i;
                windowSize=static_cast<int>(windowSizeItemsNum[i]);


 
} 
            if ( is_selected ) {
                                                ImGui::SetItemDefaultFocus();
 
} 
 
} 
        ImGui::EndCombo();
 
} 
    if ( old_windowSizeIndex!=windowSizeIndex ) {
                                old_windowSizeIndex=windowSizeIndex;


 
} 
 
            static int bandwidthIndex  = 3; 
    static int old_bandwidthIndex  = 3; 
    auto bandwidthItemsNum  = std::vector<double>({2.00e+5, 3.00e+5, 6.00e+5, 1.5360e+6, 5.00e+6, 6.00e+6, 7.00e+6, 8.00e+6}); 
    auto bandwidthValue  = bandwidthItemsNum[bandwidthIndex]; 
    auto bandwidthItemsStr  = std::vector<std::string>({"200000.0d0", "300000.0d0", "600000.0d0", "1536000.0d0", "5000000.0d0", "6000000.0d0", "7000000.0d0", "8000000.0d0"}); 
    if ( ImGui::BeginCombo("bandwidth", bandwidthItemsStr[bandwidthIndex].c_str()) ) {
                        for ( auto i = 0;i<bandwidthItemsStr.size();i+=1 ) {
                                    auto is_selected  = bandwidthIndex==i; 
            if ( ImGui::Selectable(bandwidthItemsStr[i].c_str(), is_selected) ) {
                                                                bandwidthIndex=i;
                bandwidthValue=bandwidthItemsNum[i];


 
} 
            if ( is_selected ) {
                                                ImGui::SetItemDefaultFocus();
 
} 
 
} 
        ImGui::EndCombo();
 
} 
    if ( old_bandwidthIndex!=bandwidthIndex ) {
                        sdr.set_bandwidth(bandwidthItemsNum[bandwidthIndex]);
                old_bandwidthIndex=bandwidthIndex;


 
} 
 
        if ( !file.ready()||(start+windowSize<=maxStart&&0<windowSize) ) {
                                auto x  = std::vector<double>(windowSize); 
        auto y1  = std::vector<double>(windowSize); 
        auto y2  = std::vector<double>(windowSize); 
        auto zfifo  = std::vector<std::complex<short>>(windowSize); 
                static bool realtimeDisplay  = true; 
        if ( file.ready() ) {
                                    ImGui::Checkbox("Realtime display", &realtimeDisplay);
 
} 
 
        if ( realtimeDisplay ) {
                                    sdr.processFifo([&] (const std::deque<std::complex<short>> & fifo){
                                                auto n  = windowSize; 
                for ( auto i = 0;i<n;i+=1 ) {
                                                            auto z  = fifo[i]; 
                                        zfifo[i]=z;

                                        x[i]=i;
                    y1[i]=z.real();
                    y2[i]=z.imag();


 
} 
 
}, windowSize);
 
} else {
                        for ( auto i = 0;i<windowSize;i+=1 ) {
                                                auto z  = file[(start+i)]; 
                                x[i]=i;
                y1[i]=z.real();
                y2[i]=z.imag();


 
} 
} 
                if ( ImPlot::BeginPlot("Waveform (I/Q)") ) {
                                    ImPlot::PlotLine("y1", x.data(), y1.data(), windowSize);
            ImPlot::PlotLine("y2", x.data(), y2.data(), windowSize);
            ImPlot::EndPlot();
 
} 
 
                static bool logScale  = true; 
        ImGui::Checkbox("Logarithmic Y-axis", &logScale);
        try {
                                    auto in  = std::vector<std::complex<double>>(windowSize); 
            auto nyquist  = windowSize/2.0    ; 
            auto sampleRate  = realtimeDisplay ? 1.00e+7 : 5.4560e+6; 
            auto gps_freq  = 1.575420e+9; 
            static double lo_freq  = 4.0920e+6; 
            static double centerFrequency  = realtimeDisplay ? sdr.get_frequency() : (gps_freq-lo_freq); 
            for ( auto i = 0;i<windowSize;i+=1 ) {
                                                x[i]=centerFrequency+sampleRate*(static_cast<double>(i-(windowSize/2))/windowSize);


} 
                        auto lo_phase  = 0.    ; 
            auto lo_rate  = (lo_freq/sampleRate)*4; 
            if ( realtimeDisplay ) {
                                                for ( auto i = 0;i<windowSize;i+=1 ) {
                                                            auto zs  = zfifo[i]; 
                    auto zr  = static_cast<double>(zs.real()); 
                    auto zi  = static_cast<double>(zs.imag()); 
                    auto z  = std::complex<double>(zr, zi); 
                                        in[i]=z;


 
} 
 
} else {
                                                for ( auto i = 0;i<windowSize;i+=1 ) {
                                                            auto zs  = file[(start+i)]; 
                    auto zr  = static_cast<double>(zs.real()); 
                    auto zi  = static_cast<double>(zs.imag()); 
                    auto z  = std::complex<double>(zr, zi); 
                                        const auto  lo_sin  = std::array<int,4>({1, 1, 0, 0}); 
                    const auto  lo_cos  = std::array<int,4>({1, 0, 0, 1}); 
                                        auto re  = zs.real()^lo_sin[static_cast<int>(lo_phase)] ?  -1 : 1; 
                    auto im  = zs.real()^lo_cos[static_cast<int>(lo_phase)] ?  -1 : 1; 
                                        in[i]=std::complex<double>(re, im);


 
                    lo_phase+=lo_rate;
                    if ( 4<=lo_phase ) {
                                                                        lo_phase-=4;
 
} 
 
 
} 
 
} 
 
                        auto out  = fftw.fft(in, windowSize); 
            if ( logScale ) {
                                for ( auto i = 0;i<windowSize;i+=1 ) {
                                                            y1[i]=10.    *log10(std::abs(out[i])/std::sqrt(windowSize));


} 
} else {
                                for ( auto i = 0;i<windowSize;i+=1 ) {
                                                            y1[i]=(std::abs(out[i])/std::sqrt(windowSize));


} 
} 
                                    // If there are more points than pixels on the screen, then I want to combine all the points under one pixel into three curves: the maximum, the mean and the minimum.
 
                                    static std::array<bool,32> selectedSatellites  = {false}; 
            if ( ImGui::Begin("Satellite") ) {
                                                for ( auto i = 0;i<32;i+=1 ) {
                                        ImGui::Checkbox(std::to_string(1+i).c_str(), &selectedSatellites[i]);
                                        if ( i!=(32-1) ) {
                                                                        ImGui::SameLine();
 
} 
} 
                ImGui::End();
 
} 
 
 
            if ( ImPlot::BeginPlot(logScale ? "FFT magnitude (dB)" : "FFT magnitude (linear)") ) {
                                                {
                                                            auto pointsPerPixel  = static_cast<int>((x.size())/(ImGui::GetContentRegionAvail().x)); 
 
                                        if ( pointsPerPixel<=1 ) {
                                                                        ImPlot::PlotLine("y1", x.data(), y1.data(), windowSize);
 
} else {
                                                                        // Calculate upper bound for the number of pixels, preallocate memory for vectors, initialize counter.
 
                                                auto pixels  = (x.size()+pointsPerPixel+ -1)/pointsPerPixel; 
                        auto x_downsampled  = std::vector<double>(pixels); 
                                                                        auto y_max  = std::vector<double>(pixels); 
                        auto count  = 0; 
                        // Iterate over the data with steps of pointsPerPixel
 
                        for ( int i=0;i<x.size();i+=pointsPerPixel ) {
                                                                                    auto max_val  = y1[i]; 
                            // Iterate over the points under the same pixel
 
                            for ( int j=i+1;j<i+pointsPerPixel&&j<x.size();j++ ) {
                                                                                                max_val=std::max(max_val, y1[j]);


} 
                                                        y_max[count]=max_val;

                                                        x_downsampled[count]=x[i];

                            count++;
 
} 
 
                        x_downsampled.resize(count);
                        y_max.resize(count);
                        ImPlot::PlotLine((std::string("y_max_y1_")+"").c_str(), x_downsampled.data(), y_max.data(), x_downsampled.size());
 
 
 
} 
} 
                                // handle user input. clicking into the graph allow tuning the sdr receiver to the specified frequency.
 
                                auto xmouse  = ImPlot::GetPlotMousePos().x; 
 
                if ( ImPlot::IsPlotHovered()&&(ImGui::IsMouseClicked(2)||ImGui::IsMouseDragging(2)) ) {
                                                            if ( realtimeDisplay ) {
                                                                                                centerFrequency=xmouse;

                        sdr.set_frequency(centerFrequency);
 
} else {
                                                                                                lo_freq=(xmouse-centerFrequency);


 
} 
 
} 
 
                ImPlot::EndPlot();
                if ( !realtimeDisplay ) {
                                                            ImGui::Text("lo_freq: %6.10f MHz", lo_freq*10.00e-7);
 
} 
                ImGui::Text("xmouse: %6.10f GHz", xmouse*1.00e-9);
                ImGui::Text("gps_freq-xmouse: %6.10f MHz", (gps_freq-xmouse)*10.00e-7);
                ImGui::Text("centerFrequency-xmouse: %6.10f MHz", (centerFrequency-xmouse)*10.00e-7);
                ImGui::Text("centerFrequency-gps_freq: %8.0f kHz", (centerFrequency-gps_freq)*1.00e-3);
 
} 
 
                        auto codesSize  = codes[0].size(); 
            if ( windowSize==codesSize ) {
                                if ( ImPlot::BeginPlot("Cross-Correlations with PRN sequences") ) {
                                                                                auto x_corr  = std::vector<double>(out.size()); 
                    for ( auto i = 0;i<x_corr.size();i+=1 ) {
                                                                        x_corr[i]=1.0    *i;


} 
 
                    for ( auto code_idx = 0;code_idx<32;code_idx+=1 ) {
                                                if ( selectedSatellites[code_idx] ) {
                                                                                                                auto code  = codes[code_idx]; 
                            auto len  = out.size(); 
                            auto prod  = std::vector<std::complex<double>>(len); 
                            auto dopStart  = static_cast<int>(( -5000*static_cast<int>(len))/sampleRate); 
                            auto dopEnd  = static_cast<int>((5000*len)/sampleRate); 
                            for ( int dop=dopStart;dop<=dopEnd;dop++ ) {
                                                                                                for ( auto i = 0;i<out.size();i+=1 ) {
                                                                                                            auto i1  = (i+-dop+len)%len; 
                                                                        prod[i]=std::conj(out[i])*code[i1];


 
} 
                                                                auto corr  = fftw.ifft(prod, prod.size()); 
                                auto corrAbs2  = std::vector<double>(out.size()); 
                                for ( auto i = 0;i<out.size();i+=1 ) {
                                                                                                            auto v  = std::abs(corr[i]); 
                                                                        corrAbs2[i]=((v*v)/windowSize);


 
} 
 
                                {
                                                                                                            auto pointsPerPixel  = static_cast<int>((x_corr.size())/100); 
 
                                                                        if ( pointsPerPixel<=1 ) {
                                                                                                                        ImPlot::PlotLine("corrAbs2", x_corr.data(), corrAbs2.data(), windowSize);
 
} else {
                                                                                                                        // Calculate upper bound for the number of pixels, preallocate memory for vectors, initialize counter.
 
                                                                                auto pixels  = (x_corr.size()+pointsPerPixel+ -1)/pointsPerPixel; 
                                        auto x_downsampled  = std::vector<double>(pixels); 
                                                                                                                        auto y_max  = std::vector<double>(pixels); 
                                        auto count  = 0; 
                                        // Iterate over the data with steps of pointsPerPixel
 
                                        for ( int i=0;i<x_corr.size();i+=pointsPerPixel ) {
                                                                                                                                    auto max_val  = corrAbs2[i]; 
                                            // Iterate over the points under the same pixel
 
                                            for ( int j=i+1;j<i+pointsPerPixel&&j<x_corr.size();j++ ) {
                                                                                                                                                max_val=std::max(max_val, corrAbs2[j]);


} 
                                                                                        y_max[count]=max_val;

                                                                                        x_downsampled[count]=x_corr[i];

                                            count++;
 
} 
 
                                        x_downsampled.resize(count);
                                        y_max.resize(count);
                                        ImPlot::PlotLine((std::string("y_max_corrAbs2_")+std::to_string(code_idx)+"_"+std::to_string(dop)).c_str(), x_downsampled.data(), y_max.data(), x_downsampled.size());
 
 
 
} 
} 
 
} 
 
 
} 
} 
                    ImPlot::EndPlot();
 
} 
} else {
                                ImGui::Text("Don't perform correlation windowSize=%d codesSize=%ld", windowSize, codesSize);
} 
 
 
 
 
}catch (const std::exception& e) {
                        ImGui::Text("Error while processing FFT: %s", e.what());
} 
 
 
 
} 
}
 

int main (int argc, char** argv)        {
            auto fftw  = FFTWManager(6); 
 
        glfwSetErrorCallback(glfw_error_callback);
        if ( 0==glfwInit() ) {
                        std::cout<<"glfw init failed"<<std::endl<<std::flush;
        return 1;
 
} 
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
 
            auto *window  = glfwCreateWindow(800, 600, "imgui_dsp", nullptr, nullptr); 
    if ( nullptr==window ) {
                        std::cout<<"failed to create glfw window"<<std::endl<<std::flush;
        return 1;
 
} 
    glfwMakeContextCurrent(window);
    std::cout<<"enable vsync"<<std::endl<<std::flush;
    glfwSwapInterval(1);
    IMGUI_CHECKVERSION();
    std::cout<<"create imgui context"<<std::endl<<std::flush;
    ImGui::CreateContext();
    ImPlot::CreateContext();
        auto &io  = ImGui::GetIO(); 
        io.ConfigFlags=io.ConfigFlags | ImGuiConfigFlags_NavEnableKeyboard;


 
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");
    glClearColor(0, 0, 0, 1);
 
            auto sampleRate  = 1.00e+7; 
        // based on Andrew Holme's code http://www.jks.com/gps/SearchFFT.cpp
 
        auto caFrequency  = 1.0230e+6; 
    auto caStep  = caFrequency/sampleRate; 
    auto corrWindowTime_ms  = 1.0    ; 
    auto corrLength  = static_cast<int>((corrWindowTime_ms*sampleRate)/1000); 
    auto caSequenceLength  = 1023; 
    std::cout<<"prepare CA code chips"<<" corrLength='"<<corrLength<<"' "<<std::endl<<std::flush;
        auto codes  = std::vector<std::vector<std::complex<double>>>(32); 
    for ( auto i = 0;i<32;i+=1 ) {
                // chatGPT decided to start PRN index with 1. I don't like it but leave it for now.
 
                        auto ca  = GpsCACodeGenerator(i+1); 
        auto chips  = ca.generate_sequence(caSequenceLength); 
        auto code  = std::vector<std::complex<double>>(corrLength); 
        auto caPhase  = 0.    ; 
        auto chipIndex  = 0; 
        for ( auto l = 0;l<corrLength;l+=1 ) {
                                    code[l]=chips[(chipIndex%caSequenceLength)] ? 1.0     :  -1.0    ;


                        caPhase+=caStep;
                        if ( 1<=caPhase ) {
                                                caPhase-=1.0    ;
                chipIndex++;
 
} 
} 
        std::cout<<"compute FFT"<<" i='"<<i<<"' "<<std::endl<<std::flush;
        {
                        if ( i==0 ) {
                                                // the first fft takes always long (even if wisdom is present). as a work around i just perform a very short fft. then it takes only a few milliseconds. subsequent large ffts are much faster
 
                                auto mini  = std::vector<std::complex<double>>(32); 
                                auto startBenchmark  = std::chrono::high_resolution_clock::now(); 
                fftw.fft(mini, 32);
                                auto endBenchmark  = std::chrono::high_resolution_clock::now(); 
                auto elapsed  = std::chrono::duration<double>(endBenchmark-startBenchmark); 
                auto elapsed_ms  = 1000*elapsed.count(); 
                std::cout<<""<<" elapsed_ms='"<<elapsed_ms<<"' "<<std::endl<<std::flush;
 
 
 
 
} 
                                    auto startBenchmark  = std::chrono::high_resolution_clock::now(); 
                        auto out  = fftw.fft(code, corrLength); 
 
                        auto endBenchmark  = std::chrono::high_resolution_clock::now(); 
            auto elapsed  = std::chrono::duration<double>(endBenchmark-startBenchmark); 
            auto elapsed_ms  = 1000*elapsed.count(); 
            std::cout<<""<<" elapsed_ms='"<<elapsed_ms<<"' "<<std::endl<<std::flush;
 
 
                        std::cout<<"codes"<<" i='"<<i<<"' "<<" codes.size()='"<<codes.size()<<"' "<<" out.size()='"<<out.size()<<"' "<<std::endl<<std::flush;
                                    codes[i]=out;


} 
 
} 
 
 
 
 
        try {
                                auto sdr  = SdrManager(64512, 1100000, 50000, 2000); 
        stopDaemon();
        startDaemonIfNotRunning();
        std::cout<<"initialize sdr manager"<<std::endl<<std::flush;
        sdr.initialize();
                sdr.set_gain_mode(false);
        sdr.setGainIF(20);
        sdr.setGainRF(0);
        sdr.set_frequency(1.575420e+9);
        sdr.set_sample_rate(sampleRate);
        sdr.set_bandwidth(1.5360e+6);
 
        sdr.startCapture();
 
                auto fn  = "/mnt5/gps.samples.cs16.fs5456.if4092.dat"; 
        auto file  = MemoryMappedComplexShortFile(fn); 
        if ( file.ready() ) {
                                    std::cout<<"first element"<<" fn='"<<fn<<"' "<<" file[0].real()='"<<file[0].real()<<"' "<<std::endl<<std::flush;
 
} 
 
        while ( !glfwWindowShouldClose(window) ) {
                        glfwPollEvents();
                        ImGui_ImplOpenGL3_NewFrame();
                        ImGui_ImplGlfw_NewFrame();
                        ImGui::NewFrame();
                        DrawPlot(file, sdr, fftw, codes);
                        ImGui::Render();
                                    auto w  = 0; 
            auto h  = 0; 
            glfwGetFramebufferSize(window, &w, &h);
            glViewport(0, 0, w, h);
            glClear(GL_COLOR_BUFFER_BIT);
 
                        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
                        glfwSwapBuffers(window);
} 
                sdr.stopCapture();
        sdr.close();
 
 
}catch (const std::runtime_error& e) {
                std::cout<<"error 1422:"<<" e.what()='"<<e.what()<<"' "<<std::endl<<std::flush;
                return -1;
}catch (const std::exception& e) {
                std::cout<<"error 1426:"<<" e.what()='"<<e.what()<<"' "<<std::endl<<std::flush;
                return -1;
} 
        return 0;
}
 
