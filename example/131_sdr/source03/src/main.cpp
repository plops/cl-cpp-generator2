#include <iostream>
#include <vector>
#include <filesystem>
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
 

void startDaemonIfNotRunning ()        {
        std::cout<<"verify that sdr daemon is running"<<std::endl<<std::flush;
        if ( !isDaemonRunning() ) {
                        std::cout<<"sdrplay daemon is not running. start it"<<std::endl<<std::flush;
        system("/usr/bin/sdrplay_apiService &");
        sleep(1);
 
} 
}
 
 

void DrawPlot (const MemoryMappedComplexShortFile& file, SdrManager& sdr)        {
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
 
            static bool automaticGainMode  = true; 
    static bool old_automaticGainMode  = true; 
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
    ImGui::SliderInt("Start", &start, 0, maxStart);
 
            static int windowSizeIndex  = 8; 
    auto itemsInt  = std::vector<int>({16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576}); 
    auto windowSize  = itemsInt[windowSizeIndex]; 
    auto items  = std::vector<std::string>({"16", "32", "64", "128", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65536", "131072", "262144", "524288", "1048576"}); 
    if ( ImGui::BeginCombo("Window size", items[windowSizeIndex].c_str()) ) {
                        for ( auto i = 0;i<items.size();i+=1 ) {
                                    auto is_selected  = windowSizeIndex==i; 
            if ( ImGui::Selectable(items[i].c_str(), is_selected) ) {
                                                                windowSizeIndex=i;
                windowSize=itemsInt[windowSizeIndex];


 
} 
            if ( is_selected ) {
                                                ImGui::SetItemDefaultFocus();
 
} 
 
} 
        ImGui::EndCombo();
 
} 
 
            static int bandwidthIndex  = 7; 
    static int old_bandwidthIndex  = 7; 
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
 
        if ( start+windowSize<=maxStart&&0<windowSize ) {
                                auto x  = std::vector<double>(windowSize); 
        auto y1  = std::vector<double>(windowSize); 
        auto y2  = std::vector<double>(windowSize); 
        auto zfifo  = std::vector<std::complex<short>>(windowSize); 
                static bool realtimeDisplay  = true; 
        ImGui::Checkbox("Realtime display", &realtimeDisplay);
 
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
 
                static bool logScale  = false; 
        ImGui::Checkbox("Logarithmic Y-axis", &logScale);
        try {
                                    auto fftw  = FFTWManager(); 
            auto in  = std::vector<std::complex<double>>(windowSize); 
            auto nyquist  = windowSize/2.0    ; 
            auto sampleRate  = 1.00e+7; 
            static double centerFrequency  = sdr.get_frequency(); 
            for ( auto i = 0;i<windowSize;i+=1 ) {
                                                x[i]=centerFrequency+sampleRate*(static_cast<double>(i-(windowSize/2))/windowSize);


} 
            for ( auto i = 0;i<windowSize;i+=1 ) {
                                                auto zs  = zfifo[i]; 
                auto zr  = static_cast<double>(zs.real()); 
                auto zi  = static_cast<double>(zs.imag()); 
                auto z  = std::complex<double>(zr, zi); 
                                in[i]=z;


 
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
 
            if ( ImPlot::BeginPlot(logScale ? "FFT magnitude (dB)" : "FFT magnitude (linear)") ) {
                                                                auto pointsPerPixel  = static_cast<int>(x.size()/(ImGui::GetContentRegionAvail().x)); 
 
                if ( pointsPerPixel<=1 ) {
                                                            ImPlot::PlotLine("y1", x.data(), y1.data(), windowSize);
 
} else {
                                                            // Calculate upper bound for the number of pixels, preallocate memory for vectors, initialize counter.
 
                                        auto pixels  = (x.size()+pointsPerPixel+ -1)/pointsPerPixel; 
                    auto x_downsampled  = std::vector<double>(pixels); 
                    auto y_mean  = std::vector<double>(pixels); 
                    auto count  = 0; 
                    // Iterate over the data with steps of pointsPerPixel
 
                    for ( int i=0;i<x.size();i+=pointsPerPixel ) {
                                                                        auto sum_val  = y1[i]; 
                        // Iterate over the points under the same pixel
 
                        for ( int j=i+1;j<i+pointsPerPixel&&j<x.size();j++ ) {
                                                        sum_val+=y1[j];
} 
                                                x_downsampled[count]=x[i];
                        y_mean[count]=(sum_val/pointsPerPixel);

                        count++;
 
} 
                    x_downsampled.resize(count);
                    y_mean.resize(count);
                    ImPlot::PlotLine("y_mean", x_downsampled.data(), y_mean.data(), x_downsampled.size());
 
 
} 
                                // handle user input. clicking into the graph allow tuning the sdr receiver to the specified frequency.
 
                if ( ImPlot::IsPlotHovered()&&(ImGui::IsMouseClicked(2)||ImGui::IsMouseDragging(2)) ) {
                                                                                                    centerFrequency=ImPlot::GetPlotMousePos().x;

                    sdr.set_frequency(centerFrequency);
 
 
} 
 
                ImPlot::EndPlot();
 
} 
 
 
 
}catch (const std::exception& e) {
                        ImGui::Text("Error while processing FFT: %s", e.what());
} 
 
 
 
} 
}
 

int main (int argc, char** argv)        {
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
    auto caFrequency  = 1.0230e+6; 
    auto caStep  = caFrequency/sampleRate; 
    auto corrWindowTime_ms  = 8; 
    auto corrLength  = static_cast<int>((corrWindowTime_ms*sampleRate)/1000); 
        auto codes  = std(); 
    for ( auto i = 0;i<32;i+=1 ) {
                        auto ca  = GpsCACodeGenerator(i); 
        auto chips  = ca.generate_sequence(1023); 
        auto code  = std::vector<std::complex<double>>(corrLength); 
        auto caPhase  = 0.    ; 
        auto chipIndex  = 0; 
        for ( auto i = 0;i<corrLength;i+=1 ) {
                                    code[i]=chips[chipIndex];


                        caPhase+=caStep;
                        if ( 1<=caPhase ) {
                                                caPhase-=1.0    ;
                chipIndex++;
 
} 
} 
 
} 
 
 
 
        try {
                                auto sdr  = SdrManager(64512, 1100000, 50000, 5000); 
        startDaemonIfNotRunning();
        std::cout<<"initialize sdr manager"<<std::endl<<std::flush;
        sdr.initialize();
                sdr.set_frequency(1.575420e+9);
        sdr.set_sample_rate(sampleRate);
        sdr.set_bandwidth(8.00e+6);
 
        sdr.startCapture();
 
                auto fn  = "/mnt5/capturedData_L1_rate10MHz_bw5MHz_iq_short.bin"; 
        auto file  = MemoryMappedComplexShortFile(fn); 
        std::cout<<"first element"<<" fn='"<<fn<<"' "<<" file[0].real()='"<<file[0].real()<<"' "<<std::endl<<std::flush;
 
        while ( !glfwWindowShouldClose(window) ) {
                        glfwPollEvents();
                        ImGui_ImplOpenGL3_NewFrame();
                        ImGui_ImplGlfw_NewFrame();
                        ImGui::NewFrame();
                        DrawPlot(file, sdr);
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
 
