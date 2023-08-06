// no preamble
 
#include <stdexcept>
#include <algorithm>
#include <iterator>
#include <string>
#include <fstream>
#include <iostream> 
#include <vector> 
 
 
#include "FFTWManager.h" 
 FFTWManager::FFTWManager (int number_threads)         : number_threads_(number_threads){
}[[nodiscard]] std::vector<std::complex<double>> FFTWManager::fftshift (const std::vector<std::complex<double>>& in) const        {
            auto insize2  = in.size()/2; 
    auto mid  = in.begin()+static_cast<long>(insize2); 
    auto out  = std::vector<std::complex<double>>(in.size()); 
    std::copy(mid, in.end(), out.begin());
    std::copy(in.begin(), mid, out.begin()+std::distance(mid, in.end()));
    return out;
 
}std::vector<std::complex<double>> FFTWManager::fft (std::vector<std::complex<double>>& in, size_t windowSize)         {
        if ( windowSize!=in.size() ) {
                        throw std::invalid_argument("Input size must match window size.");
 
} 
            auto out  = std::vector<std::complex<double>>(windowSize); 
    auto in1  = in; 
    fftw_execute_dft(get_plan(windowSize, FFTW_FORWARD, number_threads_), reinterpret_cast<fftw_complex*>(&in1[0]), reinterpret_cast<fftw_complex*>(&out[0]));
    return fftshift(out);
 
}std::vector<std::complex<double>> FFTWManager::ifft (std::vector<std::complex<double>>& in, size_t windowSize)         {
        if ( windowSize!=in.size() ) {
                        throw std::invalid_argument("Input size must match window size.");
 
} 
            auto out  = std::vector<std::complex<double>>(windowSize); 
    auto in1  = fftshift(in); 
    fftw_execute_dft(get_plan(windowSize, FFTW_BACKWARD, number_threads_), reinterpret_cast<fftw_complex*>(&in1[0]), reinterpret_cast<fftw_complex*>(&out[0]));
    return out;
 
} FFTWManager::~FFTWManager ()         {
        for ( const auto& kv: plans_ ) {
                        auto [size, threads]  = kv.first; 
        std::cout<<"destroy plan"<<" size='"<<size<<"' "<<" threads='"<<threads<<"' "<<"\n"<<std::flush;
        fftw_destroy_plan(kv.second);
 
} 
}fftw_plan FFTWManager::get_plan (size_t windowSize, int direction , int nThreads )         {
        std::cout<<"get_plan"<<" windowSize='"<<windowSize<<"' "<<" direction='"<<direction<<"' "<<" nThreads='"<<nThreads<<"' "<<"\n"<<std::flush;
        if ( windowSize<=0 ) {
                        throw std::invalid_argument("window size must be positive");
 
} 
            auto iter  = plans_.find({windowSize, nThreads}); 
 
            if ( plans_.end()==iter ) {
                        std::cout<<"The plan hasn't been used before. Try to load wisdom or generate it."<<" windowSize='"<<windowSize<<"' "<<" nThreads='"<<nThreads<<"' "<<"\n"<<std::flush;
                        auto wisdom_filename  = "wisdom_"+std::to_string(windowSize)+".wis"; 
 
        if ( 1<nThreads ) {
                                                wisdom_filename="wisdom_"+std::to_string(windowSize)+"_threads"+std::to_string(nThreads)+".wis";


 
} 
 
                auto in0  = std::vector<std::complex<double>>(windowSize); 
        auto out0  = std::vector<std::complex<double>>(windowSize); 
                auto wisdomFile  = std::ifstream(wisdom_filename); 
        if ( wisdomFile.good() ) {
                                    wisdomFile.close();
            fftw_import_wisdom_from_filename(wisdom_filename.c_str());
 
} else {
                        std::cout<<"can't find wisdom file"<<" wisdom_filename='"<<wisdom_filename<<"' "<<"\n"<<std::flush;
} 
        if ( 1<nThreads ) {
                                    fftw_plan_with_nthreads(nThreads);
 
} 
                auto dim  = fftw_iodim({.n=static_cast<int>(windowSize), .is=1, .os=1}); 
        auto p  = fftw_plan_guru_dft(1, &dim, 0, nullptr, reinterpret_cast<fftw_complex*>(&in0[0]), reinterpret_cast<fftw_complex*>(&out0[0]), direction, FFTW_MEASURE); 
 
        if ( nullptr==p ) {
                                    std::cout<<"error: plan not successfully created"<<"\n"<<std::flush;
 
} 
        if ( !wisdomFile.good() ) {
                                    std::cout<<"store wisdom to file"<<" wisdom_filename='"<<wisdom_filename<<"' "<<"\n"<<std::flush;
            wisdomFile.close();
            fftw_export_wisdom_to_filename(wisdom_filename.c_str());
 
} 
 
                
 
                std::cout<<"store plan in class"<<" windowSize='"<<windowSize<<"' "<<" nThreads='"<<nThreads<<"' "<<"\n"<<std::flush;
                auto insertResult  = plans_.insert({{windowSize, nThreads}, p}); 
                iter=insertResult.first;

        std::cout<<"inserted new key"<<" plans_.size()='"<<plans_.size()<<"' "<<" insertResult.second='"<<insertResult.second<<"' "<<"\n"<<std::flush;
 
 
 
 
 
} 
    return iter->second;
 
} 
 
