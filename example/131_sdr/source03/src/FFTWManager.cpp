// no preamble
 
#include <stdexcept>
#include <fstream>
#include <iostream> 
#include <vector> 
 
 
#include "FFTWManager.h" 
 FFTWManager::FFTWManager (int number_threads)         : number_threads_(number_threads){
}std::vector<std::complex<double>> FFTWManager::fftshift (const std::vector<std::complex<double>>& in) const        {
            auto mid  = in.begin()+(in.size()/2); 
    auto out  = std::vector<std::complex<double>>(in.size()); 
    std::copy(mid, in.end(), out.begin());
    std::copy(in.begin(), mid, out.begin()+std::distance(mid, in.end()));
    return out;
 
}std::vector<std::complex<double>> FFTWManager::fft (const std::vector<std::complex<double>>& in, size_t windowSize)         {
        if ( windowSize!=in.size() ) {
                        throw std::invalid_argument("Input size must match window size.");
 
} 
            auto out  = std::vector<std::complex<double>>(windowSize); 
    fftw_execute_dft(get_plan(windowSize, FFTW_FORWARD, number_threads_), reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>*>(in.data())), reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>*>(out.data())));
    return fftshift(out);
 
}std::vector<std::complex<double>> FFTWManager::ifft (const std::vector<std::complex<double>>& in, size_t windowSize)         {
        if ( windowSize!=in.size() ) {
                        throw std::invalid_argument("Input size must match window size.");
 
} 
            auto in2  = fftshift(in); 
    auto out  = std::vector<std::complex<double>>(windowSize); 
    fftw_execute_dft(get_plan(windowSize, FFTW_BACKWARD, number_threads_), reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>*>(in2.data())), reinterpret_cast<fftw_complex*>(const_cast<std::complex<double>*>(out.data())));
    return out;
 
} FFTWManager::~FFTWManager ()         {
        for ( const auto& kv: plans_ ) {
                fftw_destroy_plan(kv.second);
} 
}fftw_plan FFTWManager::get_plan (size_t windowSize, int direction , int nThreads )         {
        if ( windowSize<=0 ) {
                        throw std::invalid_argument("window size must be positive");
 
} 
            auto iter  = plans_.find({windowSize, nThreads}); 
 
            if ( plans_.end()==iter ) {
                        std::cout<<"The plan hasn't been used before. Try to load wisdom or generate it."<<" windowSize='"<<windowSize<<"' "<<" nThreads='"<<nThreads<<"' "<<std::endl<<std::flush;
                        auto wisdom_filename  = "wisdom_"+std::to_string(windowSize)+".wis"; 
 
        if ( 1<nThreads ) {
                                                wisdom_filename="wisdom_"+std::to_string(windowSize)+"_threads"+std::to_string(nThreads)+".wis";


 
} 
 
                auto in0  = std::vector<std::complex<double>>(windowSize); 
        auto out0  = std::vector<std::complex<double>>(windowSize); 
        auto *in  = reinterpret_cast<double(*)[2]>(in0.data()); 
        auto *out  = reinterpret_cast<double(*)[2]>(out0.data()); 
                auto wisdomFile  = std::ifstream(wisdom_filename); 
        if ( wisdomFile.good() ) {
                                    wisdomFile.close();
            fftw_import_wisdom_from_filename(wisdom_filename.c_str());
 
} else {
                        std::cout<<"can't find wisdom file"<<" wisdom_filename='"<<wisdom_filename<<"' "<<std::endl<<std::flush;
} 
        if ( 1<nThreads ) {
                                    fftw_plan_with_nthreads(nThreads);
 
} 
                auto dim  = fftw_iodim({.n=static_cast<int>(windowSize), .is=1, .os=1}); 
        auto p  = fftw_plan_guru_dft(1, &dim, 0, nullptr, in, out, direction, FFTW_MEASURE); 
 
        if ( !wisdomFile.good() ) {
                                    std::cout<<"store wisdom to file"<<" wisdom_filename='"<<wisdom_filename<<"' "<<std::endl<<std::flush;
            wisdomFile.close();
            fftw_export_wisdom_to_filename(wisdom_filename.c_str());
 
} 
 
                
 
                std::cout<<"store plan in class"<<" windowSize='"<<windowSize<<"' "<<" nThreads='"<<nThreads<<"' "<<std::endl<<std::flush;
                auto insertResult  = plans_.insert({{windowSize, nThreads}, p}); 
                iter=insertResult.first;

        std::cout<<"inserted new key"<<" plans_.size()='"<<plans_.size()<<"' "<<" insertResult.second='"<<insertResult.second<<"' "<<std::endl<<std::flush;
 
 
 
 
 
} else {
                        std::cout<<"plan has been used recently, reuse it."<<std::endl<<std::flush;
 
} 
    return iter->second;
 
} 
 
