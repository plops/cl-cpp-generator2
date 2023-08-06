// no preamble
 
#include <stdexcept>
#include <cstring>
#include <string> 
 
#include "GpsCACodeGenerator.h" 
 GpsCACodeGenerator::GpsCACodeGenerator (const int prn)         : prn_(prn){
        if ( prn_<1||32<prn_ ) {
                        throw std::invalid_argument("Invalid PRN: "+std::to_string(prn_));
 
} 
            const auto t0  = lut_[(prn_-1)].first; 
    const auto t1  = lut_[(prn_-1)].second; 
        tap_[0]=g2_.data()+t0;

        tap_[1]=g2_.data()+t1;


 
        std::memset(g1_.data()+1, 1, 10);
        std::memset(g2_.data()+1, 1, 10);
}std::vector<bool> GpsCACodeGenerator::generate_sequence (size_t n)         {
            auto sequence  = std::vector<bool>(n); 
    for ( size_t i = 0;i<n;i+=1 ) {
                        sequence[i]=step();


} 
    return sequence;
 
}int GpsCACodeGenerator::chip ()         {
        return g1_[10]^tap_[0][0]^tap_[1][0];
}void GpsCACodeGenerator::clock ()         {
            g1_[0]=g1_[3]^g1_[10];


            g2_[0]=g2_[2]^g2_[3]^g2_[6]^g2_[8]^g2_[9]^g2_[10];


            std::memmove(g1_.data()+1, g1_.data(), 10);
    std::memmove(g2_.data()+1, g2_.data(), 10);
 
}bool GpsCACodeGenerator::step ()         {
            auto value  = chip(); 
    clock();
    return value;
 
} 
 
