#ifndef MEMORYMAPPEDCOMPLEXSHORTFILE_H
#define MEMORYMAPPEDCOMPLEXSHORTFILE_H

#include <cstddev>
#include <string>
#include <complex>
#include <boost/iostreams/device/mapped_file.hpp> 

class MemoryMappedComplexShortFile  {
        public:
        explicit  MemoryMappedComplexShortFile (const std::string& filename)       ;   
        std::complex<short>& operator[] (std::size_t index) const      ;   
        std::size_t size () const      ;   
        bool ready () const      ;   
         ~MemoryMappedComplexShortFile ()       ;   
        private:
        boost::iostreams::mapped_file_source file_;
        std::complex<short>*     data_=nullptr;


        const std::string& filename_;
        bool     ready_=false;


};

#endif /* !MEMORYMAPPEDCOMPLEXSHORTFILE_H */