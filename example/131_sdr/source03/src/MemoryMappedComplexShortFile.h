#ifndef MEMORYMAPPEDCOMPLEXSHORTFILE_H
#define MEMORYMAPPEDCOMPLEXSHORTFILE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <boost/iostreams/device/mapped_file.hpp> 

class MemoryMappedComplexShortFile  {
        public:
        explicit  MemoryMappedComplexShortFile (const std::string& filename)       ;   
        std::complex<short>& operator[] (std::size_t index) const      ;   
        std::size_t size () const      ;   
         ~MemoryMappedComplexShortFile ()       ;   
        private:
        boost::iostreams::mapped_file_source file_;
        std::complex<short>*     data_=nullptr;


        const std::string& filename_;
};

#endif /* !MEMORYMAPPEDCOMPLEXSHORTFILE_H */