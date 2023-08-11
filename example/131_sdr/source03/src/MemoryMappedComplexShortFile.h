#ifndef MEMORYMAPPEDCOMPLEXSHORTFILE_H
#define MEMORYMAPPEDCOMPLEXSHORTFILE_H

#include <cstddef>
#include <string>
#include <complex>
#include <boost/iostreams/device/mapped_file.hpp> 

class MemoryMappedComplexShortFile  {
        public:
            MemoryMappedComplexShortFile (const MemoryMappedComplexShortFile &) = delete;
    MemoryMappedComplexShortFile & operator= (const MemoryMappedComplexShortFile &) = delete;
 
        explicit  MemoryMappedComplexShortFile (const std::string& filename, size_t length, size_t offset)       ;   
        std::complex<short> operator[] (std::size_t index) const      ;   
        std::size_t size () const      ;   
        bool ready () const      ;   
         ~MemoryMappedComplexShortFile ()       ;   
        private:
        boost::iostreams::mapped_file_source file_;
        std::complex<short>*     data_=nullptr;


        const std::string& filename_;
        size_t length_;
        size_t offset_;
        bool     ready_=false;


};

#endif /* !MEMORYMAPPEDCOMPLEXSHORTFILE_H */