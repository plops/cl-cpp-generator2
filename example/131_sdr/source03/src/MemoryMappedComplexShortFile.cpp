// no preamble
 
#include <filesystem>
#include <fstream>
#include <iostream> 
 
#include "MemoryMappedComplexShortFile.h" 
 MemoryMappedComplexShortFile::MemoryMappedComplexShortFile (const std::string& filename)         : filename_(filename){
        
        if ( std::filesystem::exists(filename_) ) {
                        file_.open(filename);
        if ( file_.is_open() ) {
                                                data_=reinterpret_cast<std::complex<short>*>(const_cast<char*>(file_.data()));

                        ready_=true;


 
} 
 
} 
}std::complex<short>& MemoryMappedComplexShortFile::operator[] (std::size_t index) const        {
        return data_[index];
}std::size_t MemoryMappedComplexShortFile::size () const        {
        return file_.size();
}bool MemoryMappedComplexShortFile::ready () const        {
        return ready_;
} MemoryMappedComplexShortFile::~MemoryMappedComplexShortFile ()         {
        file_.close();
} 
 
