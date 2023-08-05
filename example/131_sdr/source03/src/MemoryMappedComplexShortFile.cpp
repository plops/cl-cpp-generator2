// no preamble
 
#include <stdexcept>
#include <filesystem> 
 
#include "MemoryMappedComplexShortFile.h" 
 MemoryMappedComplexShortFile::MemoryMappedComplexShortFile (const std::string& filename)         : filename_(filename){
        std::cout<<"try to mmap file"<<" filename='"<<filename<<"' "<<" std::filesystem::exists(filename)='"<<std::filesystem::exists(filename)<<"' "<<std::endl<<std::flush;
        if ( std::filesystem::exists(filename) ) {
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
 
