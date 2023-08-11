// no preamble
 
#include <filesystem>
#include <fstream>
#include <iostream> 
 
#include "MemoryMappedComplexShortFile.h" 
 MemoryMappedComplexShortFile::MemoryMappedComplexShortFile (const std::string& filename, size_t length, size_t offset)         : filename_(filename), length_(length), offset_(offset){
        std::cout<<"try to mmap file"<<" filename='"<<filename<<"' "<<" std::filesystem::exists(filename)='"<<std::filesystem::exists(filename)<<"' "<<"\n"<<std::flush;
        if ( std::filesystem::exists(filename_) ) {
                        file_.open(filename, length, offset);
        if ( file_.is_open() ) {
                                    std::cout<<"MMAP"<<" file_.size()='"<<file_.size()<<"' "<<" file_.alignment()='"<<file_.alignment()<<"' "<<" file_.data()[0]='"<<file_.data()[0]<<"' "<<"\n"<<std::flush;
                        data_=reinterpret_cast<std::complex<short>*>(const_cast<char*>(file_.data()));

            std::cout<<"entry mmap"<<" data_[0]='"<<data_[0]<<"' "<<"\n"<<std::flush;
                        ready_=true;


 
} 
 
} 
}std::complex<short> MemoryMappedComplexShortFile::operator[] (std::size_t index) const        {
        return data_[index];
}std::size_t MemoryMappedComplexShortFile::size () const        {
        return file_.size();
}bool MemoryMappedComplexShortFile::ready () const        {
        return ready_;
} MemoryMappedComplexShortFile::~MemoryMappedComplexShortFile ()         {
        file_.close();
} 
 
