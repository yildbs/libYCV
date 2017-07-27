#ifndef MEMORY_HPP
#define MEMORY_HPP

namespace ycv{

template <typename T>
void SafeRelease(T*& buf)
{
    if(buf != nullptr){
        delete[] buf;
        buf = nullptr;
    }
}
}

#endif // MEMORY_HPP

