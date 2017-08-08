#pragma once

#include <memory>
#include <cstring>
#include <assert.h>

namespace ycv{

class YSize{
    private:
		int w, h;
    public:
        YSize()
            : w(0)
            , h(0)
        {}
        YSize(const int w, const int h)
			: w(w)
			, h(h)
        {}
        YSize& operator=(YSize& sz){
            this->w = sz.w;
            this->h = sz.h;
            return *this;
        }
        YSize& operator=(YSize&& sz){
            return this->operator =(sz);
        }
        bool operator==(YSize&& sz){
        	if(this->w == sz.w && this->h == sz.h){
        		return true;
        	}
        	return false;
        }
        virtual ~YSize()
		{
		}
        int GetW() const
		{
			return this->w;
		}
        int GetH() const
		{
			return this->h;
		}
};


class YRect{
	private:
		int x, y, w, h;

	public:
		YRect() = delete;
        YRect(const int x, const int y, const int w, const int h)
			: x(x)
			, y(y)
			, w(w)
			, h(h)
		{
		}
		virtual ~YRect()
		{
		}
		int GetX() const
		{
			return this->x;
		}
		int GetY() const
		{
			return this->y;
		}
		int GetW() const
		{
			return this->w;
		}
		int GetH() const
		{
			return this->h;
		}
};


class YPoint{
	private:
		int x, y;

	public:
		YPoint() = delete;
		YPoint(const int x, const int y)
			: x(x)
			, y(y)
		{
		}
		virtual ~YPoint()
		{
		}
		int GetX() const
		{
			return this->x;
		}
		int GetY() const
		{
			return this->y;
		}
};


template <typename T>
class YMat{
	private:
		std::shared_ptr<T> data;
		size_t capacity; // The total number of elements for T
		int width; // Width for T as matrix
		int height; // Height for T as matrix
		int channels; // Channels for T as matrix
	public:
		YMat()
			: data(nullptr)
			, capacity(0)
			, width(0)
			, height(0)
			, channels(0)
		{
		}
        YMat(const int width, const int height=1, const int channels=1, T* data=nullptr)
        :YMat()
		{
			this->SetSize(width, height, channels);
            if( data ){
                memcpy(this->bits(), data, sizeof(T)*this->GetLength());
            }
		}
		virtual ~YMat()
		{
		}
		void FillZeros()
		{
			if( this->capacity == 0 ) {
				return;
			}
			memset(this->bits(), 0, this->GetLength()*sizeof(T));
		}
		void SetSize(const int width, const int height=1, const int channels=1)
		{
            if( this->capacity >= static_cast<size_t>(width * height * channels) ){
				this->width = width;
				this->height = height;
				this->channels = channels;
				return;
			}
            this->capacity = width * height * channels;
			this->width = width;
			this->height = height;
			this->channels = channels;
			this->data = std::shared_ptr<T>(new T[this->capacity], [](T* ptr){
				delete[] ptr;
			});
		}
		void SetSize(const YSize& s, const int channels=1)
		{
			this->SetSize(s.GetW(), s.GetH(), channels);
		}
		int GetWidth() const
		{
			return this->width;
		}
		int GetHeight() const
		{
			return this->height;
		}
		int GetChannels() const
		{
			return this->channels;
		}
        inline int GetLength() const
        {
            return this->width*this->height*this->channels;
        }
        // Shallow copy
        YMat<T>& operator=(YMat<T>& img)
		{
        	this->capacity 	= img.capacity;
			this->data		= img.data;
			this->width		= img.width;
			this->height	= img.height;
			this->channels	= img.channels;
			return *this;
		}
        YMat<T>& operator=(YMat<T>&& img)
        {
            return this->operator =(img);
        }
        // Deep copy
		void CopyTo(YMat<T>& img)
		{
			img.SetSize(this->width, this->height, this->channels);
			memcpy(img.bits(), this->bits(), sizeof(T)*this->GetLength());
		}
		T* const bits() const
        {
			assert(this->GetLength()!=0);
			return this->data.get();
        }
};

}
