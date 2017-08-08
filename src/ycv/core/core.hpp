#pragma once

#include <cstddef>
#include <cstring>
#include <vector>
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
		std::vector<T> buffer;
		//T* data; //Point for data
		size_t length; // Number of elements for T
		int width; // Width for T as matrix
		int height; // Height for T as matrix
		int channels; // Channels for T as matrix

	public:
		YMat()
			//: data(nullptr)
			: length(0)
			, width(0)
			, height(0)
			, channels(0)
		{
		}
        YMat(const int width, const int height=1, const int channels=1, T* data=nullptr)
        :YMat()
		{
            this->length = 0;
			this->SetSize(width, height, channels);
            if( data ){
                ::memcpy(this->bits(), data, sizeof(T)*this->length);
            }
		}
		virtual ~YMat()
		{
			this->Clear();
		}
        void Clear()
		{
			if(this->length == 0){
				return;
			}

            //delete[] this->data;
			this->length = 0;
			this->width = 0;
			this->height = 0;
            this->channels = 0;

		}
		void FillZeros()
		{
			if( this->length == 0 ) {
				return;
			}
			::memset(this->bits(), 0, this->length*sizeof(T));
		}
		void SetSize(const int width, const int height=1, const int channels=1)
		{
            if( this->length == static_cast<size_t>(width * height * channels) ){
				this->length = width * height * channels;
				this->width = width;
				this->height = height;
				this->channels = channels;
				this->FillZeros();
				return;
			}
			this->Clear();
			this->length = width * height * channels;
			this->width = width;
			this->height = height;
			this->channels = channels;
			this->buffer.reserve(this->length);
			this->FillZeros();
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
        int GetLength() const
        {
            return this->length;
        }
        YMat<T>& operator=(YMat<T>& img)
		{
			//Shallow copy to this
			this->Clear();
			//this->data		= img.data;
			this->buffer 	= img.buffer;
			this->length	= img.length;
			this->width		= img.width;
			this->height	= img.height;
			this->channels	= img.channels;

            //Clear img
            img.length = 0;
            img.width = 0;
            img.height = 0;
            img.channels = 0;

			return *this;
		}
        YMat<T>& operator=(YMat<T>&& img)
        {
            return this->operator =(img);
        }
		void CopyTo(YMat<T>& img)
		{
			img.Clear();
			img.SetSize(this->width, this->height, this->channels);
			img.length	 = this->length;
			img.width	 = this->width;
			img.height	 = this->height;
			img.channels = this->channels;
			::memcpy(img.bits(), this->bits(), sizeof(T)*this->length);
		}
		T* const bits() const
        {
			assert(this->GetLength()!=0);
			return &this->buffer[0];
        }
};

}
