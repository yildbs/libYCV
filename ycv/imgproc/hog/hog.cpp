#include <ycv/imgproc/hog/hog.h>

#include <fstream>
#include <stdio.h>

namespace ycv{

namespace hog{

HOGDescriptorSingle::HOGStaticCache HOGDescriptorSingle::static_cache;

HOGDescriptorSingle::HOGStaticCache::HOGStaticCache()
{
    float* lut = new float[256];
    for(int i=0;i<256;i++){
        lut[i] = std::sqrt(i);
    }
    HOGDescriptorSingle::HOGStaticCache::gamma_lut = YMat<float>(256,1,1, lut);

    for(int i=0;i<256;i++){
        lut[i] = i;
    }
    HOGDescriptorSingle::HOGStaticCache::normal_lut = YMat<float>(256,1,1, lut);

    delete[] lut;
}

HOGDescriptorSingle::HOGStaticCache::~HOGStaticCache()
{

}

HOGDescriptorSingle::HOGCache::HOGCache()
    : gamma_correction(true)
	, width(0)
    , height(0)
	, channels(0)
	, xmap(nullptr)
	, ymap(nullptr)
	, lut(nullptr)
    , signed_gradient(false)
    , num_bins(9)
	, cell_size(0)
	, block_size(0)
	, descriptor_size_width(0)
	, descriptor_size_height(0)
	, angle_scale(0.0f)
	, cell_width(0)
	, cell_height(0)
	, block_width(0)
	, block_height(0)
	, block_width_in_descriptor(0)
	, block_height_in_descriptor(0)
	, descriptor_width(0)
	, descriptor_height(0)
	, max_width_interest(0)
	, max_height_interest(0)
	, rho(0.0f)
	, scale_factor(1.1f)
{
    this->angle_scale = this->signed_gradient
                ? static_cast<float>(this->num_bins) / (2.0 * M_PI )
                : static_cast<float>(this->num_bins) / ( M_PI );
}

HOGDescriptorSingle::HOGCache::~HOGCache()
{
}


/////////////////////////////////////////////////////////////////
/// \brief HOGDescriptor::HOGDescriptor
///

HOGDescriptorSingle::HOGDescriptorSingle()
{

}

HOGDescriptorSingle::~HOGDescriptorSingle()
{
}

void HOGDescriptorSingle::Initialize(     const int width
                                , const int height
                                , const int channels\
                                , const float scale_factor
                                , const bool gamma_correction
                                , const bool signed_gradient
                                , const int num_bins
                                , const int cell_size
                                , const int block_size
                                , const int descriptor_size_width
                                , const int descriptor_size_height)
{
    if(     this->cache.width != width
        ||  this->cache.height != height
        ||  this->cache.channels != channels
        ||  this->cache.scale_factor != scale_factor
        ||  this->cache.gamma_correction != gamma_correction
        ||  this->cache.signed_gradient != signed_gradient
        ||  this->cache.num_bins != num_bins
        ||  this->cache.cell_size != cell_size
        ||  this->cache.block_size != block_size
        ||  this->cache.descriptor_size_width != descriptor_size_width
        ||  this->cache.descriptor_size_height != descriptor_size_height
            ){

        // Set variables
        this->cache.width = width;
        this->cache.height = height;
        this->cache.channels = channels;

        // Initialize xmap as channels
        if( this->cache.channels == 1){
            int* _xmap = new int[width + 2];
            _xmap[0] = 0;
            for(int i=0;i<width;i++){
                _xmap[i+1] = i;
            }
            _xmap[width+1] = width-1;
            this->cache._xmap = YMat<int>(width+2, 1, 1, _xmap);
            //this->cache.xmap = &this->cache._xmap.bits()[1];
            SafeRelease(_xmap);
        }else if(this->cache.channels == 3){
            int total_length = (width + 2) * 3;
            int* _xmap = new int[total_length];

            _xmap[0] = 0;
            _xmap[1] = 1;
            _xmap[2] = 2;
            for(int i=0;i<width*3;i+=3){
                _xmap[i+3] = i + 0;
                _xmap[i+4] = i + 1;
                _xmap[i+5] = i + 2;
            }
            _xmap[width*3+3] = (width-1)*3+0;
            _xmap[width*3+4] = (width-1)*3+1;
            _xmap[width*3+5] = (width-1)*3+2;
            this->cache._xmap = YMat<int>(total_length, 1, 1, _xmap);
            this->cache.xmap = &this->cache._xmap.bits()[3];
            SafeRelease(_xmap);
        }

        // Initialize scale_factor
        this->cache.scale_factor = scale_factor;

        // Initialzie ymap
        int* _ymap = new int[height + 2];
        _ymap[0] = 0;
        _ymap[height+1] = height-1;
        for(int i=0;i<height;i++){
            _ymap[i+1] = i;
        }
        this->cache._ymap = YMat<int>(height+2, 1, 1, _ymap);
        this->cache.ymap = &this->cache._ymap.bits()[1];
        SafeRelease(_ymap);

        // Allocate dbuf
        this->cache.dbuf = YMat<float>(width*4);
        this->cache.orientation = YMat<char>(width, height, 2);
        this->cache.magnitude = YMat<float>(width, height, 2);

        // Set lut
        this->cache.gamma_correction = gamma_correction;
        if(this->cache.gamma_correction){
            this->cache.lut = HOGDescriptorSingle::static_cache.gamma_lut.bits();
        }else{
            this->cache.lut = HOGDescriptorSingle::static_cache.normal_lut.bits();
        }

        //Set angle scale
        this->cache.angle_scale = this->cache.signed_gradient
                    ? static_cast<float>(this->cache.num_bins) / (2.0 * M_PI + 0.0000000001 )
                    : static_cast<float>(this->cache.num_bins) / ( M_PI + 0.0000000001);

        this->cache.cell_size = cell_size;
        this->cache.block_size = block_size;
        this->cache.descriptor_size_width = descriptor_size_width;
        this->cache.descriptor_size_height = descriptor_size_height;

        this->cache.cell_width = static_cast<int>(std::floor(width /cell_size));
        this->cache.cell_height = static_cast<int>(std::floor(height /cell_size));
        this->cache.block_width = this->cache.cell_width + 1 - this->cache.block_size;
        this->cache.block_height = this->cache.cell_height + 1 - this->cache.block_size;
        this->cache.block_width_in_descriptor = this->cache.descriptor_size_width + 1 - this->cache.block_size;
        this->cache.block_height_in_descriptor = this->cache.descriptor_size_height + 1 - this->cache.block_size;

        this->cache.descriptor_width = this->cache.cell_width + 1 - this->cache.descriptor_size_width;
        this->cache.descriptor_height = this->cache.cell_height + 1 - this->cache.descriptor_size_height;
        this->cache.descriptor = YMat<float>(1, 1, num_bins * this->cache.block_size * this->cache.block_size * this->cache.block_width_in_descriptor * this->cache.block_height_in_descriptor);

        this->cache.max_width_interest = this->cache.cell_width * cell_size;
        this->cache.max_height_interest = this->cache.cell_height * cell_size;
        this->cache.histograms = YMat<float>(this->cache.cell_width, this->cache.cell_height, num_bins);
        this->cache.blocks = YMat<float>(this->cache.block_width, this->cache.block_height, num_bins * this->cache.block_size * this->cache.block_size);
    }
}

void HOGDescriptorSingle::SetImage(YMat_& image)
{
    this->image = image;
}

HOGDescriptorSingle::YRectList HOGDescriptorSingle::Detect()
{
    HOGDescriptorSingle::YRectList founds;

    const int cell_size = this->cache.cell_size;
    const int descriptor_size_width = this->cache.descriptor_size_width;
    const int descriptor_size_height = this->cache.descriptor_size_height;
    float const scale_factor = this->cache.scale_factor;
    const int width = static_cast<int>((float)cell_size * descriptor_size_width * scale_factor);
    const int height = static_cast<int>((float)cell_size * descriptor_size_height * scale_factor);

    this->MakeHistogram();
    this->MakeNormalizedBlocks();
    for(int cell_y=0;cell_y<this->cache.descriptor_height;cell_y++){
        for(int cell_x=0;cell_x<this->cache.descriptor_width;cell_x++){
            this->MakeDescriptor(cell_x, cell_y);
            if(this->Predict()){
                int x1 = static_cast<int>((float)cell_x * cell_size * scale_factor);
                int y1 = static_cast<int>((float)cell_y * cell_size * scale_factor);
                founds.push_back(YRect(x1, y1, width, height));
            }
        }
    }
    return founds;
}

void HOGDescriptorSingle::MakeDescriptorForTrain()
{
    this->MakeHistogram();
    this->MakeNormalizedBlocks();
    this->MakeDescriptor(0, 0);
}

void HOGDescriptorSingle::MakeHistogram()
{
    int const width = this->cache.width;
    int const height = this->cache.height;
    int const ch = this->cache.channels;
    int const step = width * ch;
    int* const  xmap = this->cache.xmap;
    int* const  ymap = this->cache.ymap;
    float* const lut = this->cache.lut;
    char* const orientation = this->cache.orientation.bits();
    float* const magnitude = this->cache.magnitude.bits();
    float* const histogram = this->cache.histograms.bits();
    float* const dbuf = this->cache.dbuf.bits();
    //int const

    int const max_width_interest = this->cache.max_width_interest;
    int const cell_width = this->cache.cell_width;


    int num_bins = this->cache.num_bins;

    this->cache.histograms.FillZeros();

    if(ch == 1){
    }else if(ch == 3){
        for(int y=0;y<height;y++){

            const unsigned char* img_ptr = 	&this->image.bits()[ymap[y]		* step];
            const unsigned char* up_ptr = 	&this->image.bits()[ymap[y-1] 	* step];
            const unsigned char* down_ptr = &this->image.bits()[ymap[y+1] 	* step];
            const int cell_height_idx = static_cast<int>(y / this->cache.cell_size);

            for(int x=0;x<width*3;){
                int x_idx = static_cast<int>( std::floor(x/3));
                if(x_idx >= max_width_interest){
                    break;
                }

                int up, down, left, right;
                float dx_max, dy_max, dx, dy, mag_max, mag;

                // calculate R
                up      =   static_cast<int>(lut[ up_ptr    [xmap[x]]]);
                down    =   static_cast<int>(lut[ down_ptr  [xmap[x]]]);
                left    =   static_cast<int>(lut[ img_ptr   [xmap[x-3]]]);
                right   =   static_cast<int>(lut[ img_ptr   [xmap[x+3]]]);

                dx_max = static_cast<float>(right - left);
                dy_max = static_cast<float>(up - down);
                mag_max = dx_max*dx_max + dy_max*dy_max;
                x++;

                // calculate G
                up      =   static_cast<int>(lut[ up_ptr    [xmap[x]]]);
                down    =   static_cast<int>(lut[ down_ptr  [xmap[x]]]);
                left    =   static_cast<int>(lut[ img_ptr   [xmap[x-3]]]);
                right   =   static_cast<int>(lut[ img_ptr   [xmap[x+3]]]);

                dx = static_cast<float>(right - left);
                dy = static_cast<float>(up - down);
                mag = dx*dx + dy*dy;
                if( mag_max < mag ){
                    dx_max = dx;
                    dy_max = dy;
                    mag_max = mag;
                }
                x++;

                // calculate B
                up      =   static_cast<int>(lut[ up_ptr    [xmap[x]]]);
                down    =   static_cast<int>(lut[ down_ptr  [xmap[x]]]);
                left    =   static_cast<int>(lut[ img_ptr   [xmap[x-3]]]);
                right   =   static_cast<int>(lut[ img_ptr   [xmap[x+3]]]);

                dx = static_cast<float>(right - left);
                dy = static_cast<float>(up - down);
                mag = dx*dx + dy*dy;
                if( mag_max < mag ){
                    dx_max = dx;
                    dy_max = dy;
                    mag_max = mag;
                }
                x++;

                dbuf[x_idx + width * 0] = dx_max;
                dbuf[x_idx + width * 1] = dy_max;
                dbuf[x_idx + width * 2] = mag_max;

                //Calculate angle
                float angle = 0;

                if( mag_max != 0 ){
                    auto dx_temp = dx_max;
                    auto dy_temp = dy_max;
                    if( this->cache.signed_gradient == false && dx_max < 0. ){
                        dx_temp = -1 * dx_temp;
                        dy_temp = -1 * dy_temp;
                    }
                    angle = std::atan2(dy_temp, dx_temp);

                    //TODO: Test w/ and w/o below code
                    if( angle < 0 ){
                        angle = angle + 2 * M_PI;
                    }
                }

                int const cell_width_idx = static_cast<int>(x_idx / this->cache.cell_size);
                float* const histogram_dst = &histogram[cell_height_idx*cell_width*num_bins + cell_width_idx*num_bins];

                angle = angle * this->cache.angle_scale - 0.5f;
                int hidx = std::floor(angle);
                angle = angle - hidx;

                magnitude[y*width*2 + x_idx*2 + 0] = mag_max * (1.-angle);
                magnitude[y*width*2 + x_idx*2 + 1] = mag_max * angle;

                float mag0 = magnitude[y*width*2 + x_idx*2 + 0];
                float mag1 = magnitude[y*width*2 + x_idx*2 + 1];

                if(hidx < 0 ){
                    hidx += this->cache.num_bins;
                }else if( hidx >= this->cache.num_bins ){
                    hidx -= this->cache.num_bins;
                }
                orientation[y*width*2 + x_idx*2 + 0] = hidx;
                int hidx0 = hidx;
                hidx++;
                if(hidx >= this->cache.num_bins ){
                    hidx -= this->cache.num_bins;
                }
                orientation[y*width*2 + x_idx*2 + 1] = hidx;
                int hidx1 = hidx;

                histogram_dst[hidx0] += mag0;
                histogram_dst[hidx1] += mag1;
            }
        }
    }
}

void HOGDescriptorSingle::MakeNormalizedBlocks()
{
    int const block_size = this->cache.block_size;
    int const block_width = this->cache.block_width;
    int const block_height = this->cache.block_height;
    int const block_length = this->cache.blocks.GetChannels();
    int const num_bins = this->cache.num_bins;
    float* const block = this->cache.blocks.bits();

    for(int y=0;y<block_height;y++){
        float* const dst_y = &block[y*block_width*block_length];
        for(int x=0;x<block_width;x++){
            float* const dst_x = &dst_y[x*block_length];
            int idx=0;
            for(int _y=0;_y<block_size;_y++){
                for(int _x=0;_x<block_size;_x++){
                    float* const src = this->GetHistogram(x+_x, y+_y);///////It is too odd....
                    ::memcpy(&dst_x[idx++*num_bins], src, num_bins*sizeof(float));
                }
            }
            this->NormalizeBlock(dst_x, block_length);
        }
    }
}

void HOGDescriptorSingle::NormalizeBlock(float* const hist, int const block_length)
{
    //Normalize histogram as L2-hys
    float partsum[4];
    partsum[0] = 0.;
    partsum[1] = 0.;
    partsum[2] = 0.;
    partsum[3] = 0.;
    float sum = 0.;
    auto threshold = this->cache.l2_hys_threshold;

    int idx=0;
    for(idx=0;idx<block_length-4;idx+=4){
        partsum[0] += hist[idx + 0] * hist[idx + 0];
        partsum[1] += hist[idx + 1] * hist[idx + 1];
        partsum[2] += hist[idx + 2] * hist[idx + 2];
        partsum[3] += hist[idx + 3] * hist[idx + 3];
    }
    float t0 = partsum[0] + partsum[1];
    float t1 = partsum[2] + partsum[3];
    sum = t0 + t1;
    for(;idx<block_length;idx++){
        sum += hist[idx] * hist[idx];
    }

    float scale = 1.f/(std::sqrt(sum) + block_length*0.1);

    partsum[0] = 0.;
    partsum[1] = 0.;
    partsum[2] = 0.;
    partsum[3] = 0.;
    sum = 0.;
    for(idx=0;idx<block_length-4;idx+=4){
        hist[idx+0] = std::min(hist[idx+0]*scale, threshold);
        hist[idx+1] = std::min(hist[idx+1]*scale, threshold);
        hist[idx+2] = std::min(hist[idx+2]*scale, threshold);
        hist[idx+3] = std::min(hist[idx+3]*scale, threshold);

        partsum[0] += hist[idx + 0] * hist[idx + 0];
        partsum[1] += hist[idx + 1] * hist[idx + 1];
        partsum[2] += hist[idx + 2] * hist[idx + 2];
        partsum[3] += hist[idx + 3] * hist[idx + 3];
    }
	t0 = partsum[0] + partsum[1];
    t1 = partsum[2] + partsum[3];
    sum = t0 + t1;
    for(;idx<block_length;idx++){
        hist[idx] = std::min(hist[idx]*scale, threshold);
        sum += hist[idx] * hist[idx];
    }
    scale = 1.f/(std::sqrt(sum) + 1e-3f);
    for(idx=0;idx<block_length;idx++){
        hist[idx] = hist[idx]*scale;
    }
}

void HOGDescriptorSingle::MakeDescriptor(int cell_x_idx, int cell_y_idx)
{
    auto descriptor_width = this->cache.descriptor_width;
    auto descriptor_height = this->cache.descriptor_height;
    if( cell_x_idx >= descriptor_width || cell_y_idx >= descriptor_height ){
        throw std::string("out of range error");
    }

    auto block_width_in_descriptor = this->cache.block_width_in_descriptor;
    auto block_height_in_descriptor = this->cache.block_height_in_descriptor;
    auto const block_width = this->cache.block_width;
    auto const block_length = this->cache.blocks.GetChannels();
    float* const block = this->cache.blocks.bits();
    auto const descriptor = this->cache.descriptor.bits();
    int idx=0;
    for(int y=cell_y_idx;y<cell_y_idx+block_height_in_descriptor;y++){
        float* const dst_y = &block[y*block_width*block_length];
        for(int x=cell_x_idx;x<cell_x_idx+block_width_in_descriptor;x++){
            float* const block = &dst_y[x*block_length];
            ::memcpy(&descriptor[idx++*block_length], block, block_length*sizeof(float));
        }
    }
}

int HOGDescriptorSingle::GetDescriptorLength() const
{
    return this->cache.descriptor.GetLength();
}

float* const HOGDescriptorSingle::GetDescriptor()
{
    return this->cache.descriptor.bits();
}

void HOGDescriptorSingle::SetSupportVector(int length, float* ptr, float rho)
{
    if( this->GetDescriptorLength() != length ){
        throw std::string("error. The length of support vector and descriptor is not same!");
    }
    this->cache.rho = rho;
    this->cache.support_vector = YMat<float>(length);
    ::memcpy(this->cache.support_vector.bits(), ptr, length*sizeof(float));
}

int HOGDescriptorSingle::GetSupportVectorLength() const
{
    return this->cache.support_vector.GetLength();
}
float* HOGDescriptorSingle::GetSupportVector()
{
    return this->cache.support_vector.bits();
}
bool HOGDescriptorSingle::Predict()
{
    float result = 0;
    auto length = this->GetDescriptorLength();
    auto descriptor = this->GetDescriptor();
    auto support_vector = this->GetSupportVector();
    for(int i=0;i<length;i++){
        result += descriptor[i] * support_vector[i];
    }

    if( this->cache.rho < result ){
        return true;
    }
    else{
        return false;
    }
}

void HOGDescriptorSingle::SaveDescriptor()
{
    auto const cell_size = this->cache.cell_size;
    auto const descriptor_size_width = this->cache.descriptor_size_width;
    auto const descriptor_size_height = this->cache.descriptor_size_height;
    auto const width = cell_size * descriptor_size_width;
    auto const height = cell_size * descriptor_size_height;
    auto const length = this->GetDescriptorLength();
    auto const ptr = this->GetDescriptor();
    char postfix[20];
    sprintf(postfix, "%dx%d", height, width);
    char file_name[200];
    sprintf(file_name, "GetDescriptor%s.cpp", postfix);
    FILE* fp = fopen(file_name, "w");
    fprintf(fp, "int cell_size_%s = %d;\n", postfix, this->cache.cell_size);
    fprintf(fp, "int length_%s = %d;\n", postfix, length);
    fprintf(fp, "float rho_%s = %.30f;\n", postfix, this->cache.rho);
    fprintf(fp, "float descriptor_%s[%d] = {\n", postfix, length);
    for(int i=0;i<length-1;i++){
        fprintf(fp, "%.30ff, ", ptr[i]);
		if( i % 6 == 0 ){
			fprintf(fp, "\n");
		}
    }
    fprintf(fp, "%.30f};", ptr[length-1]);
    fclose(fp);
}


void HOGDescriptorSingle::SaveSupportVector()
{
    auto const cell_size = this->cache.cell_size;
    auto const descriptor_size_width = this->cache.descriptor_size_width;
    auto const descriptor_size_height = this->cache.descriptor_size_height;
    auto const width = cell_size * descriptor_size_width;
    auto const height = cell_size * descriptor_size_height;
    auto const length = this->GetSupportVectorLength();
    auto const ptr = this->GetSupportVector();
    char postfix[20];
    sprintf(postfix, "%dx%d", height, width);
    char file_name[200];
    sprintf(file_name, "GetSupportVector%s.cpp", postfix);
    FILE* fp = fopen(file_name, "w");
    fprintf(fp, "int cell_size_%s = %d;\n", postfix, this->cache.cell_size);
    fprintf(fp, "int length_%s = %d;\n", postfix, length);
    fprintf(fp, "float rho_%s = %.30f;\n", postfix, this->cache.rho);
    fprintf(fp, "float support_vector_%s[%d] = {\n", postfix, length);
    for(int i=0;i<length-1;i++){
        fprintf(fp, "%.30ff, ", ptr[i]);
        if( i % 3 == 0 ){
			fprintf(fp, "\n");
		}
    }
    fprintf(fp, "%.30f};", ptr[length-1]);
    fclose(fp);
}

void HOGDescriptorSingle::SetHitThreahold(float hit_threshold)
{
    this->cache.rho = hit_threshold;
}

inline float* const HOGDescriptorSingle::GetHistogram(int cell_x_idx, int cell_y_idx)
{
    if( cell_x_idx >= this->cache.cell_width || cell_y_idx >= this->cache.cell_height ){
        throw std::string("out of range error!");
    }
    return &this->cache.histograms.bits()[cell_y_idx*this->cache.cell_width*this->cache.num_bins + cell_x_idx*this->cache.num_bins];
}

inline const float HOGDescriptorSingle::GetScaleFactor()
{
    return this->cache.scale_factor;
}


HOGDescriptor::HOGDescriptor::HOGDescriptor()
	: width(0)
	, height(0)
	, channels(0)
	, scale_factor(1.05f)
	, nlevels(10)
	, gamma_correction(true)
	, signed_gradient(false)
	, num_bins(9)
	, cell_size(8)
	, block_size(2)
	, descriptor_size_width(6)
	, descriptor_size_height(12)
	, rho(0.0f)
	, hog_descriptors(nullptr)
{
}

HOGDescriptor::HOGDescriptor::~HOGDescriptor()
{
    SafeRelease(this->hog_descriptors);
}

#ifdef USE_OPENCV
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
HOGDescriptor::YRectList HOGDescriptor::DetectMultiScale(YMat_& yimage)
{
    YRectList founds;
    int const nlevels = this->nlevels;
    int const width = yimage.GetWidth();
    int const height = yimage.GetHeight();

    cv::Mat image(height, width, CV_8UC3, yimage.bits());
    ::memcpy(image.data, yimage.bits(), sizeof(char)*width*height*3);

    for(int level = 0;level<nlevels;level++){
        cv::Mat resized_image;
        float const scale_factor = this->hog_descriptors[level].GetScaleFactor();
        int const resized_width = static_cast<int>((float)width/scale_factor);
        int const resized_height = static_cast<int>((float)height/scale_factor);
        cv::resize(image, resized_image, cv::Size(resized_width, resized_height));

        YMat_ yresized_image(resized_width, resized_height, 3, resized_image.data);

        this->hog_descriptors[level].SetImage(yresized_image);
        auto list = this->hog_descriptors[level].Detect();
        for(auto rc : list){
            founds.push_back(rc);
        }
    }
    return founds;
}
#endif

HOGDescriptor& HOGDescriptor::Initialize()
{
    int const width = this->width;
    int const height = this->height;
    int const channels = this->channels;
    float const scale_factor = this->scale_factor;
    int const nlevels = this->nlevels;
    bool const gamma_correction = this->gamma_correction;
    bool const signed_gradient = this->signed_gradient;
    int const num_bins = this->num_bins;
    int const cell_size = this->cell_size;
    int const block_size = this->block_size;
    int const descriptor_size_width = this->descriptor_size_width;
    int const descriptor_size_height = this->descriptor_size_height;
	int const descriptor_width = cell_size * descriptor_size_width;
	int const descriptor_height = cell_size * descriptor_size_height;

    if(     width == 0
         || height == 0
         || channels == 0
         || this->support_vector.GetLength() == 0 ){
        throw std::string("HOGDescriptor is not initialized");
    }

    this->hog_descriptors = new HOGDescriptorSingle[nlevels];

    int level = 0;
    float scale_factor_single = 1.0;
    for(level=0; level<nlevels;level++){
        int resized_width = static_cast<int>((float)width / scale_factor_single);
        int resized_height = static_cast<int>((float)height / scale_factor_single);

        if( 	resized_width < descriptor_width
        ||	 	resized_height < descriptor_height ){
			break;
		}

        this->hog_descriptors[level].Initialize(    resized_width
                                                  , resized_height
                                                  , channels
                                                  , scale_factor_single
                                                  , gamma_correction
                                                  , signed_gradient
                                                  , num_bins
                                                  , cell_size
                                                  , block_size
                                                  , descriptor_size_width
                                                  , descriptor_size_height);
        this->hog_descriptors[level].SetSupportVector(this->support_vector.GetLength(), this->support_vector.bits(), this->rho);

		scale_factor_single *= scale_factor;
    }
    this->nlevels = level;
    return *this;
}

HOGDescriptor& HOGDescriptor::SetWidth(int value)
{
    this->width = value;
    return *this;
}

HOGDescriptor& HOGDescriptor::SetHeight(int value)
{
    this->height = value;
    return *this;
}

HOGDescriptor& HOGDescriptor::SetChannels(int value)
{
    this->channels = value;
    return *this;
}

HOGDescriptor& HOGDescriptor::SetScaleFactor(float value)
{
    this->scale_factor = value;
    return *this;
}

HOGDescriptor& HOGDescriptor::SetNlevels(int value)
{
    this->nlevels = value;
    return *this;
}

HOGDescriptor& HOGDescriptor::SetGammaCorrection(bool value)
{
    this->gamma_correction = value;
    return *this;
}

HOGDescriptor& HOGDescriptor::SetSignedGradient(bool value)
{
    this->signed_gradient = value;
    return *this;
}

HOGDescriptor& HOGDescriptor::SetNumBins(int value)
{
    this->num_bins = value;
    return *this;
}

HOGDescriptor& HOGDescriptor::SetCellSize(int value)
{
    this->cell_size = value;
    return *this;
}

HOGDescriptor& HOGDescriptor::SetBlockSize(int value)
{
    this->block_size = value;
    return *this;
}

HOGDescriptor& HOGDescriptor::SetDescriptorSizeWidth(int value)
{
    this->descriptor_size_width = value;
    return *this;
}

HOGDescriptor& HOGDescriptor::SetDescriptorSizeHeight(int value)
{
    this->descriptor_size_height = value;
    return *this;
}

HOGDescriptor& HOGDescriptor::SetHitThreshold(float value)
{
    this->rho = value;
    return *this;
}

HOGDescriptor& HOGDescriptor::SetSupportVector(int length, float* ptr, float rho)
{
    this->rho = rho;
    this->support_vector = YMat<float>(length);
    ::memcpy(this->support_vector.bits(), ptr, length*sizeof(float));
    return *this;
}

}

}
