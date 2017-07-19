#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include "../../YCVCore/core.hpp"
#include "../../YCVCore/memory.hpp"

class HOGDescriptorSingle{
public:
    typedef ycv::YMat<unsigned char> YMat_;
    typedef std::vector<ycv::YRect> YRectList;

private:
    class HOGStaticCache{
        friend class HOGDescriptorSingle;
    private:
        ycv::YMat<float> gamma_lut;
        ycv::YMat<float> normal_lut;

    public:
        HOGStaticCache();
        virtual ~HOGStaticCache();
    };

    class HOGCache{
        friend class HOGDescriptorSingle;
    private:

        // Causes
        bool gamma_correction;
        int width;
        int height;
        int channels;

        // Results
        ycv::YMat<int> _xmap;
        ycv::YMat<int> _ymap;
        int* xmap;
        int* ymap;
        ycv::YMat<float> dbuf;
        float* lut;
        YMat_ gradients;

        // Causes
        bool signed_gradient;
        int num_bins;

        // Causes
        int cell_size;
        int block_size;
        int descriptor_size_width;
        int descriptor_size_height;

        // Results
        float angle_scale;
        int cell_width;
        int cell_height;
        int block_width;
        int block_height;
        int block_width_in_descriptor;
        int block_height_in_descriptor;
        int descriptor_width;
        int descriptor_height;
        int max_width_interest;
        int max_height_interest;
        ycv::YMat<char> orientation;
        ycv::YMat<float> magnitude;
        ycv::YMat<float> histograms; //cell histogram
        ycv::YMat<float> blocks; //cell histogram
        ycv::YMat<float> descriptor; //cell histogram

        //Data
        ycv::YMat<float> support_vector;
        float rho;

        float scale_factor;

        const float l2_hys_threshold = 2.0000000000000001e-01;

    public:
        HOGCache();
        virtual ~HOGCache();

    };

    static HOGStaticCache static_cache;

    HOGCache cache;
    YMat_ image;

public:
    HOGDescriptorSingle();
    virtual ~HOGDescriptorSingle();

    void Initialize(    const int width
                    , const int height
                    , const int channels
                    , const float scale_factor=1.0f
                    , const bool gamma_correction=true
                    , const bool signed_gradient=false
                    , const int num_bins=9
                    , const int cell_size=8
                    , const int block_size=2
                    , const int descriptor_size_width=6
                    , const int descriptor_size_height=12);
    void SetImage(YMat_& image);
    YRectList Detect();
    void MakeDescriptorForTrain();
    void MakeHistogram();
    void MakeNormalizedBlocks();
    void NormalizeBlock(float* const hist, int const block_length);
    void MakeDescriptor(int cell_x_idx, int cell_y_idx);
    int GetDescriptorLength() const;
    float* GetDescriptor() const;
    void SetSupportVector(int length, float* ptr, float rho);
    int GetSupportVectorLength() const;
    float* GetSupportVector() const;
    bool Predict();
    void SaveDescriptor();
    void SaveSupportVector();
    void SetHitThreahold(float hit_threshold);
    inline float* GetHistogram(int cell_x_idx, int cell_y_idx) const;
    inline float GetScaleFactor();
};

class HOGDescriptor{
    typedef ycv::YMat<unsigned char> YMat_;
    typedef std::vector<ycv::YRect> YRectList;
private:
    int width;
    int height;
    int channels;
    float scale_factor;
    int nlevels;
    bool gamma_correction;
    bool signed_gradient;
    int num_bins;
    int cell_size;
    int block_size;
    int descriptor_size_width;
    int descriptor_size_height;

    ycv::YMat<float> support_vector;
    float rho;

    HOGDescriptorSingle* hog_descriptors;

public:
    HOGDescriptor();
    virtual ~HOGDescriptor();

    YRectList DetectMultiScale(YMat_& image);

    HOGDescriptor* Initialize();
    HOGDescriptor* SetWidth(int value);
    HOGDescriptor* SetHeight(int value);
    HOGDescriptor* SetChannels(int value);
    HOGDescriptor* SetScaleFactor(float value);
    HOGDescriptor* SetNlevels(int value);
    HOGDescriptor* SetGammaCorrection(bool value);
    HOGDescriptor* SetSignedGradient(bool value);
    HOGDescriptor* SetNumBins(int value);
    HOGDescriptor* SetCellSize(int value);
    HOGDescriptor* SetBlockSize(int value);
    HOGDescriptor* SetDescriptorSizeWidth(int value);
    HOGDescriptor* SetDescriptorSizeHeight(int value);
    HOGDescriptor* SetHitThreshold(float value);
    HOGDescriptor* SetSupportVector(int length, float* ptr, float rho);

};
