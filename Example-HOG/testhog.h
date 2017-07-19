#ifndef TESTHOG_H
#define TESTHOG_H

#include <iostream>

//For opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

//For Qt
#include <QFile>

#include "../module/hog.h"


class TestHog
{
public:
    typedef struct {
        cv::Mat image;
        int label;
    }ZIP_DATA;
    typedef struct{
        std::string filename;
        int label;
    }ZIP_FILENAME;
    TestHog();



    void LoadTrainData();
    void InitHogDescriptor();
    void TrainUsingSVM();
    void SetHogDescriptorSupportVector();
    void TestUsingSVM();
    void TestUsingHOGDescriptor();

public:
    HOGDescriptorSingle hog;
    std::vector<ZIP_DATA> train_data;

    cv::Ptr<cv::ml::SVM> svm;
};

#endif // TESTHOG_H
