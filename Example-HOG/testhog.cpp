#include "testhog.h"

TestHog::TestHog()
{

}



void TestHog::LoadTrainData()
{
    std::vector<ZIP_FILENAME> train_data_file_names;
    QFile train_pos_file_list("/home/yildbs/Data/INRIA/Train_160x96/pos.lst");
    if(train_pos_file_list.open(QIODevice::ReadOnly)){
        for(;;){
            auto line = train_pos_file_list.readLine().split('\n')[0].toStdString();
            if(line.empty()){
                break;
            }
            train_data_file_names.push_back({line, 1});
        }
    }
    QFile train_neg_file_list("/home/yildbs/Data/INRIA/Train_160x96/neg.lst");
    if(train_neg_file_list.open(QIODevice::ReadOnly)){
        for(;;){
            auto line = train_neg_file_list.readLine().split('\n')[0].toStdString();
            if(line.empty()){
                break;
            }
            train_data_file_names.push_back({line, 0});
        }
    }

    auto engine = std::default_random_engine{};
    std::shuffle(std::begin(train_data_file_names), std::end(train_data_file_names), engine);


    for( auto zip_filename : train_data_file_names ){
        std::string filename = zip_filename.filename;
        int label = zip_filename.label;

        cv::Mat temp = cv::imread("/home/yildbs/Data/INRIA/Train_160x96/" + filename);

        train_data.push_back({temp, label});
    }
}

void TestHog::InitHogDescriptor()
{
    cv::Mat temp = this->train_data.at(0).image;
    int width = temp.cols;
    int height = temp.rows;
    int channels = temp.channels();

    hog.Initialize(     width
                    , height
                    , channels
                    , true
                    , false
                    , 9
                    , 8
                    , 2
                    , 12
                    , 20 );
}

void TestHog::TrainUsingSVM()
{

    // Train the SVM
    this->svm = cv::ml::SVM::create();
    this->svm->setType(cv::ml::SVM::C_SVC);

    this->svm->setKernel(cv::ml::SVM::LINEAR);
    //this->svm->setKernel(cv::ml::SVM::RBF);
    //this->svm->setKernel(cv::ml::SVM::POLY);
    //this->svm->setKernel(cv::ml::SVM::SIGMOID);
    //this->svm->setKernel(cv::ml::SVM::CHI2);
    //this->svm->setKernel(cv::ml::SVM::INTER);

    this->svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

    bool first = true;
    cv::Mat trainingDataMat;
    cv::Mat labelsMat;

    int image_idx = 0;
    int dropped = 0;
    ycv::YMat<unsigned char> yimage;
    for(auto zip_image : train_data){
        //std::cout << "label: " << image.label << std::endl;

        cv::Mat image = zip_image.image;
        int label = zip_image.label;

        if( image.empty() ){
            printf("image is empty\n");
            dropped++;
            continue;
        }

        yimage = ycv::YMat<unsigned char>(image.cols, image.rows, image.channels(), image.data);
        hog.SetImage(yimage);

        hog.MakeDescriptorForTrain();
        int length = hog.GetDescriptorLength();
        float* ptr = hog.GetDescriptor();

        if( first ){
            first = false;
            trainingDataMat = cv::Mat(1,length,CV_32F);
            ::memcpy(trainingDataMat.data, ptr, sizeof(float)*length);
            labelsMat = cv::Mat(1,1,CV_32S, &label);
            ::memcpy(labelsMat.data, &label, sizeof(int));
        }else{
            cv::Mat temp(1,length,CV_32F);
            ::memcpy(temp.data, ptr, sizeof(float)*length);
            trainingDataMat.push_back(temp);

            cv::Mat temp2(1,1,CV_32S);
            ::memcpy(temp2.data, &label, sizeof(int));
            labelsMat.push_back(temp2);
        }
        std::cout << "idx: " << image_idx++ << std::endl;
    }

    this->svm->train(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);

    cv::Mat sv = this->svm->getSupportVectors();
    cv::Mat df_alpha, df_index;
    float rho = this->svm->getDecisionFunction(0, df_alpha, df_index);

    hog.SetSupportVector(hog.GetDescriptorLength(), (float*)sv.data, rho);
}

extern int cell_size_160x96;
extern int length_160x96;
extern float rho_160x96;
extern float support_vector_160x96[];
void TestHog::SetHogDescriptorSupportVector()
{
    this->hog.SetSupportVector(length_160x96, support_vector_160x96, rho_160x96);
}

void TestHog::TestUsingSVM()
{
    //TEST
    int correct = 0;
    int image_idx = 0;
    ycv::YMat<unsigned char> yimage;

    cv::Mat sv = this->svm->getSupportVectors();
    cv::Mat df_alpha, df_index;
    float rho = this->svm->getDecisionFunction(0, df_alpha, df_index);

    for(auto zip_image : train_data){
        cv::Mat image = zip_image.image;
        int label = zip_image.label;
        if( image.empty() ){
            printf("image is empty\n");
            continue;
        }
        yimage = ycv::YMat<unsigned char>(image.cols, image.rows, image.channels(), image.data);
        hog.SetImage(yimage);
        hog.Detect();

        int length = hog.GetDescriptorLength();
        float* ptr = hog.GetDescriptor();

        cv::Mat test_mat(1,length,CV_32F);
        ::memcpy(test_mat.data, ptr, sizeof(float)*length);

        float value = this->svm->predict(test_mat);

        float sum = -rho;

        float result = 0;
        for(int i=0;i<length;i++){
            result += ptr[i] * ((float*)sv.data)[i];
        }

        int my_label;
        sum += result;
        if( sum > 0 ){
            my_label = 0;
        }else{
            my_label = 1;
        }

        printf("%d. %d: %d, %d\n", image_idx++, label, (int)value, my_label);
        if( label == (int) value ) correct++;
    }

    printf("accuracy : %d / %d, %2.1f %%\n", correct, image_idx, (float)correct/image_idx*100.);
}

void TestHog::TestUsingHOGDescriptor()
{
    int correct = 0;
    int image_idx = 0;
    ycv::YMat<unsigned char> yimage;

    for(auto zip_image : train_data){
        cv::Mat image = zip_image.image;
        int label = zip_image.label;
        if( image.empty() ){
            printf("image is empty\n");
            continue;
        }
        yimage = ycv::YMat<unsigned char>(image.cols, image.rows, image.channels(), image.data);
        this->hog.SetImage(yimage);
        this->hog.Detect();

        int value = 0;
        if(this->hog.Predict()){
            value = 1;
        }else{
            value = 0;
        }

        printf("%d. %d: %d\n", image_idx++, label, (int)value);
        if( label == (int) value ) correct++;
    }

    printf("accuracy : %d / %d, %2.1f %%\n", correct, image_idx, (float)correct/image_idx*100.);
}
