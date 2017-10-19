//
// Created by buyi on 17-10-17.
//

#include <string.h>
#include <svo/global.h>
#include <svo/config.h>
#include <svo/frame.h>
#include <svo/feature_detection.h>
#include <svo/depth_filter.h>
#include <svo/feature.h>
#include <vikit/timer.h>
#include <vikit/vision.h>
#include <vikit/abstract_camera.h>
#include <vikit/atan_camera.h>
#include "test_utils.h"

#include <fstream>

using namespace std;

void LoadImages(const string &strImageFilename, vector<string> &vstrImageFilenamesRGB, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strImageFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        //! read the first three lines of txt file
        getline(fAssociation,s);
        getline(fAssociation,s);
        getline(fAssociation,s);
        getline(fAssociation,s);

        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            //vstrImageFilenamesD.push_back(sD);

        }
    }
}

int main(int argc, char **argv)
{

    vk::AbstractCamera* cam = new vk::ATANCamera(640, 480, 0.511496, 0.802603, 0.530199, 0.496011, 0.934092);

    vector<double> dTimestamps;
    vector<string> dImageNames;
    string Datasets_Dir = "/home/buyi/Datasets/longhouse";
    string strImageFile = Datasets_Dir + "/rgb.txt";
    LoadImages(strImageFile, dImageNames, dTimestamps);

    int nImages = dImageNames.size();

    cv::Mat Image, Image_tmp;

    double start = static_cast<double>(cvGetTickCount());
    for (int i = 0; i < nImages; ++i)
    {
        string img_path = Datasets_Dir + "/" + dImageNames[i];
        Image = cv::imread(img_path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(Image, Image_tmp);

        svo::FramePtr frame(new svo::Frame(cam, Image_tmp, 0.0));
        svo::Features fts;
        svo::feature_detection::FastDetector fast_detector(
                Image_tmp.cols, Image_tmp.rows, svo::Config::gridSize(), svo::Config::nPyrLevels());
        fast_detector.detect(frame.get(), frame->img_pyr_, svo::Config::triangMinCornerScore(), fts);


        cv::Mat Image_new = Image_tmp.clone();
        if(Image_new.channels() < 3)
            cv::cvtColor(Image_new, Image_new, CV_GRAY2BGR);
        for_each(fts.begin(), fts.end(), [&](svo::Feature* feature)
        {
            cv::rectangle(Image_new,
                          cv::Point2f(feature->px[0] - 2, feature->px[1] - 2),
                          cv::Point2f(feature->px[0] + 2, feature->px[1] + 2),
                          cv::Scalar (0, 255, 0));
        });

        cv::namedWindow("Feature_Detect");
        cv::imshow("Feature_Detect", Image_new);
        cv::waitKey(1);

    }
    double time = ((double)cvGetTickCount() - start) / cvGetTickFrequency();
    cout << time/647 << "us" << endl;
    return  0;
}