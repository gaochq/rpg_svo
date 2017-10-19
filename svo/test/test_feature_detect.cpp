//
// Created by buyi on 17-10-18.
//
//
// Created by buyi on 17-10-18.
//

#include <string.h>
#include <svo/frame.h>
#include <svo/feature_detection.h>
#include <svo/config.h>

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
    int a = 2;

    svo::Features features;
    cv::Mat img;
    svo::feature_detection::FastDetector fast_detector(img.cols, img.rows, svo::Config::gridSize(),
                                                       svo::Config::nPyrLevels());
    for (int i = 0; i < 10; ++i)
    {
        a++;
    }
    return  0;
}
