#include <iostream>
#include <opencv2/opencv.hpp>
#include "fpdw_detector.h"

#define WIN32_LEAN_AND_MEAN  

#include <windows.h>

#include <stdio.h>
#include <tchar.h>


int main(int argc, char **argv)
{
	char szPath[MAX_PATH] = { 0 };
	if (GetModuleFileName(NULL, szPath, MAX_PATH))
	{
		(_tcsrchr(szPath, _T('\\')))[1] = 0;
	}
	std::string strPath = szPath;
	
	cv::Mat image = cv::imread((strPath + "\\" + argv[2]).c_str());
	fpdw::detector::FPDWDetector detector((strPath + "\\" + argv[1]).c_str(), 0.0f);
    
    detector.process(image);
    std::vector<cv::Rect> rect = detector.getBBoxes();

    for(const auto &i : rect)
    {
        cv::rectangle(image, i, cv::Scalar(255, 0, 0), 2);
    }
    
    cv::imshow("image", image);
    cv::waitKey(0);
    return 0;
}
