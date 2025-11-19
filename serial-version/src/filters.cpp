#include "../includes/filter.hpp"
using namespace cv;

int apply_invert(cv::VideoCapture cap, cv::VideoWriter out)
{
    Mat frame;
    while (cap.read(frame))
    {

        for (int i = 0; i < frame.rows; i++)
        {
            for (int j = 0; j < frame.cols; j++)
            {
                Vec3b &pix = frame.at<Vec3b>(i, j);
                pix[0] = 255 - pix[0];
                pix[1] = 255 - pix[1];
                pix[2] = 255 - pix[2];
            }
        }

        out.write(frame);
    }

    cap.release();
    out.release();
    return 0;
}

int apply_grayscale(cv::VideoCapture cap, cv::VideoWriter out){
    Mat frame;
    while (cap.read(frame))
    {

        for (int i = 0; i < frame.rows; i++)
        {
            for (int j = 0; j < frame.cols; j++)
            {
                Vec3b &pix = frame.at<Vec3b>(i, j);
                unsigned int B = pix[0];
                unsigned int G = pix[1];
                unsigned int R = pix[2];

                float L = 0.299f * R + 0.587f * G + 0.114f * B;

                pix[0] = L;
                pix[1] = L;
                pix[2] = L;

            }
        }

        out.write(frame);
    }

    cap.release();
    out.release();
    return 0;
}