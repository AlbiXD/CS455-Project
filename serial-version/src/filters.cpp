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
#define FILTER_SIZE 3
#define RADIUS 1
//This means its a radius 1

int apply_blur(cv::VideoCapture cap, cv::VideoWriter out) {
    Mat frame;
    while (cap.read(frame))
    {
        int sumR, sumG, sumB;
        //Vec3b& conv;
        for (int i = 0; i < frame.rows; i++)
        {
            for (int j = 0; j < frame.cols; j++)
            {
                Vec3b& pix = frame.at<Vec3b>(i, j);
                sumR = 0; sumG = 0; sumB = 0;

                //pix is the pixel, 0 is red, 1 is blue, and 2 is green?
                //We're producing a weighted sum
                sumR = 0; sumG = 0; sumB = 0;
                for (int filterX = i - RADIUS; filterX < RADIUS+1+i; filterX++) {
                    for (int filterY = j - RADIUS; filterY < RADIUS+1+j; filterY++) {
                         if (filterX >= 0 && filterX < frame.rows && filterY >= 0 && filterY < frame.cols) {
                            Vec3b& conv = frame.at<Vec3b>(filterX, filterY);
                            sumB += conv[0];
                            sumG += conv[1];
                            sumR += conv[2];
                            //std::cout << sumR;std::cout << sumG;std::cout << sumB;

                        }
                    }
                }
                //Dividing each color average is badddd i think

                pix[0] = cv::saturate_cast<uchar>(sumB / 9);
                pix[2] = cv::saturate_cast<uchar>(sumR / 9);
                pix[1] = cv::saturate_cast<uchar>(sumG / 9);
            }
        }

        out.write(frame);
    }

    cap.release();
    out.release();
    return 0;
}
