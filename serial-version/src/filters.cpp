#include "../includes/filter.hpp"
using namespace cv;
#define FILTER_SIZE 3
#define RADIUS 1

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

    int edgeX[3][3] ={
        {-1,0,1},
        {-2,0,2},
        {-1,0,1}
    };
    int edgeY[3][3] ={
        {-1,-2,-1},
        { 0, 0, 0},
        { 1, 2, 1}
    };

int apply_edge(cv::VideoCapture cap, cv::VideoWriter out)
{
    Mat frame;


    while (cap.read(frame))
    {
        Mat original = frame.clone();
        int sumR, sumG, sumB;
        int rX, rY, gX, gY, bX, bY, edgeXH, edgeYH;
        //Vec3b& conv;
        for (int i = 1; i < frame.rows-1; i++)
        {
            for (int j = 1; j < frame.cols-1; j++)
            {
                Vec3b& pix = frame.at<Vec3b>(i, j);
                //sumR = 0; sumG = 0; sumB = 0, yTemp = 0;
                rX = rY = 0;
                gX = gY = 0;
                bX = bY = 0;

                for(int fx = -1; fx < 2; fx++){
                    for(int fy = -1; fy < 2; fy++){
                        Vec3b& conv = original.at<Vec3b>(fx+i, fy+j);

                        edgeXH = edgeX[fx+1][fy+1];
                        edgeYH = edgeY[fx+1][fy+1];

                        bX += conv[0] * edgeXH; bY += conv[0] * edgeYH;
                        gX += conv[1] * edgeXH; gY += conv[1] * edgeYH;
                        rX += conv[2] * edgeXH; rY += conv[2] * edgeYH;

                    }
                }

                sumR = sqrt(rX * rX + rY * rY);
                sumB = sqrt(bX * bX + bY * bY);
                sumG = sqrt(gX * gX + gY * gY);

                pix[0] = cv::saturate_cast<uchar>(sumB);
                pix[2] = cv::saturate_cast<uchar>(sumR);
                pix[1] = cv::saturate_cast<uchar>(sumG);


            }
        }

        out.write(frame);
    }

    cap.release();
    out.release();
    return 0;
}

                /*
                //pix is the pixel, 0 is red, 1 is blue, and 2 is green?
                //We're producing a weighted sum
                sumR = 0; sumG = 0; sumB = 0;
                for (int filterX = i - RADIUS; filterX < RADIUS+1+i; filterX++) {
                    for (int filterY = j - RADIUS; filterY < RADIUS+1+j; filterY++) {
                         if (filterX >= 0 && filterX < frame.rows && filterY >= 0 && filterY < frame.cols) {
                            Vec3b& conv = frame.at<Vec3b>(filterX, filterY);

                            //std::cout << sumR;std::cout << sumG;std::cout << sumB;

                        }
                    }
                }
                sumG = sqrt(sumG);
                sumB = sqrt(sumB);
                sumR = sqrt(sumR);
                //instead of blurring, we take the weights?
                pix[0] = cv::saturate_cast<uchar>(sumB / 9);
                pix[2] = cv::saturate_cast<uchar>(sumR / 9);
                pix[1] = cv::saturate_cast<uchar>(sumG / 9);*/
