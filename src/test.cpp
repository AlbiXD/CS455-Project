#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <unistd.h>

int main()
{

    cv::VideoCapture cap("./demo.mp4");

    if (!cap.isOpened())
    {
        printf("Could not open video file\n");
        return -1;
    }

    cv::Mat frame;

    bool ret;

    while ((ret = cap.read(frame)))
    {
        if (ret)
        {
            // Display the frame using imshow
            cv::imshow("First Frame", frame);
            cv::waitKey(0);          // Wait for a key press to close the window
            cv::destroyAllWindows(); // Close the window
        }
        else
        {
            std::cout << "Error: Could not read the frame." << std::endl;
        }
    }
    // Release the video capture object
    cap.release();
    return 0;

    return 0;
}