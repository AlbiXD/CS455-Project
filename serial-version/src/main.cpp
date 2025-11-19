#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <unistd.h>
#include <ostream>

using namespace std;

int main()
{

	while (1)
	{
		string input_path;
		string choice;

		cout << "Please enter path for the video: ";
		cin >> input_path;

		cv::VideoCapture cap(input_path);

		if (!cap.isOpened())
		{
			printf("Could not open video file\n");
			return -1;
		}

		cout << "Apply Filter\n1) \tGrayscale \n2) \tBlur Effect\n";
		cin >> choice;

		cv::Mat frame;

		bool ret;

		while ((ret = cap.read(frame)))
		{
			if (ret)
			{
				// Display the frame using imshow
				cv::imshow("First Frame", frame);
				cv::waitKey(0);			 // Wait for a key press to close the window
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
	}

	return 0;
}
