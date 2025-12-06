#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <unistd.h>
#include <ostream>
#include <time.h>
#include "../includes/filter.hpp"
using namespace std;
using namespace cv;

int main()
{

	string input_path;
	string output_path = "../output-videos";
	string output_filename;

	cout << "Please enter path for the video: ";
	cin >> input_path;
	//input_path = "../input-videos/greyscale.mp4";
	cout << input_path << endl;

	cout << "Please enter video name for the output: ";
	cin >> output_filename;

	cv::VideoCapture cap(input_path);

	if (!cap.isOpened())
	{
		printf("Could not open video file\n");
		return -1;
	}
	int choice;

	cout << "Apply Filter\n1) \tGrayscale \n2) \tBlur Effect\n3) \tInvert Filter\n4) \tEdge Detection\n";
	cout << "Choose a filter: ";
	cin >> choice;

	int width = (int)cap.get(CAP_PROP_FRAME_WIDTH);
	int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
	double fps = cap.get(CAP_PROP_FPS);
	std::string pipeline = "appsrc ! videoconvert ! x264enc bitrate=2000 speed-preset=ultrafast "
	"! mp4mux ! filesink location=" + output_path + "/" + output_filename;
	cv::VideoWriter out(pipeline, cv::CAP_GSTREAMER, 0, fps, cv::Size(width, height), true);

	time_t start_time = time(NULL);
	switch(choice){
		case 1:
			cout << "Applying Grayscale...\n";
			apply_grayscale(cap, out);
			break;
		case 2:
			cout << "Applying Blur...\n";
			apply_blur(cap, out);
			break;
		case 3:
			cout << "Applying Inverting Filter...\n";
			apply_invert(cap, out);
			break;

		case 4:
			cout << "Applying Edge Detection Filter...\n";
			apply_edge(cap, out);
			break;
		default:
			printf("Invalid Option");
			break;
	}

	// Release the video capture object
	time_t end_time = time(NULL);

	printf("Time Elapsed: %ld secs\n", end_time - start_time);

	return 0;
}
