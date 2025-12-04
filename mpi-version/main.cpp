#include <mpi.h>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <time.h>

#define RADIUS 1

using namespace cv;

void apply_inverse(unsigned char *my_frame, int frame_width, int my_work)
{
	int index = 0;

	for (int i = 0; i < my_work; i++)
	{
		for (int j = 0; j < frame_width; j++)
		{
			my_frame[index] = 255 - my_frame[index];
			index++;
			my_frame[index] = 255 - my_frame[index];
			index++;
			my_frame[index] = 255 - my_frame[index];
			index++;
		}
	}
}

int main(int argc, char **argv)
{
	time_t start_time = time(NULL);

	std::string input = "demo.mp4";
	std::string output = "out.mp4";

	MPI_Init(&argc, &argv);
	int rank, num_process, my_work = 0, frame_width = 0, frame_height = 0;
	VideoCapture cap;
	Mat frame;
	VideoWriter writer;
	int totalFrames;
	unsigned char *my_frame = nullptr, *full_frame = nullptr;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);

	if (rank == 0)
	{
		cap.open(input);
		frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
		frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
		my_work = frame_height / num_process;
		totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

		full_frame = (unsigned char *)malloc(frame_width * frame_height * 3);

		int fourcc = VideoWriter::fourcc('a', 'v', 'c', '1');
		double fps = cap.get(CAP_PROP_FPS);
		writer.open(output, fourcc, fps, Size(frame_width, frame_height), true);
	}

	MPI_Bcast(&my_work, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&frame_width, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&totalFrames, 1, MPI_INT, 0, MPI_COMM_WORLD);

	my_frame = (unsigned char *)malloc(my_work * frame_width * 3);

	for (int i = 0; i < totalFrames; i++)
	{
		if (rank == 0)
		{
			cap.read(frame);
		}

		MPI_Scatter(frame.data, my_work * frame_width * 3, MPI_UNSIGNED_CHAR, my_frame, my_work * frame_width * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
		apply_inverse(my_frame, frame_width, my_work);
		MPI_Gather(my_frame, my_work * frame_width * 3, MPI_UNSIGNED_CHAR, full_frame, my_work * frame_width * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

		if (rank == 0)
		{
			Mat processed_frame(frame_height, frame_width, CV_8UC3, full_frame);
			writer.write(processed_frame);
		}
	}

	if (rank == 0)
	{

		free(full_frame);
	}
	free(my_frame);
	time_t end_time = time(NULL);

	if(rank == 0)
			printf("Time Elapsed: %ld secs\n", end_time - start_time);
	
	MPI_Finalize();




	return 0;
}
