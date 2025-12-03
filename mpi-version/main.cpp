#include <mpi.h>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int rank, num_process;

	// Initialize Ranks
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_process);

	unsigned char *my_image;

	int my_work = 0;
	int col = 0;
	int row = 0;

	Mat img;

	if (rank == 0)
	{

		std::string image_path = samples::findFile("demo.jpg");
		img = imread(image_path, IMREAD_COLOR);

		cv::Size sz = img.size();
		col = sz.width;
		row = sz.height;

		my_work = row / num_process;
	}

	MPI_Bcast(&my_work, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&col, 1, MPI_INT, 0, MPI_COMM_WORLD);

	my_image = (unsigned char *)malloc(my_work * col * 3);

	// MPI_SCATTER IMAGE
	MPI_Scatter(img.data, my_work * col * 3, MPI_UNSIGNED_CHAR, my_image, my_work * col * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	Mat new_image(my_work, col, CV_8UC3, my_image);

	char window_name[50];
	snprintf(window_name, sizeof(window_name), "Rank: %d", rank);
	imshow(window_name, new_image);			
	waitKey(0);

	free(my_image);

	MPI_Finalize();

	return 0;
}