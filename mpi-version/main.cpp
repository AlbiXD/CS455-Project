#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdlib.h>

#define NODES 2


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

	MPI_Init(&argc, &argv);

	int rank, num_procs, totalFrames = 0, my_frames = 0, remainder = 0;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	FILE *f = NULL;
	unsigned char *buf = NULL;
	long file_size = 0;

	VideoCapture cap;

	if ((num_procs) % NODES != 0)
	{
		printf("Must be multiple of %d\n", NODES);
		return -1;
	}

	if (rank == 0)
	{
		cap.open("input.mp4");
		totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

		f = fopen("input.mp4", "rb");
		fseek(f, 0, SEEK_END);
		file_size = ftell(f);
		printf("filesize = %ld\n", file_size);
		fseek(f, 0, SEEK_SET);
	}

	// Broadcast File Size
	MPI_Bcast(&file_size, 1, MPI_LONG, 0, MPI_COMM_WORLD);
	MPI_Bcast(&totalFrames, 1, MPI_INT, 0, MPI_COMM_WORLD);


	int frames_per_rank = totalFrames / num_procs;
	int start = rank * frames_per_rank;
	int end = (rank == num_procs - 1)  ? totalFrames : start + frames_per_rank;

	buf = (unsigned char *)malloc(file_size);
	printf("filesize = %ld, Rank %d\n", file_size, rank);

	if (rank == 0)
	{
		int r = fread(buf, 1, file_size, f);
		printf("Successfully read %d\n", r);
		printf("Frames: %d\n", totalFrames);
		MPI_Send(buf, file_size, MPI_UNSIGNED_CHAR, 1, 123, MPI_COMM_WORLD);
	}

	if (rank == 1)
	{
		MPI_Recv(buf, file_size, MPI_UNSIGNED_CHAR, 0, 123, MPI_COMM_WORLD, &status);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	printf("Video Recieved all processes ready to go\n");

	// Video Broadcasted

	if (rank == 1)
	{
		FILE *out = fopen("input.mp4", "wb");
		fwrite(buf, 1, file_size, out);
		fclose(out);
	}

		cap.open("input.mp4");
		cap.set(CAP_PROP_POS_FRAMES, start);
		int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
		int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
		unsigned char * pixels = (unsigned char *) malloc(frame_width * frame_height * 3);

		Mat frame;
		VideoWriter writer;
		int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
		double fps = cap.get(CAP_PROP_FPS);

		writer.open("part_" + std::to_string(rank) + ".mp4", fourcc, fps, Size(frame_width, frame_height), true);


		for(int i = start; i < end; i++){
			cap.read(frame);
			pixels = frame.data;
			apply_inverse(pixels, frame_width, frame_height);
			Mat processed_frame(frame_height, frame_width, CV_8UC3, pixels);
			writer.write(processed_frame);
		}

		writer.release();
		cap.release();


		if(rank == 0){
			FILE* open 
			for(int i = 1; i < size; i++){

			}
		}

	MPI_Finalize();

	return 0;

}
