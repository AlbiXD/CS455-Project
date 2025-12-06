#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include "../includes/filter.hpp"
#include <string.h>
#include "../includes/filter_cuda.hpp"
#define NODES 2


using namespace cv;


int main(int argc, char **argv)
{
	time_t start_time, end_time;

	MPI_Init(&argc, &argv);

	int rank, num_procs, totalFrames = 0, my_frames = 0, remainder = 0, choice = 0, cuda_Flag = 0;
	char input_path[64];
	std::string output_path = "";
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	std::string part = "part_" + std::to_string(rank) + ".mp4";
	FILE *f = NULL;
	unsigned char *buf = NULL;
	long file_size = 0;

	if(rank == 0){
		start_time = time(NULL);

		std::string input_file = "";
		std::cout << "Please enter input name (include extension): ";
		std::string test = "";
		std::cin >> input_file;
		test = "../input-videos/" + input_file;
		strcpy(input_path, test.c_str());
		std::string output_file = "";
		std::cout << "Please enter output name (include extension): ";
		std::cin >> output_file;
		
		output_path = "../output-videos/" + output_file;

		std::cout << "Apply Filter\n1) \tGrayscale \n2) \tBlur Effect\n3) \tInvert Filter\n4) \tEdge Detection\n";
		std::cout << "Choose a filter: ";
		std::cin >> choice;
		std::cout << "Do you want to use CUDA? ";
		std::string cuda_Choice = "";
		std::cin >> cuda_Choice;
		if("yes" == cuda_Choice){
			cuda_Flag = 1;
		}
		if(cuda_Flag){std::cout << "You are using CUDA!" << std::endl;}
		else{std::cout << "You are NOT using CUDA :(" << std::endl;}

	}

	
	VideoCapture cap;

	if ((num_procs) % NODES != 0)
	{
		printf("Must be multiple of %d\n", NODES);
		return -1;
	}

	if (rank == 0)
	{
		cap.open(input_path);
		totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

		f = fopen(input_path, "rb");
		fseek(f, 0, SEEK_END);
		file_size = ftell(f);
		printf("filesize = %ld\n", file_size);
		fseek(f, 0, SEEK_SET);
	}

	// Broadcast File Size
	MPI_Bcast(&file_size, 1, MPI_LONG, 0, MPI_COMM_WORLD);
	MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&totalFrames, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cuda_Flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

	
	MPI_Bcast(input_path, 64, MPI_CHAR, 0, MPI_COMM_WORLD);


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
		FILE *out = fopen(input_path, "wb");
		fwrite(buf, 1, file_size, out);
		fclose(out);

	}

	MPI_Barrier(MPI_COMM_WORLD);

	printf("Video Recieved all processes ready to go\n");

	cap.open(input_path);
	cap.set(CAP_PROP_POS_FRAMES, start);
	int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
	//Size is the number of values- to get it in bytes is more
	int size = frame_width * frame_height * 3;
	unsigned char * pixels = (unsigned char *) malloc(size);

	Mat frame;
	VideoWriter writer;
	int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
	double fps = cap.get(CAP_PROP_FPS);

	writer.open(part, fourcc, fps, Size(frame_width, frame_height), true);


	//We need to alloc outside the loop
	//unsigned char* d_frame;
	//size_t size;
	//if(cuda_Flag){
		//cudaMalloc(&d_frame, frame_width * frame_height *3);

	//}else{
	for(int i = start; i < end; i++){

		if(!cap.read(frame)){
			break;
		}
//std::cout << "frame.step: " << frame.step
//         << " expected: " << frame_width * 3 << std::endl;
		memcpy(pixels, frame.data, frame_width * frame_height * 3);


		/*Apply filter is the serial version, so we do the cuda for the cuda flag*/
		if(cuda_Flag){
			//printf("before CUDA: %d %d %d\n", pixels[0], pixels[1], pixels[2]);
			process_frame_cuda(choice, pixels, frame_width, frame_height);
			//printf("After CUDA: %d %d %d\n", pixels[0], pixels[1], pixels[2]);
		}
		else{
			if(apply_filter(choice, pixels, frame_width, frame_height) < -1) exit(-1);

		}

		Mat processed_frame(frame_height, frame_width, CV_8UC3, pixels);
		writer.write(processed_frame);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if(rank == 0){
		time_t final_compute = time(NULL);
		printf("Finished Processing All Frames %lds\n", final_compute-start_time);
	}
	writer.release();
	cap.release();

	//All ranks send to rank buffer

	unsigned char * raw_part = NULL;
	if(rank != 0){
		f = fopen(part.c_str(), "rb");
		fseek(f, 0, SEEK_END);
		file_size = ftell(f);
		printf("filesize = %ld\n", file_size);
		fseek(f, 0, SEEK_SET);
		raw_part = (unsigned char*) malloc(file_size);
		int r = fread(raw_part, 1, file_size, f);
		printf("Successfully read %d\n", r);
		MPI_Send(&file_size, 1, MPI_LONG, 0, 3, MPI_COMM_WORLD);
		MPI_Send(raw_part, file_size, MPI_UNSIGNED_CHAR, 0, 4, MPI_COMM_WORLD);
	}

	long other_size = 0;
	if(rank == 0){
		for(int i = 1; i < num_procs; i++){
			MPI_Recv(&other_size, 1, MPI_LONG, i, 3, MPI_COMM_WORLD, &status);
			printf("Recieved total file size for Rank %d\n", i);
			raw_part = (unsigned char *) malloc(other_size);
			MPI_Recv(raw_part, other_size, MPI_UNSIGNED_CHAR, i, 4, MPI_COMM_WORLD, &status);
			printf("Recieved in buffer for Rank %d\n", i);
			std::string output = "part_" + std::to_string(i) + ".mp4";
			FILE *out = fopen(output.c_str(), "wb");
			fwrite(raw_part, 1, other_size, out);
			fclose(out);
			free(raw_part);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if(rank == 0){
		std::string inputs = "";
		FILE* l = fopen("lists.txt", "w");
		fprintf(l, "ffconcat version 1.0\n");
		for(int i = 0; i < num_procs; i++){
			fprintf(l, "file 'part_%d.mp4'\n", i);///home/gpgpup/project/CS455-Project/mpi-version/

		}
		fclose(l);

		pid_t pid = fork();
		if (pid == 0) {
			execlp("ffmpeg","ffmpeg",
					"-y",
					"-loglevel","error",
					"-stats",
					"-f","concat",
					"-safe","0",
					"-i","lists.txt",
					"-c","copy",
					output_path.c_str(),
					(char*)NULL);
			_exit(1);
		}

			waitpid(pid,nullptr,0);

		end_time = time(NULL);

		printf("%ld\n", end_time-start_time);

	}

	char path[64];
/*
	if(rank == 0){
		remove("lists.txt");
		for(int i = 0; i < num_procs; i++){
			sprintf(path, "part_%d.mp4", i);
			remove(path);

		}
	}

	if(rank == 1){
		for(int i = 0; i < num_procs; i++){
			if(i%2 != 0){
			sprintf(path, "part_%d.mp4", i);
			remove(path);
			}

		}
	}*/



	MPI_Finalize();

	return 0;

}
