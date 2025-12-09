#include "../includes/filter.hpp"
#define FILTER_SIZE 3
#define RADIUS 1

using namespace std;
void apply_invert(unsigned char *my_frame, int frame_width, int frame_height)
{	
	int index = 0;

	for (int i = 0; i < frame_width; i++)
	{
		for (int j = 0; j < frame_height; j++)
		{
			my_frame[index] = 255 - my_frame[index];
			index++;
			my_frame[index] = 255 - my_frame[index];
			index++;
			my_frame[index] = 255 - my_frame[index];
			index++;
		}
	}

	return;
}

void apply_grayscale(unsigned char *my_frame, int frame_width, int frame_height)
{
	int index = 0;

	for (int i = 0; i < frame_width; i++)
	{
		for (int j = 0; j < frame_height; j++)
		{
			unsigned int B = my_frame[index];
			unsigned int G = my_frame[index + 1];
			unsigned int R = my_frame[index + 2];

			float L = 0.299f * R + 0.587f * G + 0.114f * B;
			unsigned char gray = static_cast<unsigned char>(L);

			my_frame[index]     = gray;
			my_frame[index + 1] = gray;
			my_frame[index + 2] = gray;

			index += 3;
		}
	}

	return;
}
//This means its a radius 1
void apply_blur(unsigned char *my_frame, int frame_width, int frame_height)
{
	const int bytes_per_pixel = 3;
	const int frame_size = frame_width * frame_height * bytes_per_pixel;

	// Copy original frame
	std::vector<unsigned char> original(frame_size);
	std::memcpy(original.data(), my_frame, frame_size);

	for (int x = 0; x < frame_width; x++)
	{
		for (int y = 0; y < frame_height; y++)
		{
			int sumR = 0, sumG = 0, sumB = 0;
			int count = 0;

			for (int filterX = x - RADIUS; filterX <= x + RADIUS; filterX++)
			{
				for (int filterY = y - RADIUS; filterY <= y + RADIUS; filterY++)
				{
					if (filterX >= 0 && filterX < frame_width &&
							filterY >= 0 && filterY < frame_height)
					{
						// correct indexing: (row = y, col = x)
						int nindex = (filterY * frame_width + filterX) * bytes_per_pixel;

						unsigned char b = original[nindex];
						unsigned char g = original[nindex + 1];
						unsigned char r = original[nindex + 2];

						sumB += b;
						sumG += g;
						sumR += r;
						count++;
					}
				}
			}

			int idx = (y * frame_width + x) * bytes_per_pixel;

			my_frame[idx]     = cv::saturate_cast<uchar>(sumB / count);
			my_frame[idx + 1] = cv::saturate_cast<uchar>(sumG / count);
			my_frame[idx + 2] = cv::saturate_cast<uchar>(sumR / count);
		}
	}
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

int apply_edge(unsigned char *my_frame, int frame_width, int frame_height)
{
	const int bytes_per_pixel = 3;
	const int frame_size = frame_width * frame_height * bytes_per_pixel;

	std::vector<unsigned char> original(frame_size);
	std::memcpy(original.data(), my_frame, frame_size);

	int sumR, sumG, sumB;
	int rX, rY, gX, gY, bX, bY, edgeXH, edgeYH;

	for (int y = 1; y < frame_height - 1; ++y) {
		for (int x = 1; x < frame_width - 1; ++x) {

			rX = rY = 0;

			for (int fx = -1; fx <= 1; ++fx) {
				for (int fy = -1; fy <= 1; ++fy) {

					int nx = x + fx;
					int ny = y + fy;

					int nindex = (ny * frame_width + nx) * bytes_per_pixel;
					//           ^^^      ^^^^^^^^^
					//           row      width

					unsigned char b = original[nindex + 0];
					unsigned char g = original[nindex + 1];
					unsigned char r = original[nindex + 2];

					int edgeXH = edgeX[fx+1][fy+1];
					int edgeYH = edgeY[fx+1][fy+1];

					rX += r * edgeXH;
					rY += r * edgeYH;
				}
			}

			int mag = (int)std::sqrt(rX*rX + rY*rY);
			unsigned char v = cv::saturate_cast<uchar>(mag * 0.25f);

			int idx = (y * frame_width + x) * bytes_per_pixel;
			my_frame[idx+0] = v;
			my_frame[idx+1] = v;
			my_frame[idx+2] = v;
		}
	}


	return 0;
}
int apply_filter(int choice, unsigned char *my_frame, int frame_width, int frame_height){
	int rval = 0;

	switch(choice){
		case 1:
			apply_grayscale(my_frame, frame_width, frame_height);
			break;
		case 2:
			apply_blur(my_frame, frame_width, frame_height);
			break;
		case 3:
			apply_invert(my_frame, frame_width, frame_height);
			break;

		case 4:
			apply_grayscale(my_frame, frame_width, frame_height);
			apply_edge(my_frame, frame_width, frame_height);
			break;
		default:
			printf("Invalid Option");
			rval = -1;
			break;
	}


	return rval;
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
