/**************************
** TDDD56 Lab 3
***************************
** Author:
** August Ernstsson
**************************/

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <iterator>

#include <skepu>

#include "support.h"


unsigned char median_kernel(skepu::Region2D<unsigned char> image, size_t elemPerPx)
{
	  //Instead of using theaverage pixel value of a region, the output is the median value.
	int index = 0;
	unsigned char tab[100000];

	// saves the pixel of the region in an tab to sort
	for (int y = -image.oi; y <= image.oi; ++y)
		for (int x = -image.oj; x <= image.oj; x += elemPerPx)
			tab[index++] = image(y,x);

	//The challenge therefore is to sort the pixels in the region in such a way that the median value can be identified
	// Quicksort

	int nb_pixels = (image.oj/elemPerPx*2+1)*(image.oi*2+1);
	unsigned char tmp1, tmp2;
	int p;


		// Create an auxiliary stack
		int l=0, h=nb_pixels;
		int stack[10000];
		// initialize top of stack
		int top = -1;
		// push initial values of l and h to stack
		stack[++top] = l;
		stack[++top] = h;
		// Keep popping from stack while is not empty
		while (top >= 0) {
			// Pop h and l
			h = stack[top--];
			l = stack[top--];
			// Set pivot element at its correct position
			// in sorted tab
			int x = tab[h];
			int i = (l - 1);
			for (int jpart = l; jpart <= h - 1; jpart++) {
				if (tab[jpart] <= x) {
					i++;
					tmp1 = tab[i];
					tab[i] = tab[jpart];
					tab[jpart] = tmp1;
				}
			}
			tmp2 = tab[i + 1];
			tab[i + 1] = tab[h];
			tab[h] = tmp2;
			p = (i + 1);
			// If there are elements on left side of pivot,
			// then push left side to stack
			if (p - 1 > l) {
				stack[++top] = l;
				stack[++top] = p - 1;
			}

			// If there are elements on right side of pivot,
			// then push right side to stack
			if (p + 1 < h) {
				stack[++top] = p + 1;
				stack[++top] = h;
			}
		}

    if(nb_pixels%2==0)
        return((tab[nb_pixels/2] + tab[nb_pixels/2 - 1]) / 2.0);
    else
        return tab[nb_pixels/2];


}



int main(int argc, char* argv[])
{
	LodePNGColorType colorType = LCT_RGB;

	if (argc < 5)
	{
		std::cout << "Usage: " << argv[0] << " input output radius [backend]\n";
		exit(1);
	}

	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	const int radius = atoi(argv[3]);
	auto spec = skepu::BackendSpec{argv[4]};
	skepu::setGlobalBackendSpec(spec);

	// Create the full path for writing the image.
	std::stringstream ss;
	ss << (2 * radius + 1) << "x" << (2 * radius + 1);
	std::string outputFileNamePad = outputFileName + ss.str() + "-median.png";

	// Read the padded image into a matrix. Create the output matrix without padding.
	ImageInfo imageInfo;
	skepu::Matrix<unsigned char> inputMatrix = ReadAndPadPngFileToMatrix(inputFileName, radius, colorType, imageInfo);
	skepu::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);

	// Skeleton instance
	auto calculateMedian = skepu::MapOverlap(median_kernel);
	calculateMedian.setOverlap(radius, radius  * imageInfo.elementsPerPixel);

	auto timeTaken = skepu::benchmark::measureExecTime([&]
	{
		calculateMedian(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
	});

	WritePngFileMatrix(outputMatrix, outputFileNamePad, colorType, imageInfo);

	std::cout << "Time: " << (timeTaken.count() / 10E6) << "\n";

	return 0;
}
