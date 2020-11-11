#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>


static int MAXITER;

#define XRANGE 3840
#define YRANGE 2160

#define XMIN -2.0
#define YMIN -1
#define XMAX 1
#define YMAX 1

uint32_t *image;

uint32_t escape(double x, double y) {
	double complex c= x + I * y;
	double complex z= 0.0;

	int iter=0;
	
	while(abs(z) < 2.0 && iter < MAXITER) {
		z = z * z + c;
		iter ++;
	}
	return iter;
}

int main(int argc, char **argv) {
	assert(argc == 2);
	MAXITER=atoi(argv[1]);
	image = malloc( XRANGE * YRANGE * sizeof(uint32_t) );

	for(int i=0; i < XRANGE; i++) 
		for(int j=0; j < YRANGE; j++) {
			double x = XMIN + i * (XMAX-XMIN)/(double)XRANGE;
			double y = YMIN + j * (YMAX-YMIN)/(double)YRANGE;
			image[i + j*XRANGE] = escape(x,y);
		}

	uint64_t somme=0;
	for(int i=0; i < XRANGE; i++) 
		for(int j=0; j < YRANGE; j++) {
			somme += image[i + j *XRANGE];
		}

	printf("somme %lld\n", somme);
}
