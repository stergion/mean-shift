#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define VAR		1
#define EPSILON		0.0001
#define RESFILE		"./res_serial.txt"

typedef struct {
	int width;
	int height;
	double *elements;
} Matrix;


double gausian (double);
void duplicateMatrix(Matrix from, Matrix to);
void loadFile(char name[65], Matrix M);
void printMatrix(Matrix M);
int meanShift(Matrix Y, Matrix X);

int main(int argc, char **argv) {
	if(argc != 4) {
		printf("Invalid number of arguments. (%d)\n", argc-1);
		printf("You must enter three arguments.\n");
		printf("arg 1: filename.\n");
		printf("arg 2: number of points.\n");
		printf("arg 3: number of dimention of points.\n");
		exit(1);
	}
	int N, d;
	char *filename;

	filename = argv[1];
	N = atoi(argv[2]);	// Number of points
	d = atoi(argv[3]);	// Number of dimentions

	Matrix X, Y, YNew;
	X.height = N; X.width = d;
	X.elements = (double*)malloc(X.width * X.height * sizeof(double));
	Y.height = N; Y.width = d;
	Y.elements = (double*)malloc(Y.width * Y.height * sizeof(double));

	loadFile(filename, X);
	duplicateMatrix(X, Y);
	printf("done reading files...\n");

	int count = 0, conv;

	struct timeval startwtime, endwtime;
	double seq_time;
	gettimeofday (&startwtime, NULL);

	// start meanShift
	do {
		count++;
		conv = meanShift(Y, X);
	} while(conv == 0 && count < 30);

	gettimeofday (&endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

	//printMatrix(Y);
	printf("%d\n", count);

	// save results in a FILE
	FILE *fp;
	fp = fopen (RESFILE, "a");

	fprintf(fp, "time to converge:\t%f \t count:%d\n", seq_time, count);

	fclose(fp);

	// free allocated memory
	free(Y.elements);
	free(X.elements);

	return 0;
}


double gausian (double x) {
	double denom = VAR * 2;

	return	exp(- x / denom);
}

void duplicateMatrix(Matrix from, Matrix to) {
	int i;
	for(i = 0; i < from.height * from.width; i++){
		to.elements[i] = from.elements[i];
	}
}

void loadFile(char name[65], Matrix M)
{
	FILE *pointFile;
	int row;
	//pointFile=fopen("./data/points_600_2.bin","rb");
	pointFile=fopen(name,"rb");
	if (!pointFile){ printf("Unable to open file!\n"); /*return 1;*/ exit(1); }

	for (row = 0; row < M.height; row++)
	{
		//Loading a row of coordinates
		if (!fread(&(M.elements[row * M.width + 0]), sizeof(double), M.width, pointFile))
		{
			printf("Unable to read from file!");
			exit(1);
		}
	}
	fclose(pointFile);
}

void printMatrix(Matrix M) {
	int i, j;
	for(i = 0; i < M.height; i++) {
		for(j = 0; j < M.width; j++) {
			printf("%f \t", M.elements[i*M.width + j]);
		}
		printf("\n");
	}
}

int meanShift(Matrix Y, Matrix X) {
	double dist = 0, k;
	int i, j, yRow, conv = 1;

	double *nominator, *shift;
	double denominator;
	nominator = (double*)malloc(Y.width * sizeof(double));
	shift = (double*)malloc(Y.width * sizeof(double));

	for(yRow = 0; yRow < Y.height; yRow++) {
		// init variables
		denominator = 0;
		for(j = 0; j < Y.width; j++) {
			nominator[j] = 0;
			shift[j] = 0;
		}

		// Mean Shift algorithm
		for(i = 0; i < X.height; i++) {
			dist = 0;
			for(j = 0; j < X.width; j++) {
				dist += pow(Y.elements[yRow * Y.width + j] - X.elements[i * X.width + j], 2); // actually it is dist^2
			}

			k = gausian(dist);

			if (sqrt(dist) <= (VAR)) {
				for(j = 0; j < Y.width; j++) {
					nominator[j] += k * X.elements[i * X.width + j];
				}
				denominator += k;
			}
		}

		if (denominator > 0) {
			for(j = 0; j < Y.width; j++) {
				shift[j] = nominator[j] / denominator;  // actually now is the new y
				shift[j] = shift[j] - Y.elements[yRow * Y.width + j];
				Y.elements[yRow * Y.width + j] = nominator[j] / denominator;
			}
		}

		dist = 0;
		for(j = 0; j < Y.width; j++) {
			dist += pow(shift[j], 2);
		}
		dist = sqrt(dist);

		// if point doesn't converge
		if( !(dist < EPSILON) ) conv = 0;
	}

	return conv;
}
