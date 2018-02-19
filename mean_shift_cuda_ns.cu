#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>


#define VAR		1
#define EPSILON		0.0001
#define BLOCK_SIZE	512
#define RESFILE		"./res_ns.txt"

/*
  Matrices are stored in row-major order:
  M(row, col) = *(M.elements + row * M.width + col)
 */
typedef struct {
	int width;
	int height;
	double *elements;
} Matrix;


void printMatrix(Matrix M);
void duplicateMatrix(Matrix from, Matrix to);
void loadFile(char name[65], Matrix M);
void meanShift(Matrix Y, Matrix X, Matrix YNew);
__global__ void getDist(Matrix Y, Matrix X,  double *d_conv);
__global__ void convergence(Matrix Y, Matrix YNew);
__device__ double gausian (double);


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
	YNew.height = N; YNew.width = d;
	YNew.elements = (double*)malloc(YNew.width * YNew.height * sizeof(double));
  loadFile(FILENAME, X);
	duplicateMatrix(X, Y);

  meanShift(Y, X, YNew);

	free(X.elements);
	free(Y.elements);
	free(YNew.elements);

	return 0;
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


void meanShift(Matrix Y, Matrix X, Matrix YNew) {
  // Load Y and X to device memory
	// printf("Load Y and X to device memory\n");

  Matrix d_Y;
	d_Y.width = Y.width; d_Y.height = Y.height;
	size_t size = d_Y.width * d_Y.height * sizeof(double);
	cudaMalloc(&d_Y.elements, size);
	cudaMemcpy(d_Y.elements, Y.elements, size, cudaMemcpyHostToDevice);

	Matrix d_X;
	d_X.width = X.width; d_X.height = X.height;
	size = d_X.width * d_X.height * sizeof(double);
	cudaMalloc(&d_X.elements, size);
	cudaMemcpy(d_X.elements, X.elements, size, cudaMemcpyHostToDevice);

	double *conv, *d_conv;
	conv = (double*)malloc(sizeof(double));
	cudaMalloc(&d_conv, sizeof(double));

	// printf("Print Matrix X\n");
	// printMatrix(X);
	// printf("Print Matrix Y\n");
	// printMatrix(Y);
	cudaDeviceSynchronize();

	int count = 0;
	struct timeval startwtime, endwtime;
	double seq_time;
	gettimeofday (&startwtime, NULL);

  // Invoke kernel
  do{
		count++;
		conv[0] = 1;
		cudaMemcpy(d_conv, conv, sizeof(double), cudaMemcpyHostToDevice);

		// printf("Invoke kernel getDist...\n");
	  dim3 dimBlock(BLOCK_SIZE); // Number of threads per block
	  dim3 dimGrid(X.height / BLOCK_SIZE + 1); // Number of blocks per Grid
	  // printf("dimBlock(x,y,z): (%d,%d,%d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
	  // printf("dimGrid(x,y,z): (%d,%d,%d)\n", dimGrid.x, dimGrid.y, dimGrid.z);

	  getDist<<<dimGrid, dimBlock>>>(d_Y, d_X, d_conv);
	  cudaDeviceSynchronize();

		cudaMemcpy(conv, d_conv, sizeof(double), cudaMemcpyDeviceToHost);
	} while(conv[0] == 0);

	gettimeofday (&endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

	// save results in a FILE
	FILE *fp;
	fp = fopen (RESFILE, "a");

	fprintf(fp, "time to converge:\t%f \t count:%d\n", seq_time, count);

	fclose(fp);

	// Copy result from device memory to host memory
	size = X.width * X.height * sizeof(double);
  cudaMemcpy(X.elements, d_X.elements, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(Y.elements, d_Y.elements, size, cudaMemcpyDeviceToHost);

	// printf("Print Matrix X\n");
	// printMatrix(X);
	// printf("Print Matrix Y: %d\n", count);
	//printMatrix(Y);


	free(conv);

	// Free device memory
  cudaFree(d_Y.elements);
	cudaFree(d_X.elements);
	cudaFree(d_conv);

	// printf("Exit meanShift\n");
}


__device__ double gausian (double x) {
	return	exp(- x / (2 * VAR));
}


__global__ void getDist(Matrix Y, Matrix X, double *d_conv) {
  int yRow = blockIdx.x * blockDim.x + threadIdx.x;

  if (yRow < Y.height) {
    double dist = 0, k;
    int i, j;

    double *nominator, *shift;
		double denominator;
    nominator = (double*)malloc(Y.width * sizeof(double));
		shift = (double*)malloc(Y.width * sizeof(double));

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
		if( !(dist < EPSILON) ) d_conv[0] = 0;

		free(nominator);
		free(shift);
  }
}
