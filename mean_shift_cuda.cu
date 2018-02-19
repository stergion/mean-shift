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
#define RESFILE		"./res.txt"

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
__global__ void getDist(Matrix Y, Matrix X, Matrix YNew);
__global__ void convergence(Matrix Y, Matrix YNew, double *d_conv);
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
  loadFile(filename, X);
	duplicateMatrix(X, Y);

  meanShift(Y, X, YNew);

	free(X.elements);
	free(Y.elements);
	free(YNew.elements);

	return 0;
}


void printMatrix(Matrix M) {
	int i, j;
	for(i = 0; i < M.height; i+=300) {
		printf("%d: ", i);
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

	pointFile=fopen(name,"rb");
	if (!pointFile){ printf("Unable to open file!\n"); exit(1); }

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


	/*
	 Matrix dist
	 each cell with coordinates (i,j)
	 is the distance between the points y_i and x_j
	 */
	// printf("Allocate memory for d_YNew\n");
	Matrix d_YNew;
	d_YNew.height = d_X.height; d_YNew.width = d_X.width;
	size = d_YNew.height * d_YNew.width * sizeof(double);
	cudaMalloc(&d_YNew.elements, size);

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

	// Invoke kernels
	do {
		count++;
		conv[0] = 1;
		cudaMemcpy(d_conv, conv, sizeof(double), cudaMemcpyHostToDevice);

		// printf("Invoke kernel getDist...\n");
		dim3 dimBlock(BLOCK_SIZE); // Number of threads per grid
		dim3 dimGrid(X.height, X.width); // Number of blocks per Grid
		// printf("dimBlock(x,y,z): (%d,%d,%d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
		// printf("dimGrid(x,y,z): (%d,%d,%d)\n", dimGrid.x, dimGrid.y, dimGrid.z);

		getDist<<<dimGrid, dimBlock>>>(d_Y, d_X, d_YNew);
		cudaDeviceSynchronize();

		dimBlock.x = BLOCK_SIZE;								dimBlock.y =1;		dimBlock.z = 1;
		dimGrid.x = Y.height/dimBlock.x + 1;		dimGrid.y = 1;		dimGrid.z = 1;

	  // printf("Invoke kernel convergence...\n");
		convergence<<<dimGrid, dimBlock>>>(d_Y, d_YNew, d_conv);
		cudaDeviceSynchronize();

		cudaMemcpy(conv, d_conv, sizeof(double), cudaMemcpyDeviceToHost);
	} while(conv[0] == 0 && count < 50);

	gettimeofday (&endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);

	// save results in a FILE
	FILE *fp;
	fp = fopen (RESFILE, "a");

	fprintf(fp, "time to converge:\t%f \t count:%d\n", seq_time, count);

	fclose(fp);

	// Copy result from device memory to host memory
	size = d_X.width * d_X.height * sizeof(double);
  cudaMemcpy(X.elements, d_X.elements, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(Y.elements, d_Y.elements, size, cudaMemcpyDeviceToHost);

	// printf("Print Matrix X\n");
	// printMatrix(X);
	//printf("Print Matrix Y: %d\n", count);
	//printMatrix(Y);


	// Free device memory
  cudaFree(d_Y.elements);
	cudaFree(d_X.elements);
	cudaFree(d_YNew.elements);
	cudaFree(d_conv);

	free(conv);

	// printf("Exit meanShift\n");
}


__device__ double gausian (double x) {
	return	exp(- x / (2 * VAR));
}


__global__ void getDist(Matrix Y, Matrix X, Matrix YNew) {
	int tid = threadIdx.x;

	/*
	 * dimSum[]: every thread writes the result of k*X[][blockIdx.x],
	 * in the end a reduction is performed to get the final Sum.
	 */
	__shared__ double dimSum[BLOCK_SIZE];
	__shared__ double denom[BLOCK_SIZE];
  // Fill dimSum with zeros.
  dimSum[tid] = 0;
	denom[tid] = 0;
	__syncthreads();

	int i, j, xRow;
	double dist;
	for(i = 0; i < X.height/blockDim.x + 1; i++) {
		xRow = i * blockDim.x + tid;
		dist = 0;

		if(xRow < X.height) {
			for(j = 0; j < X.width; j++) {
				dist += pow(Y.elements[blockIdx.x * Y.width + j] - X.elements[xRow * X.width + j], 2);
			}
			double k = gausian(dist);
			if(dist > VAR) k = 0;

			dimSum[tid] += k * X.elements[xRow * X.width + blockIdx.y];
			denom[tid] += k;
		}
	}
	__syncthreads();

	// reduction
	for(i = blockDim.x/2; i > 0; i >>= 1) {
		if(tid < i) {
			dimSum[tid] += dimSum[tid + i];
			denom[tid] += denom[tid + i];
		}
		__syncthreads();
	}

	// After the reduction is finished save value.
	if(tid == 0) YNew.elements[blockIdx.x * YNew.width + blockIdx.y] = dimSum[tid] / denom[tid];
}


__global__ void convergence(Matrix Y, Matrix YNew, double *d_conv) {
	double M = 0;

	int yRow = blockIdx.x * blockDim.x + threadIdx.x;
	if(yRow < Y.height) {
		int i;
		for(i = 0; i < Y.width; i++) {
			M += pow(YNew.elements[yRow * Y.width + i] - Y.elements[yRow * Y.width + i], 2);
			Y.elements[yRow * Y.width + i] = YNew.elements[yRow * Y.width + i];
		}

		// if point doesn't converge
		if( !(sqrt(M) < EPSILON) ) d_conv[0] = 0;
	}

}
