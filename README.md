# mean-shift
implementation of mean shift algorithm.

compile **mean_shift_serial.c** with

 ` gcc -O3 mean_shift_serial.c -o mean_shift_serial -lm `
 
compile **mean_shift_cuda.cu** and **mean_shift_cuda_ns.cu** with:

 ` nvcc -O3 mean_shift_cuda.cu -o mean_shift_cuda `
 
 ``` nvcc -O3 mean_shift_cuda_ns.cu -o mean_shift_cuda_ns ```
