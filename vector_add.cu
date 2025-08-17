#include <iostream>
#include <vector>


// threadIdx.x <- index of the thread inside the block
// blockIdx.x <- index of the block in the grid
// blockDim.x <- how many threads are in a block
// gridDim.x <- how many blocks in the grid

__global__ void add(int n, float *x, float *y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) y[i] = x[i] +	y[i];
}

__global__ void scale(int n, float c, float *x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// if ( i < n ) x[i] = x[i] * c;
	if ( i < n ) x[i] = x[i] * c;
}

bool scale_item(void) {
	int N = 1 << 20;
	std::vector<float> x(N, 2.0f);
	float c = 3.14f;
	float *d_x;
	cudaMalloc(&d_x, N*sizeof(float));
	cudaMemcpy(d_x, x.data(), N*sizeof(float), cudaMemcpyHostToDevice);
	
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	
	scale<<<numBlocks, blockSize>>>(N, c, d_x);
	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << "\n";
    }
    cudaDeviceSynchronize();
	cudaMemcpy(x.data(), d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_x);

	std::cout << "scaler should be 2*pi = " << x[0] << "\n";
	std::cout << "scaler should be 2*pi = " << x[N-10] << "\n";
	return true;
}

int add_print(void) {
	int N = 1<<20;
	std::vector<float> x(N, 1.0f), y(N, 2.0f);
	float *d_x, *d_y;
	cudaMalloc(&d_x, N*sizeof(float));
	cudaMalloc(&d_y, N*sizeof(float));
	cudaMemcpy(d_x, x.data(), N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y.data(), N*sizeof(float), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	add<<<numBlocks, blockSize>>>(N, d_x, d_y);

	cudaMemcpy(y.data(), d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_x);
	cudaFree(d_y);

	std::cout << "y[0] = " << y[0] << "\n";
	std::cout << "y[n-1] = " << y[N-1] << "\n";
	return 1;
}

int main(void) {
	add_print();
	scale_item();
}
