#include <iostream>
#include <vector>

__global__ void add(int n, float *x, float *y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) y[i] = x[i] +	y[i];
}



int main(void) {
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
	std::cout << "y[N-1] = " << y[N-1] << "\n";

}
