#include <iostream>
#include <vector>
using std::vector;


// threadIdx.x <- index of the thread inside the block
// blockIdx.x <- index of the block in the grid
// blockDim.x <- how many threads are in a block
// gridDim.x <- how many blocks in the grid

constexpr int BLOCKSIZE = 256;

__global__ void add(int n, float *x, float *y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) y[i] = x[i] +	y[i];
}

__global__ void scale(int n, float c, float *x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) x[i] = x[i] * c;
}


vector<float> pipeline(
	int n,
	float c,
	vector<float>& x,
	vector<float>& y
) {
	int blocks = (n + BLOCKSIZE - 1 ) / BLOCKSIZE;
	float *d_x, *d_y;
	cudaMalloc(&d_x, n * sizeof(float));
	cudaMalloc(&d_y, n * sizeof(float));
	
	cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice);

	add<<<blocks, BLOCKSIZE>>>(n, d_x, d_y);
	scale<<<blocks, BLOCKSIZE>>>(n, c, d_y);
	cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_x);
	cudaFree(d_y);
	
	return y;
}

void predict_input(void) {
	int N = 1<<20;
	float c = 3.14f;
	vector<float> x(N, 2.0f);
	vector<float> y(N, 2.71f);

	y = pipeline(N, c, x, y );

	std::cout << "y[0] = " << y[0] << "\n";
	std::cout << "y[n-1] = " << y[N-1] << "\n";
}

int main(void) {
	predict_input();
}
