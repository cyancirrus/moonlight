#include <iostream>
#include <vector>
using std::vector;


// threadIdx.x <- index of the thread inside the block
// blockIdx.x <- index of the block in the grid
// blockDim.x <- how many threads are in a block
// gridDim.x <- how many blocks in the grid
// sharedMemBytes

constexpr int BLOCKSIZE = 256;

__global__ void add(int n, float *x, float *y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) y[i] = x[i] +	y[i];
}

__global__ void scale(int n, float c, float *x) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) x[i] = x[i] * c;
}

// __global__ void reduce_sum_atomic(int n, float *in, float *out) {
__global__ void reduce_sum_atomic(int n, float *in, float *global_sum) {
	// need to share how much memory extra argument
	__shared__ float smem[BLOCKSIZE];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	smem[tid] = (i < n ? in[i] : 0);
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1 ) {
		if (tid < stride) {
			smem[tid] += smem[tid + stride];
		}
		__syncthreads();
	}
	// if (tid == 0) out[blockIdx.x] = smem[0];
	if (tid == 0) {
		atomicAdd(global_sum, smem[0]);
	}
}


float pipeline(
	int n,
	float c,
	vector<float>& x,
	vector<float>& y
) {
	int blocks = (n + BLOCKSIZE - 1 ) / BLOCKSIZE;
	float result = 0.0f;
	float *d_x, *d_y, *d_r;

	cudaMalloc(&d_r, sizeof(float));
	cudaMalloc(&d_x, n * sizeof(float));
	cudaMalloc(&d_y, n * sizeof(float));

	cudaMemcpy(d_r, &result, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice);

	add<<<blocks, BLOCKSIZE>>>(n, d_x, d_y);
	scale<<<blocks, BLOCKSIZE>>>(n, c, d_y);
	reduce_sum_atomic<<<blocks, BLOCKSIZE>>>(n, d_y, d_r);
	cudaMemcpy(&result, d_r, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_r);
	
	return result;
}

void predict_input(void) {
	int N = 1<<20;
	float c = 3.14f;
	float r = 0.0f;
	vector<float> x(N, 2.0f);
	vector<float> y(N, 2.71f);

	r = pipeline(N, c, x, y );
	std::cout << "total = " << r << "\n";

	// std::cout << "y[0] = " << y[0] << "\n";
	// std::cout << "y[n-1] = " << y[N-1] << "\n";
}

int main(void) {
	predict_input();
}
