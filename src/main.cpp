#include <vector>
#include <iostream>
#include "pipeline.h"
using std::vector;

// warps <- a collection of threads
// sharedMemBytes <- shareds

// threadIdx.x <- index of the thread inside the block
// blockIdx.x <- index of the block in the grid
// blockDim.x <- how many threads are in a block
// gridDim.x <- how many blocks in the grid

void predict_input(void) {
	int N = 1<<20;
	float c = 3.14f;
	vector<float> x(N, 2.0f);
	vector<float> y(N, 2.71f);

	float r = pipeline(N, c, x, y );
	std::cout << "total = " << r << "\n";

	// std::cout << "y[0] = " << y[0] << "\n";
	// std::cout << "y[n-1] = " << y[N-1] << "\n";
}

int main(void) {
	predict_input();
}
