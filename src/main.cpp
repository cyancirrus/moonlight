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


// x ~ M[i, k]
// y ~ M[k, j]
// r ~ M[i, j]

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


vector<float> mat_mul(
	int i, int j, int k,
	const vector<float> &x,
	const vector<float> &y
) {
	vector<float>  r(i * j, 0.0f);
		
	for (int idx = 0; idx < i; idx ++) {
		for (int kdx = 0; kdx < k; kdx ++) {
			for (int jdx = 0; jdx < j; jdx ++ ) {
				r[idx * j + jdx] += x[idx * k + kdx] * y[kdx * j + jdx];
			}
		}
	}
	return r;
}

int main(void) {
	// predict_input();
	vector<float> x{1.0, 2.0,
                    3.0, 4.0};  // shape 2x2
    vector<float> y{5.0, 6.0,
                    7.0, 8.0};  // shape 2x2

    vector<float> r = mat_mul(2, 2, 2, x, y);

    std::cout << "matrix\n";
    std::cout << r[0] << ", " << r[1] << "\n";
    std::cout << r[2] << ", " << r[3] << "\n";

}
