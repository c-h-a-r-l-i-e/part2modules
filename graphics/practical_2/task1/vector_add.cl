__kernel void simple_add(__global const int* A, __global const int* B, __global int* C, int length) {
	int index = get_global_id(0);
	if( index < length ) {
		C[index] = A[index] + B[index];
	}
};
