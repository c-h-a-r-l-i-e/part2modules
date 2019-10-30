__kernel void mat_multiply(__global const float* A, __global const float* B, __global float* C,
		const int n) {

	const int row_id = get_global_id(0);
	const int column_id = get_global_id(1);
    
    float val = 0;
    for (int i = 0; i < n; i++) {
         val += A[row_id * n + i] * B[i * n + column_id];
    }

    C[row_id * n + column_id] = val;
}
