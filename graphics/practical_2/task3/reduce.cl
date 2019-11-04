/**
 * Sum all the values in the large array of uchars (unsigned bytes)
 * buffer - pointer to the vary large array
 * scratch - fast, local memory, of the size get_local_size(0)
 * length - the numer of elements in the very large array
 * result - store the result from each work group in that array
 */
__kernel void reduce(__global const uchar*  buffer, __local int* scratch, __const int length, __global int* result) {
    
    int global_index = get_global_id(0);
    int local_index = get_local_id(0);
    int global_size = get_global_size(0);
    int local_size = get_local_size(0);
    int accumulator = 0;

    // First stage: Sequencial summation by multiple threads
    // Add sequencially elements of the array. 
    // Each thread should sum all elements that are "get_global_size(0)" apart 
    // up to the end of the array (given by "length"). Then, put the result to 
    // "scratch".

    for (int i = global_index; i < length; i+=global_size) {
        accumulator += buffer[i];
    }
    scratch[local_index] = accumulator;

    barrier(CLK_LOCAL_MEM_FENCE);
    // Second stage: perform reduction in the local memory
    // Sum all the elements in the scratch[] and store the result in 
    // result[get_group_id(0)]

    for (int offset = local_size / 2; offset > 0; offset >>= 1) {
        if (local_index < offset) {
            scratch[local_index] = scratch[local_index] + scratch[local_index + offset];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        result[get_group_id(0)] = scratch[0];
    }
}
