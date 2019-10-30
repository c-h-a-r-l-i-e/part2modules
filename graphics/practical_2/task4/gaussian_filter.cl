const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE 
                        | CLK_ADDRESS_CLAMP_TO_EDGE 
                        | CLK_FILTER_NEAREST;
 
__kernel void gaussian_filter_X(
        __read_only image2d_t image,
        __constant float * filter,
        __write_only image2d_t blurredImage,
        __private int filterSize
    ) {
 
    const int2 pos = {get_global_id(0), get_global_id(1)};

    float4 pixel = (float4)(0);

    // Iterate through pixels, scaling them by the filter and 
    // adding them to the new pixel value.
    for (int i = 0; i < filterSize; i++) {
        int2 ipos = {pos.x - filterSize/2 + i, pos.y};
        float4 filter_val = (float4)filter[i];
        // Set A in RGBA to the same value
        filter_val.w = 1;

        pixel += convert_float4(read_imageui(image, sampler, ipos)) 
                    * filter_val;
    }

    write_imageui(blurredImage, pos, convert_uint4_rte(pixel));
}

__kernel void gaussian_filter_Y(
        __read_only image2d_t image,
        __constant float * filter,
        __write_only image2d_t blurredImage,
        __private int filterSize
    ) {
 
    const int2 pos = {get_global_id(0), get_global_id(1)};
	 
    // Iterate through pixels, scaling them by the filter and 
    // adding them to the new pixel value.
    float4 pixel = (float4)(0);
    for (int i = 0; i < filterSize; i++) {
        int2 ipos = {pos.x, pos.y - filterSize/2 + i};
        float4 filter_val = (float4)filter[i];
        // Set A in RGBA to the same value
        filter_val.w = 1;

        pixel += convert_float4(read_imageui(image, sampler, ipos)) 
                    * filter_val;
    }
    write_imageui(blurredImage, pos, convert_uint4_rte(pixel));
}
