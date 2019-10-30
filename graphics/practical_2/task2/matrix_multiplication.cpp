// The line below is needed to solve some Windows'y issues with MSVS
#define NOMINMAX

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <exception>
#include <chrono>

#include <math.h>

#define __CL_ENABLE_EXCEPTIONS

#if __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/opencl.h>
#endif

// #include <CL/cl.hpp>
#include <iostream>

#include "ocl_utils.hpp"
#include "cxxopts.hpp"

int matrix_multiplication(uint platformID, uint deviceID, uint test_size) {
	
	std::cout << "Generating test matrices ...\n";
	float *A = new float[test_size*test_size];
	float *B = new float[test_size*test_size];
	for (int i = 0; i < test_size*test_size; i++)
	{
		A[i] = ((float) rand()) / (float) RAND_MAX;
		B[i] = ((float) rand()) / (float) RAND_MAX;
	}
	try
	{
		//-----------------------------------------------------
		// STEP 1: Initialize OpenCL (Read function definition
		// for more details)
		//-----------------------------------------------------
		cl::Program program;
		cl::Device device;
		cl::Context context;
		std::tie(program, context, device) = initializeOCL("matrix_multiplication.cl", platformID, deviceID);

		//-----------------------------------------------------
		// STEP 2: Create device buffers
		//----------------------------------------------------- 
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(float) * test_size * test_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(float) * test_size * test_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(float) * test_size * test_size);

		//-----------------------------------------------------
		// STEP 3: Create a command queue which we will use to
		// push commands to the device
		//----------------------------------------------------- 
		cl::CommandQueue queue(context, device);

		//-----------------------------------------------------
		// STEP 4: Populate the input buffer
		//----------------------------------------------------- 
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(float) * test_size * test_size, A);
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(float) * test_size * test_size, B);

		//-----------------------------------------------------
		// STEP 5: Create the kernel
		//----------------------------------------------------- 
		cl::Kernel kernel(program, "mat_multiply");

		//-----------------------------------------------------
		// STEP 6: Set the kernel arguments
		//----------------------------------------------------- 
		kernel.setArg(0, buffer_A);
		kernel.setArg(1, buffer_B);
		kernel.setArg(2, buffer_C);
		kernel.setArg(3, test_size);
		
		std::cout << "Running parallel matrix multiplication ...\n";
		// Begin timer
		std::chrono::steady_clock::time_point begin_clock = std::chrono::steady_clock::now();

		//-----------------------------------------------------
		// STEP 7: Enqueue the kernel for execution
		//----------------------------------------------------- 
    // TODO: Pass below an appropriate size for the global work size using 1D cl::NDRange
        
        cl::NDRange range(test_size, test_size);
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, range, cl::NullRange );
		

		queue.finish();
		// End timer before validating
		std::chrono::steady_clock::time_point end_clock = std::chrono::steady_clock::now();
		std::cout << "Time taken for parallel matrix multiplication = " << std::chrono::duration_cast<std::chrono::milliseconds> (end_clock - begin_clock).count() << " milliseconds\n";

		//-----------------------------------------------------
		// STEP 8: Read the output buffer back to the host
		//----------------------------------------------------- 
		float *C = new float[test_size*test_size];
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float) * test_size * test_size, C);
		queue.finish();

		if (test_size > 1000) {
			std::cout << "Skipping validation because the matrix is too big." << std::endl;
		}
		else {
			// Validate the output
			for (int i = 0; i < test_size; i++)
				for (int j = 0; j < test_size; j++)
				{
					float sum = 0;
					for (int k = 0; k < test_size; k++)
						sum += A[i*test_size + k] * B[k*test_size + j];

					if (fabs(C[i*test_size + j] - sum) > 1e-3)
					{
						std::cout << "ERROR: Matrix multiplication result is incorrect. " << std::endl;
						return -1;
					}
				}
			std::cout << "Output validated" << std::endl;
		}
	}
	catch (cl::Error err)
	{
		std::cerr << "ERROR: " << err.what() << "(" << getOCLErrorString(err.err()) << ")" << std::endl;
		return -1;
	}
	catch (const char* msg)
	{
		printf("ERROR: initializing OpenCL\n%s \n", msg);
		return -1;
	}
    return 0;
}

int main(int argc, char* argv[])
{
	uint platform_id, device_id, test_size;

	std::tie(platform_id, device_id, test_size) = parseOptions(argc, argv);

	return matrix_multiplication(platform_id, device_id, test_size);
}
