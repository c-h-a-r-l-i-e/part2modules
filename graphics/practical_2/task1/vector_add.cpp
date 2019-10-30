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

#include <CL/cl.hpp>
#include <iostream>

#include "ocl_utils.hpp"
#include "cxxopts.hpp"

int vector_add(uint platformID, uint deviceID, uint test_size){
	try{
		//-----------------------------------------------------
		// STEP 1: Initialize OpenCL (Read function definition
		// for more details)
		//-----------------------------------------------------
		cl::Program program;
		cl::Device device;
		cl::Context context;
		std::tie(program, context, device) = initializeOCL("vector_add.cl", platformID, deviceID);

		//-----------------------------------------------------
		// STEP 2: Create device buffers
		//----------------------------------------------------- 
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(int) * test_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(int) * test_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(int) * test_size);

		// Allocate our test arrays on the heap
		int *A = new int[test_size];
		int *B = new int[test_size];
		for (int i=0;i<test_size;i++){
			A[i] = i;
			B[i] = test_size - 1 - i;
		}

		//-----------------------------------------------------
		// STEP 3: Create a command queue which we will use to
		// push commands to the device
		//----------------------------------------------------- 
		cl::CommandQueue queue(context, device);

		//-----------------------------------------------------
		// STEP 4: Transfer arrays A and B from host to device 
		// memory
		//----------------------------------------------------- 
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(int) * test_size, A);
        delete[] A;
		queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(int) * test_size, B);

		//-----------------------------------------------------
		// STEP 5: Create the kernel
		//----------------------------------------------------- 
		cl::Kernel kernel(program, "simple_add");

		//-----------------------------------------------------
		// STEP 6: Set the kernel arguments
		//----------------------------------------------------- 
		kernel.setArg(0, buffer_A);
		kernel.setArg(1, buffer_B);
		kernel.setArg(2, buffer_C);
		kernel.setArg(3, test_size);

		queue.finish();
		// Begin timer now to avoid counting opencl compilation time
		std::chrono::steady_clock::time_point begin_time = std::chrono::steady_clock::now();

		//-----------------------------------------------------
		// STEP 7: Enqueue the kernel for execution
		//----------------------------------------------------- 
		// We do not specify local_work_size (last argument cl::NullRange) and let the 
		// implementation determine how to break global_work_size
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(test_size), cl::NullRange);

		queue.finish();
		std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
		std::cout << "Finished in " << std::chrono::duration_cast<std::chrono::milliseconds> (end_time - begin_time).count() << " milliseconds \n";

		//-----------------------------------------------------
		// STEP 8: Read the output buffer back to the host
		//----------------------------------------------------- 
		int *C = new int[test_size];
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * test_size, C);
		queue.finish();

		std::cout << " result (showing the first 100 elements): \n";
		for (int i = 0; i < std::min((int)test_size,100); i++){
			std::cout << C[i] << " ";
		}
		std::cout << std::endl;
        //delete[] A;
		delete[] B;
		delete[] C;
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

	return vector_add(platform_id, device_id, test_size);
}
