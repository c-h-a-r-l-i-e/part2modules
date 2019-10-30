// The line below is needed to solve some Windows'y issues with MSVS
#define NOMINMAX

#include <chrono>
#include <algorithm>

#define __CL_ENABLE_EXCEPTIONS

#include "ocl_utils.hpp"
#include "cxxopts.hpp"

int reduction(uint platformID, uint deviceID, uint test_size){

    //-----------------------------------------------------
    // STEP 1: Generate Data
    //-----------------------------------------------------
    std::cout << "Generating rundom numbers... "  << std::endl;
    std::chrono::steady_clock::time_point begin_clock = std::chrono::steady_clock::now();
    std::vector<int> numbers_int = generate_random_vector(test_size, 1, 20);
    std::vector<uint8_t> numbers = intToByte(numbers_int); // To decrease memory on GPU; only works when range is >0 && <256
    std::chrono::steady_clock::time_point end_clock = std::chrono::steady_clock::now();
    std::cout << "Time Taken to generate random numbers = " << std::chrono::duration_cast<std::chrono::milliseconds> (end_clock - begin_clock).count() << " milliseconds \n";

    //-----------------------------------------------------
    // STEP 2: Serially calculate the result to compare and
    // and verify the parallel implementation
    //-----------------------------------------------------
    std::cout << "Serial reduction on the CPU ... "  << std::endl;
    begin_clock = std::chrono::steady_clock::now();
    long long int sum_serial = 0;
    for(size_t i=0; i<numbers.size();i++)
    {
        sum_serial += numbers[i];
    }
    end_clock = std::chrono::steady_clock::now();
    std::cout << "Serial Implementation: Sum = " << sum_serial << "\n";
    std::cout << "Serial Implementation: Mean = " << (double)sum_serial / numbers.size() << "\n";
    std::cout << "Serial Implementation: Time Taken = " << std::chrono::duration_cast<std::chrono::milliseconds> (end_clock - begin_clock).count() << " milliseconds \n";

    // OpenCL Implementation
    long long int sum_parallel = 0;

    try{
        //-----------------------------------------------------
		// STEP 1: Initialize OpenCL
		//-----------------------------------------------------
		cl::Program program;
		cl::Device device;
		cl::Context context;
		std::tie(program, context, device) = initializeOCL("reduce.cl", platformID, deviceID);

		//-----------------------------------------------------
		// STEP 2: Create the kernel
		//-----------------------------------------------------
        cl::Kernel kernel(program, "reduce");

        //-----------------------------------------------------
		// STEP 3: Configure the work-item structure
		//-----------------------------------------------------
        int work_group_size = std::min((int)kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device), (int)test_size);
        int num_work_groups = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

		//-----------------------------------------------------
		// STEP 4: Create device buffers
		//-----------------------------------------------------
        cl::Buffer buffer_input(context, CL_MEM_READ_ONLY, sizeof(uint8_t) * numbers.size());
        cl::Buffer buffer_result(context, CL_MEM_WRITE_ONLY, sizeof(int) * num_work_groups);

		//-----------------------------------------------------
		// STEP 5: Create a command queue which we will use to
		// push commands to the device
		//-----------------------------------------------------
        cl::CommandQueue queue(context, device);

		//-----------------------------------------------------
		// STEP 6: Fill the input buffer
		//-----------------------------------------------------
        queue.enqueueWriteBuffer(buffer_input, CL_TRUE, 0, sizeof(uint8_t) * numbers.size(), numbers.data());

		//-----------------------------------------------------
		// STEP 7: Create the kernel and setup its arguments
		//-----------------------------------------------------

        kernel.setArg(0, buffer_input); //__global const uchar*  buffer
        kernel.setArg(1, sizeof(int)*work_group_size, NULL); //__local int* scratch
        kernel.setArg(2, (int)numbers.size()); //__const int length
        kernel.setArg(3, buffer_result); //__global int* result

        queue.finish();
        // Begin timer now to avoid counting opencl compilation time
        std::cout << "Parallel reduction on the GPU ... "  << std::endl;
        begin_clock = std::chrono::steady_clock::now();

        //-----------------------------------------------------
		// STEP 8: Enqueue the kernel for execution
		//-----------------------------------------------------
        // Pass below an appropriate size for the global and the local work size using 1D cl::NDRange
        
        cl::NDRange global_work_size(num_work_groups * work_group_size);
        cl::NDRange local_work_size(work_group_size);

        std::cout << work_group_size << "," << num_work_groups;

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, local_work_size);

        queue.finish();
        end_clock = std::chrono::steady_clock::now();

		//-----------------------------------------------------
		// STEP 9: Read results back from the device
		//-----------------------------------------------------
        int* partial_sums = new int[num_work_groups];
        queue.enqueueReadBuffer(buffer_result, CL_TRUE, 0, sizeof(int) * num_work_groups, partial_sums);
        queue.finish();

		//-----------------------------------------------------
		// STEP 10: Add up partial sums of each work-group
		//-----------------------------------------------------
        for(int i=0; i<num_work_groups; i++){
            sum_parallel += partial_sums[i];
        }

        std::cout << "Parallel Implementation: Sum = " << sum_parallel << "\n";
        std::cout << "Parallel Implementation: Mean = " << (double)sum_parallel / (double)numbers.size() << "\n";
        std::cout << "Parallel Implementation: Time Taken = " << std::chrono::duration_cast<std::chrono::milliseconds> (end_clock - begin_clock).count() << " milliseconds \n";
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

	return reduction(platform_id, device_id, test_size);
}
