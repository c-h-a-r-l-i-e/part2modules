#define __CL_ENABLE_EXCEPTIONS
#include <math.h>
#include <chrono>

#include "ocl_utils.hpp"
#include "cxxopts.hpp"

/** Creates a 1D Gaussian kernel for the low-pass filter
 * @param sigma standard deviation of the Gaussian in pixel units
 * @param filterSize the size of the filter in pixels
 * @return array of the size filterSize with the kernel
*/
float* get1DGaussianFilter(float sigma, int filterSize) {
    float *filter = new float[filterSize];
    float sum = 0;
    for(int i = -filterSize/2; i <= filterSize/2; i++) {
        filter[i + filterSize/2] = expf(-(i*i/(2*sigma*sigma)));
        sum += filter[i + filterSize/2];
    }
    // Normalize the filter
    for(int i = 0; i < filterSize; i++) {
        filter[i] /= sum;
    }
    return filter;
}

int gaussian_filter(char *imageLocation, int platformID, int deviceID){
    try {

      //-----------------------------------------------------
      // STEP 1: Load the image
      //-----------------------------------------------------
      STB_Image image = load_image(imageLocation);

      //-----------------------------------------------------
      // STEP 2: Initialize OpenCL
      //-----------------------------------------------------
      cl::Program program;
      cl::Device device;
      cl::Context context;
      std::tie(program, context, device) = initializeOCL("gaussian_filter.cl", platformID, deviceID);

      //-----------------------------------------------------
      // STEP 3: Create an OpenCL Image / texture and 
      // transfer data to the device
      // Note that we need to use CL_RGBA image instead of CL_RGB
      // as many platforms do not support CL_RGB images
      //-----------------------------------------------------
      
      cl::Image2D clImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                  cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                  image.width,
                  image.height,
                  0,
                  image.data);

      //-----------------------------------------------------
      // STEP 4: Create output image object
      //-----------------------------------------------------
      cl::Image2D xFilteredImage(context,
                  CL_MEM_READ_WRITE,
                  cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                  image.width,
                  image.height,
                  0,
                  NULL);
  
      cl::Image2D filteredImage(context,
                  CL_MEM_WRITE_ONLY,
                  cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                  image.width,
                  image.height,
                  0,
                  NULL);

      //-----------------------------------------------------
      // STEP 5: Create Gaussian blur filter
      //-----------------------------------------------------
      float sigma = 3.0f;
      int filterSize = (int)ceilf(sigma*3.f/2.f)*2+1;
      float *filter = get1DGaussianFilter(sigma, filterSize);

      //-----------------------------------------------------
      // STEP 6: Create device buffer for filter and transfer
      // it to the device     
      //----------------------------------------------------- 
      cl::Buffer clFilter (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                  sizeof(float)*(filterSize),
                  filter);

      //-----------------------------------------------------
      // STEP 7: Create the kernel and setup its arguments
      // Make use of Gaussian kernel separability
      //----------------------------------------------------- 
      cl::Kernel kernelX(program, "gaussian_filter_X");
      kernelX.setArg(0, clImage);
      kernelX.setArg(1, clFilter);
      kernelX.setArg(2, xFilteredImage);
      kernelX.setArg(3, filterSize);

      cl::Kernel kernelY(program, "gaussian_filter_Y");
      kernelY.setArg(0, xFilteredImage);
      kernelY.setArg(1, clFilter);
      kernelY.setArg(2, filteredImage);
      kernelY.setArg(3, filterSize);
	
      //-----------------------------------------------------
      // STEP 8: Create a command queue which we will use to
      // push commands to the device
      //----------------------------------------------------- 

      cl::CommandQueue queue(context, device);

      // Begin timer
      std::chrono::steady_clock::time_point begin_clock = std::chrono::steady_clock::now();

      queue.enqueueNDRangeKernel(
          kernelX,
          cl::NullRange,
          cl::NDRange(image.width , image.height),
          cl::NullRange
      );
      queue.enqueueNDRangeKernel(
          kernelY,
          cl::NullRange,
          cl::NDRange(image.width , image.height),
          cl::NullRange
      );

        queue.finish();
      std::chrono::steady_clock::time_point end_clock = std::chrono::steady_clock::now();

      //-----------------------------------------------------
      // STEP 9: Transfer image back to host
      //----------------------------------------------------- 
      unsigned char* data = new unsigned char[image.width * image.height * 4];
      cl::size_t<3> origin;
      origin[0] = 0; origin[1] = 0, origin[2] = 0;
      cl::size_t<3> region;
      region[0] = image.width; region[1] = image.height; region[2] = 1;
      queue.enqueueReadImage(filteredImage, CL_TRUE, origin, region, 0, 0 , data,  NULL, NULL);
      queue.finish();

      std::cout << "Time taken to blur the image using Gaussian filter in OpenCL = " << std::chrono::duration_cast<std::chrono::milliseconds> (end_clock - begin_clock).count() << " milliseconds\n";

      //-----------------------------------------------------
      // STEP 10: Compress and write the image to the disc
      //----------------------------------------------------- 
      STB_Image outputImage;
      outputImage.data = data;
      outputImage.width = image.width;
      outputImage.height = image.height;
      write_image(outputImage, "blurred_result.png");
      std::cout << "Results written to blurred_result.png" << std::endl;
    }
    catch (cl::Error err)
    {
      std::cerr << "ERROR: " << err.what() << "(" << getOCLErrorString(err.err()) << ")" << std::endl;
      return -1;
    }
    catch (const char* msg)
    {
      printf("ERROR: initializing...\n%s \n", msg);
      return -1;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    uint platform_id, device_id, test_size;

    std::tie(platform_id, device_id, test_size) = parseOptions(argc, argv, true);

    char *image_location = argv[argc-1];

	return gaussian_filter(image_location, platform_id, device_id);
}
