include_directories ("${PROJECT_SOURCE_DIR}/include" ${OpenCL_INCLUDE_DIRS})

add_executable(task4_gaussian_filter gaussian_filter.cpp)
target_compile_features(task4_gaussian_filter PRIVATE cxx_range_for)
target_link_libraries(task4_gaussian_filter ${OpenCL_LIBRARIES})
