include_directories ("${PROJECT_SOURCE_DIR}/include" ${OpenCL_INCLUDE_DIRS})

add_executable(task2_matrix_multiplication matrix_multiplication.cpp)
target_compile_features(task2_matrix_multiplication PRIVATE cxx_range_for)
target_link_libraries(task2_matrix_multiplication ${OpenCL_LIBRARIES})
