include_directories ("${PROJECT_SOURCE_DIR}/include" ${OpenCL_INCLUDE_DIRS})

add_executable(task1_vector_add vector_add.cpp)
target_compile_features(task1_vector_add PRIVATE cxx_range_for)
target_link_libraries(task1_vector_add ${OpenCL_LIBRARIES})
