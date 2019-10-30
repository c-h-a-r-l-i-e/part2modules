include_directories ("${PROJECT_SOURCE_DIR}/include" ${OpenCL_INCLUDE_DIRS})

add_executable(task3_reduction reduction.cpp)
target_compile_features(task3_reduction PRIVATE cxx_range_for)
target_link_libraries(task3_reduction ${OpenCL_LIBRARIES})
