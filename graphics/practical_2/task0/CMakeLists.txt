include_directories ("${PROJECT_SOURCE_DIR}/include" ${OpenCL_INCLUDE_DIRS})

add_executable(task0_list_platforms list_platforms.cpp)
target_compile_features(task0_list_platforms PRIVATE cxx_range_for)
target_link_libraries(task0_list_platforms ${OpenCL_LIBRARIES})
