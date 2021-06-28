cmake_minimum_required(VERSION 3.4.3)

project(ocl_layer CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(SYCL REQUIRED)

set(CMAKE_CXX_COMPILER ${SYCL_COMPILER})
#set(CMAKE_CXX_STANDARD 11)

find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)
find_package(TBB REQUIRED tbb tbbmalloc)

include_directories(
  "include"
)

file(GLOB SOURCES "src/*.cpp")

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})
target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_range_for)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE ${SYCL_FLAGS})

target_link_libraries(${CMAKE_PROJECT_NAME}
  ${InferenceEngine_LIBRARIES}
  ${NGRAPH_LIBRARIES}
  ${TBB_IMPORTED_TARGETS}
  ${SYCL_LIBRARY}
)
