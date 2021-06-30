cmake_minimum_required(VERSION 3.4.3)

project(openvino_link_test CXX)

#set(CMAKE_CXX_COMPILER clang++-12)
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

target_link_libraries(${CMAKE_PROJECT_NAME}
  ${InferenceEngine_LIBRARIES}
  ${NGRAPH_LIBRARIES}
  ${TBB_IMPORTED_TARGETS}
)
