cmake_minimum_required(VERSION 3.18)
project(cuda_rasterizer LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
option(WITH_TORCH "Builds also rasterizer compatible with torch tensors" ON)

find_package(Thrust REQUIRED)

add_library(ds_cuda_rasterizer
    cpp/rasterizer_cuda.cu
    cpp/rasterizer_kernels.cu)

target_include_directories(ds_cuda_rasterizer PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(ds_cuda_rasterizer PRIVATE ${CUDA_INCLUDE_DIRS})
target_include_directories(ds_cuda_rasterizer PUBLIC cpp)

target_link_libraries(ds_cuda_rasterizer ${CUDA_LIBRARIES} thrust::thrust)

set_target_properties(ds_cuda_rasterizer PROPERTIES
                      CUDA_ARCHITECTURES "native")

if(WITH_TORCH)
  find_package(Torch REQUIRED)
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

  target_sources(ds_cuda_rasterizer PRIVATE
    cpp/rasterizer_torch.cu
    cpp/gsplat/backward.cu
    cpp/gsplat/bindings.cu)

  target_link_libraries(ds_cuda_rasterizer ${TORCH_LIBRARIES} Python3::Python)

  if(BUILD_TESTING AND WITH_TORCH)
    enable_testing()
    find_package(GTest REQUIRED)
    include(CTest)

    add_executable(rasterizer_test tests/rasterizer_test.cpp)
    target_link_libraries(rasterizer_test ds_cuda_rasterizer gtest::gtest ${TORCH_LIBRARIES} ${CUDA_LIBRARIES})
  endif()
endif()
