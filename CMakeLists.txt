cmake_minimum_required(VERSION 3.18)   # 3.18+:CUDA 语言与 CUDA::cudart 导入目标
project(myfft LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 源码为 UTF-8(含中文注释):MSVC 下显式声明,避免 C4819/乱码
# (仅作用于 C/C++;nvcc 不接受 MSVC 风格选项,需用生成器表达式隔离)
if (MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
endif()

option(MYFFT_ENABLE_CUDA "CUDA 后端(M1:需 CUDA Toolkit,4090=sm_89)" OFF)
option(MYFFT_BUILD_TESTS "构建正确性测试" ON)
option(MYFFT_BUILD_BENCH "构建性能测试" ON)

if (MYFFT_ENABLE_CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # 模仿 cuFFT 的多架构 fatbin:75=GTX1650, 86/89=3090/4090, 90=H100
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)
  endif()
  enable_language(CUDA)
endif()

add_subdirectory(src)

if (MYFFT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if (MYFFT_BUILD_BENCH)
  add_subdirectory(bench)
  add_subdirectory(examples)
endif()
