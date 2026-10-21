# 单一动态库目标:模仿 cuFFT 的发布形态(libmyfft.so.1 / myfft.dll)
set(MYFFT_LIB_SRC
  api.cpp
  graph.cpp
  heuristics.cpp
  plan.h
  executor.cpp
  cpu/cpu_fft.cpp)

if (MYFFT_ENABLE_CUDA)
  list(APPEND MYFFT_LIB_SRC gpu/gpu_fft.cu)   # nvcc 编译单元:kernel 实例化 + 发射
endif()

add_library(myfft SHARED ${MYFFT_LIB_SRC})

target_compile_definitions(myfft PRIVATE MYFFT_BUILD)
if (MYFFT_ENABLE_CUDA)
  target_compile_definitions(myfft PRIVATE MYFFT_HAVE_CUDA)
  # 静态链接 cudart:发布产物仍只有一个 so/dll,不额外依赖 cudart 动态库
  target_link_libraries(myfft PRIVATE CUDA::cudart_static)
  set_target_properties(myfft PROPERTIES CUDA_VISIBILITY_PRESET hidden)
endif()

target_include_directories(myfft
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
          $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(myfft PROPERTIES
  VERSION   1            # ABI 主版本 = soname 版本,与 myfftVersion() 对齐
  SOVERSION 1)

if (MSVC)
  target_compile_options(myfft PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  target_compile_options(myfft PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
  set_target_properties(myfft PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()

# 安装布局照抄 CUDA:include/ 与 lib/ 分离
install(TARGETS myfft RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
