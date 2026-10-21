add_executable(test_correctness test_correctness.cpp ref_dft.h)
target_link_libraries(test_correctness PRIVATE myfft)
add_custom_command(TARGET test_correctness POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:myfft> $<TARGET_FILE_DIR:test_correctness>)

# M1:同一个测试二进制跑两种后端——默认 GPU,MYFFT_BACKEND=cpu 走 CPU 黄金参考。
# 两条测试即构成 GPU vs CPU 的对拍(各自独立对拍朴素 DFT)。
add_test(NAME correctness COMMAND test_correctness)
if (MYFFT_ENABLE_CUDA)
  set_tests_properties(correctness PROPERTIES ENVIRONMENT "MYFFT_BACKEND=gpu")
  add_test(NAME correctness_cpu COMMAND test_correctness)
  set_tests_properties(correctness_cpu PROPERTIES ENVIRONMENT "MYFFT_BACKEND=cpu")
endif()
