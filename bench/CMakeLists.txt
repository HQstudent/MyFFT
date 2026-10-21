add_executable(bench_1d bench_1d.cpp)
target_link_libraries(bench_1d PRIVATE myfft)
add_custom_command(TARGET bench_1d POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:myfft> $<TARGET_FILE_DIR:bench_1d>)
