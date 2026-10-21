add_executable(example_1d example_1d.cpp)
target_link_libraries(example_1d PRIVATE myfft)
add_custom_command(TARGET example_1d POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:myfft> $<TARGET_FILE_DIR:example_1d>)
