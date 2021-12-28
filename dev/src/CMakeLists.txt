cmake_minimum_required(VERSION 3.2)
project(azah)
set(LOCAL_DIR ${CMAKE_CURRENT_LIST_DIR})

add_executable(azah)
target_link_libraries(azah absl::container absl::strings glog eigen)

set(NN_ACTIVATION
  nn/activation.h
  nn/activation.cc)

set(NN_BUILDER
  nn/builder.h
  nn/builder.cc)

set(NN_GRAPH
  nn/graph.h
  nn/graph.cc)

set(NN_CORE
  nn/core.h)

target_sources(azah PRIVATE
  main.cc
  ${NN_ACTIVATION}
  ${NN_BUILDER}
  ${NN_GRAPH}
  ${NN_CORE})

# ------------------------------------------------------------------------------

add_executable(azah_nn_activation_test
  ${NN_ACTIVATION}
  nn/activation_test.cc)
target_link_libraries(azah_nn_activation_test gtest gtest_main)
add_test(azah azah_nn_activation_test)
