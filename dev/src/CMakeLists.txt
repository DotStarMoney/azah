cmake_minimum_required(VERSION 3.2)
project(azah)
set(LOCAL_DIR ${CMAKE_CURRENT_LIST_DIR})

add_executable(azah)
target_link_libraries(azah absl::base absl::flat_hash_map absl::strings glog eigen)
include_directories(${LOCAL_DIR})

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

set(UTIL_NONCOPYABLE
  util/noncopyable.h)

set(UTIL_RANDOM
  util/random.h
  util/random.cc)

set(ASYNC_SEMAPHOR
  async/semaphor.h
  async/semaphor.cc)

set(ASYNC_GATEWAY
  async/gateway.h
  async/gateway.cc)
 
set(ASYNC_WORK_QUEUE
  async/work_queue.h
  async/work_queue.cc)
 
target_sources(azah PRIVATE
  main.cc
  ${NN_ACTIVATION}
  ${NN_BUILDER}
  ${NN_GRAPH}
  ${NN_CORE}
  ${UTIL_NONCOPYABLE}
  ${UTIL_RANDOM}
  ${ASYNC_SEMAPHOR}
  ${ASYNC_GATEWAY}
  ${ASYNC_WORK_QUEUE})

# ------------------------------------------------------------------------------

add_executable(azah_nn_activation_test
  ${NN_ACTIVATION}
  nn/activation_test.cc)
target_link_libraries(azah_nn_activation_test gtest gtest_main)
add_test(azah azah_nn_activation_test)
