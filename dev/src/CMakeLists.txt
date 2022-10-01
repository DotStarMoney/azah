cmake_minimum_required(VERSION 3.2)
project(azah)
set(LOCAL_DIR ${CMAKE_CURRENT_LIST_DIR})

add_executable(azah)
target_link_libraries(azah eigen glog)

set(NN_ACTIVATION
    nn/activation.h
    nn/activation.cc)

set(NN_OPERATION
    nn/binary_op.h
	nn/constant.h
    nn/data_types.h
    nn/unary_op.h
	nn/node.h
	nn/op.h
    nn/op/add.h
    nn/op/matmul.h
    nn/op/multiply.h
    nn/op/swish.h
    nn/variable.h)

target_sources(azah PRIVATE
    main.cc
    ${NN_OPERATION}
    ${NN_ACTIVATION})

# ------------------------------------------------------------------------------

add_executable(azah_nn_activation_test
    ${NN_ACTIVATION}
    nn/activation_test.cc)
target_link_libraries(azah_nn_activation_test gtest gtest_main)
add_test(azah azah_nn_activation_test)
