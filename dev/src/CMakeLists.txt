cmake_minimum_required(VERSION 3.2)
project(azah)
set(LOCAL_DIR ${CMAKE_CURRENT_LIST_DIR})

add_executable(azah)
target_link_libraries(azah eigen glog)

set(NN_ACTIVATION
    nn/activation.h
    nn/activation.cc)

set(NN_CORE
    nn/binary_op.h
    nn/constant.h
    nn/constant_base.h
    nn/data_types.h
    nn/network.h
    nn/network.cc
    nn/node.h
    nn/node_base.h
    nn/op.h
    nn/op/add.h
    nn/op/concat.h
    nn/op/fork.h
    nn/op/group_matmul.h
    nn/op/layer_norm.h
    nn/op/matmul.h
    nn/op/mean.h
    nn/op/multiply.h
    nn/op/scalar_fmadd.h
    nn/op/scalar_mse.h
    nn/op/sigmoid.h
    nn/op/softmax.h
    nn/op/softmax_cross_ent.h
    nn/op/swish.h
    nn/op/tanh.h
    nn/op/transpose.h
    nn/op/z_score.h
    nn/unary_op.h
    nn/init.h
    nn/variable.h
    nn/variable_base.h)

target_sources(azah PRIVATE
    main.cc
    ${NN_CORE}
    ${NN_ACTIVATION}
    ${NN_NETWORK})

# ------------------------------------------------------------------------------

add_executable(azah_nn_activation_test
    ${NN_ACTIVATION}
    nn/activation_test.cc)
target_link_libraries(azah_nn_activation_test gtest gtest_main)
add_test(azah azah_nn_activation_test)
