cmake_minimum_required(VERSION 3.2)
project(azah)
set(LOCAL_DIR ${CMAKE_CURRENT_LIST_DIR})

add_executable(azah)
target_link_libraries(azah eigen glog absl_flat_hash_map absl_hash)

set(SRC_NN_ACTIVATION
    nn/activation.h
    nn/activation.cc)

set(SRC_NN_CORE
    nn/adam.h
    nn/adam.cc
    nn/binary_op.h
    nn/constant.h
    nn/constant_base.h
    nn/data_types.h
    nn/init.h
    nn/network.h
    nn/network.cc
    nn/node.h
    nn/node_base.h
    nn/op.h
    nn/op/add.h
    nn/op/broadcast_matmul.h
    nn/op/concat.h
    nn/op/fmadd.h
    nn/op/fork.h
    nn/op/group_matmul.h
    nn/op/layer_norm.h
    nn/op/matmul.h
    nn/op/mean.h
    nn/op/mse.h
    nn/op/multiply.h
    nn/op/row_mean.h
    nn/op/scalar_fmadd.h
    nn/op/scalar_mse.h
    nn/op/scalar_sub.h
    nn/op/sigmoid.h
    nn/op/softmax.h
    nn/op/softmax_cross_ent.h
    nn/op/square_mean.h
    nn/op/swish.h
    nn/op/tanh.h
    nn/op/transpose.h
    nn/op/scalar_inv_sqrt.h
    nn/sgd_optimizer.h
    nn/unary_op.h
    nn/variable.h
    nn/variable_base.h
    ${SRC_NN_ACTIVATION})

set(SRC_MCTS
    mcts/state_cache.h
    mcts/visit_table.h)

target_sources(azah PRIVATE
    main.cc
    ${SRC_NN_CORE}
    ${SRC_MCTS})

# ------------------------------------------------------------------------------

add_executable(azah_nn_activation_test
    ${SRC_NN_ACTIVATION}
    nn/activation_test.cc)
target_link_libraries(azah_nn_activation_test gtest gtest_main)
add_test(azah azah_nn_activation_test)
