cmake_minimum_required(VERSION 3.2)
project(azah)
set(LOCAL_DIR ${CMAKE_CURRENT_LIST_DIR})

add_executable(azah)
target_link_libraries(azah absl_flat_hash_map absl_hash absl_random_random
                      eigen glog)

set(SRC_GAMES_TICTACTOE_H
    games/tictactoe/tictactoe.h
    games/tictactoe/tictactoe_network.h)
set(SRC_GAMES_TICTACTOE_CC
    games/tictactoe/tictactoe.cc
    games/tictactoe/tictactoe_network.cc)
source_group("Header Files\\games\\tictactoe" FILES ${SRC_GAMES_TICTACTOE_H})
source_group("Source Files\\games\\tictactoe" FILES ${SRC_GAMES_TICTACTOE_CC})
set(SRC_GAMES_TICTACTOE
    ${SRC_GAMES_TICTACTOE_H}
    ${SRC_GAMES_TICTACTOE_CC})
    
set(SRC_GAMES_MANCALA_H
    games/mancala/mancala.h
    games/mancala/mancala_network.h)
set(SRC_GAMES_MANCALA_CC
    games/mancala/mancala.cc
    games/mancala/mancala_network.cc)
source_group("Header Files\\games\\mancala" FILES ${SRC_GAMES_MANCALA_H})
source_group("Source Files\\games\\mancala" FILES ${SRC_GAMES_MANCALA_CC})
set(SRC_GAMES_MANCALA
    ${SRC_GAMES_MANCALA_H}
    ${SRC_GAMES_MANCALA_CC})

set(SRC_GAMES_IGNOBLE_H
    games/ignoble/ignoble4.h
    games/ignoble/ignoble4_network.h)
set(SRC_GAMES_IGNOBLE_CC
    games/ignoble/ignoble4.cc
    games/ignoble/ignoble4_network.cc)
source_group("Header Files\\games\\ignoble" FILES ${SRC_GAMES_IGNOBLE_H})
source_group("Source Files\\games\\ignoble" FILES ${SRC_GAMES_IGNOBLE_CC})
set(SRC_GAMES_IGNOBLE
    ${SRC_GAMES_IGNOBLE_H}
    ${SRC_GAMES_IGNOBLE_CC})

set(SRC_GAMES_H
    games/game.h
    games/game_network.h)
set(SRC_GAMES_CC
    games/game_network.cc)
source_group("Header Files\\games" FILES ${SRC_GAMES_H})
source_group("Source Files\\games" FILES ${SRC_GAMES_CC})
set(SRC_GAMES
    ${SRC_GAMES_H}
    ${SRC_GAMES_CC}
    ${SRC_GAMES_TICTACTOE}
    ${SRC_GAMES_MANCALA}
    ${SRC_GAMES_IGNOBLE})

set(SRC_IO_H
    io/serializable.h)
source_group("Header Files\\io" FILES ${SRC_IO_H})
set(SRC_IO
    ${SRC_IO_H})

set(SRC_MCTS_H
    mcts/callbacks.h
    mcts/work_queue.h
    mcts/self_play.h
    mcts/rl_player.h)
source_group("Header Files\\mcts" FILES ${SRC_MCTS_H})
set(SRC_MCTS
    ${SRC_MCTS_H})

set(SRC_NN_ACTIVATION_H
    nn/activation.h)
set(SRC_NN_ACTIVATION_CC
    nn/activation.cc)
source_group("Header Files\\nn" FILES ${SRC_NN_ACTIVATION_H})
source_group("Source Files\\nn" FILES ${SRC_NN_ACTIVATION_CC})
set(SRC_NN_ACTIVATION
    ${SRC_NN_ACTIVATION_H}
    ${SRC_NN_ACTIVATION_CC})

set(SRC_NN_H
    nn/adam.h
    nn/binary_op.h
    nn/constant.h
    nn/constant_base.h
    nn/data_types.h
    nn/init.h
    nn/network.h
    nn/node.h
    nn/node_base.h
    nn/op.h
    nn/op/add.h
    nn/op/broadcast_add.h
    nn/op/broadcast_matmul.h
    nn/op/concat.h
    nn/op/concat_cols.h
    nn/op/fmadd.h
    nn/op/fork.h
    nn/op/group_matmul.h
    nn/op/layer_norm.h
    nn/op/matmul.h
    nn/op/mean.h
    nn/op/mixer.h
    nn/op/mse.h
    nn/op/multiply.h
    nn/op/row_mean.h
    nn/op/scalar_fmadd.h
    nn/op/scalar_mse.h
    nn/op/sigmoid.h
    nn/op/softmax.h
    nn/op/softmax_cross_ent.h
    nn/op/swish.h
    nn/op/tanh.h
    nn/op/transpose.h
    nn/sgd_optimizer.h
    nn/unary_op.h
    nn/variable.h
    nn/variable_base.h)
set(SRC_NN_CC
    nn/adam.cc
    nn/network.cc)
source_group("Header Files\\nn" FILES ${SRC_NN_H})
source_group("Source Files\\nn" FILES ${SRC_NN_CC})
set(SRC_NN
    ${SRC_NN_H}
    ${SRC_NN_CC}
    ${SRC_NN_ACTIVATION})

set(SRC_THREAD_H
    thread/dispatch_queue.h
    thread/semaphore.h)
set(SRC_THREAD_CC
    thread/semaphore.cc)
source_group("Header Files\\thread" FILES ${SRC_THREAD_H})
source_group("Source Files\\thread" FILES ${SRC_THREAD_CC})
set(SRC_THREAD
    ${SRC_THREAD_H}
    ${SRC_THREAD_CC})

target_sources(azah PRIVATE
    main.cc
    ${SRC_GAMES}
    ${SRC_IO}
    ${SRC_MCTS}
    ${SRC_NN}
    ${SRC_THREAD})

# ------------------------------------------------------------------------------

add_executable(azah_nn_activation_test
    ${SRC_NN_ACTIVATION}
    nn/activation_test.cc)
target_link_libraries(azah_nn_activation_test gtest gtest_main)
add_test(azah azah_nn_activation_test)
