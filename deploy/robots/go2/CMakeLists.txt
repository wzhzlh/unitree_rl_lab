cmake_minimum_required(VERSION 3.12)
project(go2_controller)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

include_directories(
  /usr/include/eigen3
  /home/zhangjiayi/unitree_sdk2/include # unitree_sdk2
  /home/zhangjiayi/unitree_sdk2/thirdparty/include # dds thirdparty
  /home/zhangjiayi/unitree_sdk2/thirdparty/include/ddscxx # ddscxx include
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_directories(
  /home/zhangjiayi/unitree_sdk2/lib/x86_64
  /home/zhangjiayi/unitree_sdk2/thirdparty/lib/x86_64
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib
)

file(GLOB_RECURSE ADD_SRC_LIST
  src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})

add_executable(go2_ctrl main.cpp)

target_link_libraries(go2_ctrl
  ${PROJECT_NAME}_lib
  unitree_sdk2 ddsc ddscxx rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0
)