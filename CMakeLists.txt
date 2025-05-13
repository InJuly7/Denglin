cmake_minimum_required(VERSION 3.5)
project(Yolov5s)

# 设置 SDK_DIR 目录
# DLICC_PATH: /dl/sdk/bin  SDK_DIR /dl/sdk/
# set(SDK_DIR $ENV{DLICC_PATH}/../)
set(SDK_DIR "/dl/sdk/")
set(OPENCV_DIR "/usr/include/opencv4/opencv2/")
set(DLTOOLS_DIR "/home/firefly/Tingshuo/DlDeepTookit/c++/install/")
set(CMAKE_C_COMPILER "/dl/sdk/bin/dlcc")
set(CMAKE_CXX_COMPILER "/dl/sdk/bin/dlcc")


message(STATUS CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB CPPS 
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/yolov5.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/kernel_function.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/utils.cpp
)
message(STATUS CPPS = ${CPPS})


# 添加头文件目录
# include_directories(${DLTOOLS_DIR}/include/dlcuda)
# include_directories(${DLTOOLS_DIR}/include/dlnne)
# include_directories(${DLTOOLS_DIR}/include/dlpreprocess)
# include_directories(${DLTOOLS_DIR}/include/dlutil)
# include_directories(${DLTOOLS_DIR}/include/dlvid)

# /dl/sdk/include /dl/include/dlnne 
include_directories(${SDK_DIR}/include)
include_directories(${SDK_DIR}/include/dlnne)
include_directories(${OPENCV_DIR})
find_package(OpenCV 4.0)


set(CXX_FLAGS -std=c++11 -fPIC -fpermissive -Wno-attributes)
set(CMAKE_CXX_FLAGS "-g")
set(CMAKE_C_FLAGS "-g")

add_compile_options(${CXX_FLAGS})
# add_library(${PROJECT_NAME} SHARED ${dlnne_srcs})

# 设置目标属性  使用C++ 链接器来链接 ${Target} 目标。
set_target_properties(${Target} PROPERTIES LINKER_LANGUAGE CXX)

# /dl/sdk/lib   /home/firefly/Tingshuo/SeAFusion/build
link_directories(${SDK_DIR}/lib ${CMAKE_BINARY_DIR})
link_directories(${DLTOOLS_DIR}/lib ${CMAKE_BINARY_DIR})

add_executable(test ${CPPS})
target_compile_options(test PRIVATE -x cuda --cuda-gpu-arch=dlgpuc64 ${CXX_FLAGS})
target_link_libraries(test PRIVATE curt dlnne ${OpenCV_LIBS})
