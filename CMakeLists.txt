cmake_minimum_required(VERSION 3.5)
# 将test.cc 文件 存储在 cc_file变量中
project(SeAFusion)
file(GLOB cc_file seafusion.cc)

# 设置 SDK_DIR 目录
# DLICC_PATH: /dl/sdk/bin  SDK_DIR /dl/sdk/
# set(SDK_DIR $ENV{DLICC_PATH}/../)
set(SDK_DIR "/dl/sdk/")
set(OPENCV_DIR "/usr/include/opencv4/opencv2/")
set(DLTOOLS_DIR "/home/firefly/Tingshuo/DlDeepTookit/c++/install/")

# 添加头文件目录
# /dl/sdk/include /dl/include/dlnne \
include_directories(${DLTOOLS_DIR}/include/dlcuda)
include_directories(${DLTOOLS_DIR}/include/dlnne)
include_directories(${DLTOOLS_DIR}/include/dlpreprocess)
include_directories(${DLTOOLS_DIR}/include/dlutil)
include_directories(${DLTOOLS_DIR}/include/dlvid)

include_directories(${SDK_DIR}/include)
include_directories(${SDK_DIR}/include/dlnne)
include_directories(${OPENCV_DIR})


set(CXX_FLAGS -std=c++11 -fPIC -fpermissive -Wno-attributes)
set(CMAKE_CXX_FLAGS "-g")
set(CMAKE_C_FLAGS "-g")


add_compile_options(${CXX_FLAGS})
# add_library(${PROJECT_NAME} SHARED ${dlnne_srcs})

# 设置目标属性  使用C++ 链接器来链接 ${Target} 目标。
set_target_properties(${Target} PROPERTIES LINKER_LANGUAGE CXX)

find_package(OpenCV 4.0)

# /dl/sdk/lib   /home/firefly/Tingshuo/SeAFusion/build
link_directories(${SDK_DIR}/lib ${CMAKE_BINARY_DIR})
link_directories(${DLTOOLS_DIR}/lib ${CMAKE_BINARY_DIR})

add_executable(test ${cc_file})
target_compile_options(test PRIVATE -I${SDK_DIR}/include -I${CMAKE_SOURCE_DIR} ${CXX_FLAGS})
target_link_libraries(test PRIVATE curt dlnne ${OpenCV_LIBS})
