# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

project("rknndetector")

set(CMAKE_CXX_STANDARD 17)

# 1. 包含 RKNN API 头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 2. 导入预编译的 NPU 运行时库 (librknnrt.so)
add_library(rknnrt SHARED IMPORTED)
set_target_properties(rknnrt PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/librknnrt.so)

# 3. 编译本地 JNI 动态库 (librknndetector.so)
add_library(${CMAKE_PROJECT_NAME} SHARED
        native-lib.cpp
        postprocess.cpp)

# 4. 链接系统库与 RKNN 库
find_library(log-lib log)

target_link_libraries(${CMAKE_PROJECT_NAME}
        rknnrt
        ${log-lib}
        android
        z)
