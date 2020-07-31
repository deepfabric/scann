cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

project(scann)

# Settings
set(CMAKE_C_COMPILER "/usr/bin/clang-8" CACHE string "clang compiler" FORCE)
set(CMAKE_CXX_COMPILER "/usr/bin/clang++-8" CACHE string "clang compiler" FORCE)

set(CMAKE_THREAD_PREFER_PTHREAD ON)

IF(NOT CMAKE_BUILD_TYPE)
    message(STATUS "CMAKE_BUILD_TYPE is not set, set to default = RELWITHDEBINFO")
    set(CMAKE_BUILD_TYPE            "RELWITHDEBINFO")
ENDIF()
MESSAGE( STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )

set(CMAKE_CONFIGURATION_TYPES "RelWithDebInfo;Debug;Release;MinSizeRel" CACHE STRING "" FORCE)

set(MACHINE_FLAGS "-mavx2 -mfma -fsized-deallocation -w")

set(COMMON_WARNING_FLAGS "-Wall -Werror -Wnon-virtual-dtor -Wno-sign-compare -Wno-unused-variable -Wno-unused-function -Wno-unused-result")

set(CMAKE_BUILD_COLOR_MAKEFILE          ON)
set(CMAKE_CXX_FLAGS "-std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 ${COMMON_WARNING_FLAGS} ${MACHINE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE             "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO      "-O3 -g")
set(CMAKE_EXE_LINKER_FLAGS              "-static-libgcc -static-libstdc++")

# cmake -DCMAKE_BUILD_TYPE=Debug ..
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

message(STATUS "CXX_FLAGS:=${CMAKE_CXX_FLAGS}")

message(STATUS "CMAKE_MODULE_PATH:=${CMAKE_MODULE_PATH}")

# System Checking
find_package(Threads)
find_package(TensorFlow REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Abseil REQUIRED)


set(SCANN_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        CACHE PATH "Path to SCANN include files.")

MESSAGE(STATUS "SCANN Include Dir: ${SCANN_INCLUDE_DIRS}")


file(GLOB_RECURSE SCANN_SOURCE_FILES "*.h" "*.cc")
file(GLOB_RECURSE REMOVE_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/app/*")
list(REMOVE_ITEM SCANN_SOURCE_FILES ${REMOVE_SOURCE_FILES})

add_library(scann ${SCANN_SOURCE_FILES})

file(GLOB_RECURSE DEMO_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/app/*")
add_executable(demo ${DEMO_SOURCE_FILES})


target_include_directories(scann PUBLIC
        ${SCANN_INCLUDE_DIRS}
        ${TensorFlow_INCLUDE_DIR}
        ${Protobuf_INCLUDE_DIR}
        ${Abseil_INCLUDE_DIR})

target_link_libraries(scann PUBLIC
        z rt nsl
        ${CMAKE_THREAD_LIBS_INIT}
        ${TensorFlow_FRAMEWORK_LIBRARY}
        ${Protobuf_LIBRARIES}
        ${Abseil_LIBS})

target_link_libraries(demo scann)
