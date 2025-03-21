cmake_minimum_required(VERSION 2.8.12)
set(PROJECT_NAME lsTM_LOOP)

set(CMAKE_CXX_STANDARD 14)

if(${TARGET_PLATFORM} STREQUAL MAC)
    set(ON_MAC TRUE)
elseif(${TARGET_PLATFORM} STREQUAL ANDROID)
    set(ON_ANDROID TRUE)
elseif(${TARGET_PLATFORM} STREQUAL HARMONY)
    set(ON_HARMONY TRUE)
endif()

if (ON_MAC)
    set(TOOLCHAIN_DIR /usr/bin)
    set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/clang)
    set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/clang++)
    set(AIM_PLATFORM_NAME mac)

    add_executable(${PROJECT_NAME} main.cpp)
elseif (ON_ANDROID)
    include(3rdParty/android.toolchain.cmake)
    add_compile_options (
        -O3
        -DARM_NEON_ENABLE
        -DGL_SILENCE_DEPRECATION
        -Wno-deprecated-declarations
        -fvisibility=hidden
    )
    set(AIM_PLATFORM_NAME android)

    add_executable(${PROJECT_NAME} src/Threadpool.hpp src/tool.hpp src/main.cpp)
elseif (ON_HARMONY)
    set(AIM_PLATFORM_NAME harmony)
    add_definitions(-D__HARMONY__)
    file(GLOB LIB_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*)

    add_library(${PROJECT_NAME} ${LIB_SRCS})
endif()

set(THIRD_PARTY_PATH ${CMAKE_SOURCE_DIR}/3rdParty)
# find_library(LOG_LIB log)

target_include_directories(${PROJECT_NAME} PRIVATE ${THIRD_PARTY_PATH}/includes)
target_link_directories(${PROJECT_NAME} PRIVATE ${THIRD_PARTY_PATH}/libs/android)

target_link_libraries(${PROJECT_NAME} PUBLIC MNN)
if (ON_ANDROID)
    target_link_libraries(${PROJECT_NAME} PUBLIC
        log
        z
        android)
endif()
if (ON_HARMONY)
    target_link_libraries(${PROJECT_NAME} PUBLIC 
        libhilog_ndk.z.so)
endif()

# file(GLOB LIB_FILES
#     "${CMAKE_SOURCE_DIR}/3rdParty/libs/${AIM_PLATFORM_NAME}/*"
# )
# foreach(file ${LIB_FILES})
#     file(COPY ${file} DESTINATION ${CMAKE_SOURCE_DIR}/builds/${AIM_PLATFORM_NAME})
# endforeach()
# file(COPY ${CMAKE_SOURCE_DIR}/bankcard_detect_fp16.mnn DESTINATION ${CMAKE_SOURCE_DIR}/builds/${AIM_PLATFORM_NAME})
# file(COPY ${CMAKE_SOURCE_DIR}/7.jpeg DESTINATION ${CMAKE_SOURCE_DIR}/builds/${AIM_PLATFORM_NAME})

# cmake ../ -DCMAKE_TOOLCHAIN_FILE=/Users/wangyonglin/Library/Android/sdk/ndk/21.4.7075529//build/cmake/android.toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21 -DANDROID_STL=c++_static -DANDROID_TOOLCHAIN=clang  -DTARGET_PLATFORM=ANDROID
# cmake ../ -DCMAKE_TOOLCHAIN_FILE=/Users/wangyonglin/Library/Huawei/Sdk/openharmony/9/native/build/cmake/ohos.toolchain.cmake -DHARMONYOS=1 -DCMAKE_BUILD_TYPE=Release  -DTARGET_PLATFORM=HARMONY -DOHOS_PLATFORM=OHOS -DOHOS_STL=c++_static -DOHOS_ARCH=arm64-v8a
