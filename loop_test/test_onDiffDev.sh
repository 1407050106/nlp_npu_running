#!/bin/bash
set -e
echo "build mnn_test_program now!"

AIM_PLATFORM=$1
if [ -z "$1" ]; then 
    echo "Error: No target platform specified."
    exit 1; 
fi

TARGET_PLATFORM=mac
if [ "$AIM_PLATFORM" = 'android' ]; then
    TARGET_PLATFORM=android
elif [ "$AIM_PLATFORM" = 'harmony' ]; then
    TARGET_PLATFORM=harmony
fi
#echo "AIM_PLATFORM: $AIM_PLATFORM, TARGET_PLATFORM: $TARGET_PLATFORM"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#echo "ROOT_DIR: $ROOT_DIR"

if [ -d builds/$TARGET_PLATFORM ]; then
    rm -rf builds/$TARGET_PLATFORM
fi

ARM_ABI_64="arm64-v8a"
mkdir -p builds/$TARGET_PLATFORM

build_mac_target() {
    cd $ROOT_DIR/builds/$TARGET_PLATFORM

    cmake -DTARGET_PLATFORM=MAC \
        ../../
    if [ $? -ne 0 ]; then
        echo "Cmake failed! Error code = $?"
        exit 1
    fi
    cmake --build . --config Release
    if [ $? -ne 0 ]; then
        echo "Cmake build failed! Error code = $?"
        exit 1
    fi
}

build_android_target() {
    target_arch=$1
    cd $ROOT_DIR/builds/$TARGET_PLATFORM
    NDK_PATH1=$(which ndk-build | sed 's/ndk-build//g')
    if [[ "_$NDK_PATH1" != '_' ]]; then
        export ANDROID_NDK=$NDK_PATH1
    else
        echo "Android NDK path not found. Please set ANDROID_NDK environment variable."
        exit 1
    fi

    cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DANDROID_ABI=$target_arch \
        -DANDROID_ARM_NEON=ON \
        -DANDROID_PLATFORM=android-21 \
        -DANDROID_STL=c++_static \
        -DANDROID_TOOLCHAIN=clang \
        -DTARGET_PLATFORM=ANDROID \
        ../../

    if [ $? -ne 0 ]; then
        echo "Cmake failed! Error code = $?"
        exit 1
    fi
    cmake --build . --config Release
    if [ $? -ne 0 ]; then
        echo "Cmake build failed! Error code = $?"
        exit 1
    fi
}

build_harmony_target() {
    target_arch=$1
    cd $ROOT_DIR/builds/$TARGET_PLATFORM
    NDK_PATH2=$(dirname $(dirname $(which hdc)))
    if [[ "_$NDK_PATH2" != '_' ]]; then
        export HARMONY_NDK=$NDK_PATH2
    else
        echo "Harmony NDK path not found. Please set HARMONY_NDK environment variable."
        exit 1
    fi

    cmake -DCMAKE_TOOLCHAIN_FILE=$HARMONY_NDK/native/build/cmake/ohos.toolchain.cmake \
        -DHARMONYOS=1 \
        -DCMAKE_BUILD_TYPE=Release \
        -DTARGET_PLATFORM=HARMONY \
        -DOHOS_PLATFORM=OHOS \
        -DOHOS_STL=c++_static \
        -DOHOS_ARCH=$target_arch \
        ../../

    if [ $? -ne 0 ]; then
        echo "Cmake failed! Error code = $?"
        exit 1
    fi
    cmake --build . --config Release
    if [ $? -ne 0 ]; then
        echo "Cmake build failed! Error code = $?"
        exit 1
    fi
}

if [ "$AIM_PLATFORM" = 'android' ]; then
    build_android_target $ARM_ABI_64
elif [ "$AIM_PLATFORM" = 'harmony' ]; then
    build_harmony_target $ARM_ABI_64
else
    build_mac_target
fi