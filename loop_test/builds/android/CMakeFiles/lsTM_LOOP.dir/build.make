# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.28.1/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.28.1/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/wangyonglin/w_producer/loop_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/wangyonglin/w_producer/loop_test/builds/android

# Include any dependencies generated for this target.
include CMakeFiles/lsTM_LOOP.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/lsTM_LOOP.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/lsTM_LOOP.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lsTM_LOOP.dir/flags.make

CMakeFiles/lsTM_LOOP.dir/src/main.cpp.o: CMakeFiles/lsTM_LOOP.dir/flags.make
CMakeFiles/lsTM_LOOP.dir/src/main.cpp.o: /Users/wangyonglin/w_producer/loop_test/src/main.cpp
CMakeFiles/lsTM_LOOP.dir/src/main.cpp.o: CMakeFiles/lsTM_LOOP.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/wangyonglin/w_producer/loop_test/builds/android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lsTM_LOOP.dir/src/main.cpp.o"
	/Users/wangyonglin/Library/Android/sdk/ndk/21.4.7075529/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --gcc-toolchain=/Users/wangyonglin/Library/Android/sdk/ndk/21.4.7075529/toolchains/llvm/prebuilt/darwin-x86_64 --sysroot=/Users/wangyonglin/Library/Android/sdk/ndk/21.4.7075529/toolchains/llvm/prebuilt/darwin-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/lsTM_LOOP.dir/src/main.cpp.o -MF CMakeFiles/lsTM_LOOP.dir/src/main.cpp.o.d -o CMakeFiles/lsTM_LOOP.dir/src/main.cpp.o -c /Users/wangyonglin/w_producer/loop_test/src/main.cpp

CMakeFiles/lsTM_LOOP.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/lsTM_LOOP.dir/src/main.cpp.i"
	/Users/wangyonglin/Library/Android/sdk/ndk/21.4.7075529/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --gcc-toolchain=/Users/wangyonglin/Library/Android/sdk/ndk/21.4.7075529/toolchains/llvm/prebuilt/darwin-x86_64 --sysroot=/Users/wangyonglin/Library/Android/sdk/ndk/21.4.7075529/toolchains/llvm/prebuilt/darwin-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/wangyonglin/w_producer/loop_test/src/main.cpp > CMakeFiles/lsTM_LOOP.dir/src/main.cpp.i

CMakeFiles/lsTM_LOOP.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/lsTM_LOOP.dir/src/main.cpp.s"
	/Users/wangyonglin/Library/Android/sdk/ndk/21.4.7075529/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang++ --target=aarch64-none-linux-android21 --gcc-toolchain=/Users/wangyonglin/Library/Android/sdk/ndk/21.4.7075529/toolchains/llvm/prebuilt/darwin-x86_64 --sysroot=/Users/wangyonglin/Library/Android/sdk/ndk/21.4.7075529/toolchains/llvm/prebuilt/darwin-x86_64/sysroot $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/wangyonglin/w_producer/loop_test/src/main.cpp -o CMakeFiles/lsTM_LOOP.dir/src/main.cpp.s

# Object files for target lsTM_LOOP
lsTM_LOOP_OBJECTS = \
"CMakeFiles/lsTM_LOOP.dir/src/main.cpp.o"

# External object files for target lsTM_LOOP
lsTM_LOOP_EXTERNAL_OBJECTS =

lsTM_LOOP: CMakeFiles/lsTM_LOOP.dir/src/main.cpp.o
lsTM_LOOP: CMakeFiles/lsTM_LOOP.dir/build.make
lsTM_LOOP: CMakeFiles/lsTM_LOOP.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/wangyonglin/w_producer/loop_test/builds/android/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable lsTM_LOOP"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lsTM_LOOP.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lsTM_LOOP.dir/build: lsTM_LOOP
.PHONY : CMakeFiles/lsTM_LOOP.dir/build

CMakeFiles/lsTM_LOOP.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lsTM_LOOP.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lsTM_LOOP.dir/clean

CMakeFiles/lsTM_LOOP.dir/depend:
	cd /Users/wangyonglin/w_producer/loop_test/builds/android && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/wangyonglin/w_producer/loop_test /Users/wangyonglin/w_producer/loop_test /Users/wangyonglin/w_producer/loop_test/builds/android /Users/wangyonglin/w_producer/loop_test/builds/android /Users/wangyonglin/w_producer/loop_test/builds/android/CMakeFiles/lsTM_LOOP.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/lsTM_LOOP.dir/depend

