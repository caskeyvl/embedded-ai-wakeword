# arm-none-eabi.cmake
# CMake toolchain file for ARM Cortex-M4 bare-metal cross-compilation.
#
# Place this at:
#   ~/executorch/executorch/cmake/toolchains/arm-none-eabi.cmake
#
# Referenced by CMakePresets.json stm32 preset.

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER   arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)

# Cortex-M4 with hardware FPU (FPv4-SP, single-precision only)
# -mfloat-abi=hard: use FPU registers for float args (ABI must match libraries)
set(CMAKE_C_FLAGS_INIT
    "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mno-unaligned-access -fno-pic -fno-pie")
set(CMAKE_CXX_FLAGS_INIT
    "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mno-unaligned-access -fno-pic -fno-pie")

# Prevent CMake from linking a test executable against the host toolchain
# during compiler detection — we are cross-compiling, output is not runnable.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Search for headers/libraries only in the sysroot, not on the host.
# Search for build tools (cmake, python) only on the host.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
