# Build system: ExecuTorch import/embedding pattern adapted from
# ../first_et_project/CMakeLists.txt; MIC_SOURCE (real/synthetic) option
# adapted from ../stm32f4_audio_examples/audio_in_to_wav/CMakeLists.txt.
# See README.md for the full rationale.

cmake_minimum_required(VERSION 3.20)
project(wakeword CXX C ASM)

# ------------------------------------------------------------------ #
#  Validation                                                        #
# ------------------------------------------------------------------ #
if(NOT DEFINED TARGET_PLATFORM)
    message(FATAL_ERROR
        "TARGET_PLATFORM must be set. Use a CMake preset:\n"
        "  cmake --preset host          (Linux host, validation without hardware)\n"
        "  cmake --preset stm32         (STM32F407 bare-metal, real PDM mic)\n"
        "  cmake --preset stm32-renode  (STM32F407 bare-metal, synthetic mic for Renode)")
endif()

if(NOT DEFINED EXECUTORCH_BUILD_DIR)
    message(FATAL_ERROR "EXECUTORCH_BUILD_DIR must be set (via preset).")
endif()

# ExecuTorch headers come from the source tree, not the build tree.
set(EXECUTORCH_SRC_DIR "$ENV{HOME}/executorch")
set(ET_LIB_DIR ${EXECUTORCH_BUILD_DIR})

message(STATUS "TARGET_PLATFORM:      ${TARGET_PLATFORM}")
message(STATUS "EXECUTORCH_BUILD_DIR: ${EXECUTORCH_BUILD_DIR}")

# ------------------------------------------------------------------ #
#  Imported ExecuTorch libraries (prebuilt; ExecuTorch 1.3.0 has no  #
#  CMake config package, so STATIC IMPORTED rather than find_package) #
# ------------------------------------------------------------------ #
add_library(executorch_core STATIC IMPORTED)
set_target_properties(executorch_core PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/libexecutorch_core.a)

add_library(executorch STATIC IMPORTED)
set_target_properties(executorch PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/libexecutorch.a)

add_library(executorch_selected_kernels STATIC IMPORTED)
set_target_properties(executorch_selected_kernels PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/libexecutorch_selected_kernels.a)

add_library(portable_kernels STATIC IMPORTED)
set_target_properties(portable_kernels PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/kernels/portable/libportable_kernels.a)

add_library(extension_data_loader STATIC IMPORTED)
set_target_properties(extension_data_loader PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/extension/data_loader/libextension_data_loader.a)

add_library(flatccrt STATIC IMPORTED)
set_target_properties(flatccrt PROPERTIES
    IMPORTED_LOCATION ${ET_LIB_DIR}/third-party/flatcc_ep/lib/libflatccrt.a)

set(ET_INCLUDES
    ${EXECUTORCH_SRC_DIR}
    ${EXECUTORCH_SRC_DIR}/executorch/third-party/flatbuffers/include
    ${ET_LIB_DIR}/third-party/flatcc_ep/include
    # torch/include needed for c10/util/irange.h referenced by exec_aten headers
    $ENV{HOME}/executorch/et-env/lib/python3.12/site-packages/torch/include
)

# EXECUTORCH_SELECT_OPS_LIST is intentionally NOT referenced anywhere in this
# file: it is a property of the prebuilt libexecutorch_selected_kernels.a in
# EXECUTORCH_BUILD_DIR, set when ExecuTorch itself is (re)built -- not
# something this project's CMake controls. See README.md "Building/rebuilding
# ExecuTorch" and README-training-parameters.md for how to determine the op
# list your model needs.

# ------------------------------------------------------------------ #
#  spectrogram_lib — shared by every target (host + stm32 + stm32-   #
#  renode all link the identical compiled C, which is what makes     #
#  tools/compare_spectrogram.py's cross-check meaningful end-to-end) #
# ------------------------------------------------------------------ #
add_library(spectrogram_lib STATIC
    src/spectrogram.c
    src/fft.c
)
target_include_directories(spectrogram_lib PUBLIC src)

# ------------------------------------------------------------------ #
#  Host (Linux) build — validates the feature pipeline + ExecuTorch  #
#  runtime against WAV files, without any hardware.                  #
# ------------------------------------------------------------------ #
if(TARGET_PLATFORM STREQUAL "host")

    add_executable(inference_host
        src/host_test_main.cpp
    )

    target_compile_options(inference_host PRIVATE -Wall -Wextra -g -Og)
    target_include_directories(inference_host PRIVATE ${ET_INCLUDES})

    # --whole-archive is required for executorch_selected_kernels because op
    # registration happens via static initializers (global constructors).
    # Without it the linker discards object files with no directly
    # referenced symbols, and OperatorMissing results at runtime.
    target_link_libraries(inference_host PRIVATE
        spectrogram_lib
        -Wl,--whole-archive
        executorch_selected_kernels
        -Wl,--no-whole-archive
        portable_kernels
        executorch
        executorch_core
        extension_data_loader
        flatccrt
    )

# ------------------------------------------------------------------ #
#  STM32F407 (bare-metal Cortex-M4) build                            #
# ------------------------------------------------------------------ #
elseif(TARGET_PLATFORM STREQUAL "stm32")

    # "real"      -> pdm_mic.c        (I2S2/DMA capture from the onboard mic)
    # "synthetic" -> synthetic_mic.c  (WAV-derived playback, for Renode --
    #                Renode has no PDM microphone model)
    # Both referenced via relative path from the shared stm32f4_audio_examples
    # repo, never copied -- see README.md.
    set(MIC_SOURCE "real" CACHE STRING "Audio source: real or synthetic")
    set_property(CACHE MIC_SOURCE PROPERTY STRINGS real synthetic)

    set(AUDIO_EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/../stm32f4_audio_examples/audio_in_to_wav)

    if(MIC_SOURCE STREQUAL "real")
        set(MIC_SRC ${AUDIO_EXAMPLES_DIR}/src/pdm_mic.c)
    elseif(MIC_SOURCE STREQUAL "synthetic")
        set(MIC_SRC ${AUDIO_EXAMPLES_DIR}/src/synthetic_mic.c)
    else()
        message(FATAL_ERROR "MIC_SOURCE must be 'real' or 'synthetic', got: ${MIC_SOURCE}")
    endif()

    # Which trained model gets embedded in firmware. Only ImprovedKWSNet is
    # meant to be flashed (ExactMatchCNN is host-only, see
    # README-training-parameters.md) -- defaults to the submission layout's
    # expected export location (see README-submission-instructions.md).
    set(WAKEWORD_PTE "${CMAKE_SOURCE_DIR}/submission/model.pte"
        CACHE FILEPATH "Path to the .pte model embedded in firmware")

    set(CPU_FLAGS "-mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard")

    # -g adds DWARF debug info to firmware.elf only -- arm-none-eabi-objcopy
    # -O binary (below) strips it when producing firmware.bin, so this has
    # zero cost on the flashed image. You'll want this for source-level GDB
    # debugging (breakpoints by file:line, printing local variables by name)
    # -- e.g. reading VAD state for submission/debug_values.json.
    set(CMAKE_C_FLAGS
        "${CPU_FLAGS} -Os -g -ffunction-sections -fdata-sections -mno-unaligned-access -DTARGET_STM32 -DET_LOG_ENABLED=0"
        CACHE STRING "" FORCE)
    set(CMAKE_CXX_FLAGS
        "${CPU_FLAGS} -Os -g -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -std=c++17 -mno-unaligned-access -DTARGET_STM32 -DET_LOG_ENABLED=0"
        CACHE STRING "" FORCE)
    set(CMAKE_ASM_FLAGS "${CPU_FLAGS} -g" CACHE STRING "" FORCE)

    #
    # Embed the .pte as a C array at configure time (xxd -i), with the
    # alignment/const post-processing ExecuTorch's FlatBuffer parser
    # requires. See ../first_et_project/CMakeLists.txt for the original
    # pattern this is copied from.
    #
    get_filename_component(WAKEWORD_PTE_DIR ${WAKEWORD_PTE} DIRECTORY)
    get_filename_component(WAKEWORD_PTE_NAME ${WAKEWORD_PTE} NAME)
    string(REPLACE "." "_" WAKEWORD_PTE_VARNAME ${WAKEWORD_PTE_NAME})

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E chdir ${WAKEWORD_PTE_DIR}
            xxd -i ${WAKEWORD_PTE_NAME}
        OUTPUT_VARIABLE XXD_OUTPUT
        RESULT_VARIABLE XXD_RESULT
    )
    if(NOT XXD_RESULT EQUAL 0)
        message(FATAL_ERROR
            "xxd failed. Is ${WAKEWORD_PTE} present?\n"
            "Run: python3 models/train_wakeword.py --model improved  (see README-training-parameters.md)")
    endif()

    string(REPLACE
        "unsigned char ${WAKEWORD_PTE_VARNAME}[]"
        "__attribute__((aligned(16))) const unsigned char ${WAKEWORD_PTE_VARNAME}[]"
        XXD_OUTPUT "${XXD_OUTPUT}")
    string(REPLACE
        "unsigned int ${WAKEWORD_PTE_VARNAME}_len"
        "const unsigned int ${WAKEWORD_PTE_VARNAME}_len"
        XXD_OUTPUT "${XXD_OUTPUT}")
    # Normalize the generated symbol names to fixed names main.cpp includes,
    # regardless of which .pte file WAKEWORD_PTE points at.
    string(REPLACE "${WAKEWORD_PTE_VARNAME}_len" "wakeword_pte_len" XXD_OUTPUT "${XXD_OUTPUT}")
    string(REPLACE "${WAKEWORD_PTE_VARNAME}" "wakeword_pte" XXD_OUTPUT "${XXD_OUTPUT}")

    file(WRITE ${CMAKE_BINARY_DIR}/wakeword_pte.h "${XXD_OUTPUT}")
    message(STATUS "Generated wakeword_pte.h from ${WAKEWORD_PTE} in ${CMAKE_BINARY_DIR}")

    add_executable(firmware.elf
        startup_stm32f407.s
        src/syscalls.c
        src/main.cpp
        ${MIC_SRC}
        ${AUDIO_EXAMPLES_DIR}/src/hal.c
    )

    target_link_libraries(firmware.elf PRIVATE spectrogram_lib)

    target_include_directories(firmware.elf PRIVATE
        ${ET_INCLUDES}
        ${CMAKE_BINARY_DIR}          # generated wakeword_pte.h
        src
        ${AUDIO_EXAMPLES_DIR}/src    # pdm_mic.h / hal.h / stm32f407.h
    )

    target_link_libraries(firmware.elf PRIVATE
        -Wl,--whole-archive
        executorch_selected_kernels
        -Wl,--no-whole-archive
        portable_kernels
        executorch
        executorch_core
        extension_data_loader
        flatccrt
    )

    # -nostartfiles excludes crti.o/crtn.o; startup_stm32f407.s provides its
    # own _init/_fini stubs instead (see startup_stm32f407.s comments).
    target_link_options(firmware.elf PRIVATE
        -nostartfiles
        -specs=nano.specs
        -T${CMAKE_SOURCE_DIR}/stm32f407.ld
        -Wl,--gc-sections
        -lc -lm -lgcc
    )

    add_custom_command(TARGET firmware.elf POST_BUILD
        COMMAND arm-none-eabi-objcopy -O binary firmware.elf firmware.bin
        COMMENT "Generating firmware.bin"
    )
    add_custom_command(TARGET firmware.elf POST_BUILD
        COMMAND arm-none-eabi-size firmware.elf
        COMMENT "Firmware size:"
    )

    add_custom_target(flash
        COMMAND openocd
            -f interface/stlink.cfg
            -f target/stm32f4x.cfg
            -c "transport select hla_swd"
            -c "program ${CMAKE_BINARY_DIR}/firmware.bin 0x08000000 verify reset exit"
        DEPENDS firmware.elf
        COMMENT "Flashing firmware via OpenOCD"
        USES_TERMINAL
    )

    add_custom_target(disasm
        COMMAND arm-none-eabi-objdump -d -S ${CMAKE_BINARY_DIR}/firmware.elf
        DEPENDS firmware.elf
        COMMENT "Disassembly:"
        USES_TERMINAL
    )

else()
    message(FATAL_ERROR
        "Unknown TARGET_PLATFORM: '${TARGET_PLATFORM}'. Must be 'host' or 'stm32'.")
endif()
