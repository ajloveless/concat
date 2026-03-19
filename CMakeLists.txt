cmake_minimum_required(VERSION 3.19)
project(concat_tilde)

# -----------------------------------------------------------------------
# User must set MAX_SDK_PATH to the root of the Max SDK
# (containing source/c74support/max-includes and source/c74support/msp-includes)
# e.g.:  cmake -DMAX_SDK_PATH=/path/to/max-sdk ..
# -----------------------------------------------------------------------

if(NOT DEFINED MAX_SDK_PATH)
    # Try a few common locations
    set(MAX_SDK_CANDIDATES
        "$ENV{HOME}/Documents/Max\ SDK"
        "/Applications/Max.app/Contents/Resources/C74"
        "${CMAKE_SOURCE_DIR}/../max-sdk"
    )
    foreach(candidate ${MAX_SDK_CANDIDATES})
        if(EXISTS "${candidate}/source/c74support/max-includes/ext.h")
            set(MAX_SDK_PATH "${candidate}")
            break()
        endif()
    endforeach()
    if(NOT DEFINED MAX_SDK_PATH OR NOT EXISTS "${MAX_SDK_PATH}")
        message(FATAL_ERROR
            "Max SDK not found.  Set -DMAX_SDK_PATH=/path/to/max-sdk")
    endif()
endif()

set(MAX_INCLUDES  "${MAX_SDK_PATH}/source/c74support/max-includes")
set(MSP_INCLUDES  "${MAX_SDK_PATH}/source/c74support/msp-includes")

# Verify headers exist
foreach(inc "${MAX_INCLUDES}/ext.h" "${MSP_INCLUDES}/z_dsp.h")
    if(NOT EXISTS "${inc}")
        message(FATAL_ERROR "Cannot find Max SDK header: ${inc}")
    endif()
endforeach()

# -----------------------------------------------------------------------
# Platform settings
# -----------------------------------------------------------------------
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
    set(EXTERN_SUFFIX ".mxo")
    set(PLATFORM_LINK_FLAGS "-framework CoreAudio -framework AudioUnit")
elseif(WIN32)
    set(EXTERN_SUFFIX ".mxe64")
    set(PLATFORM_LINK_FLAGS "")
endif()

# -----------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------
set(CONCAT_SOURCES
    source/concat_tilde.c
    source/concat_task.c
    source/concat_buffer.c
    source/threadpooltask.c
    source/particle.c
    source/observer.c
    source/propagator.c
    source/audio_features.c
    source/corpus.c
    source/utils.c
)

# -----------------------------------------------------------------------
# Target: concat~ external
# -----------------------------------------------------------------------
add_library(concat_tilde MODULE ${CONCAT_SOURCES})

set_target_properties(concat_tilde PROPERTIES
    OUTPUT_NAME        "concat~"
    PREFIX             ""
    SUFFIX             "${EXTERN_SUFFIX}"
    C_STANDARD         11
    C_STANDARD_REQUIRED YES
)

target_include_directories(concat_tilde PRIVATE
    "${MAX_INCLUDES}"
    "${MSP_INCLUDES}"
    "${CMAKE_SOURCE_DIR}/source"
)

target_compile_definitions(concat_tilde PRIVATE
    $<$<BOOL:${APPLE}>:MACOSX_DEPLOYMENT_TARGET=10.13>
)

target_compile_options(concat_tilde PRIVATE
    $<$<C_COMPILER_ID:AppleClang,Clang,GNU>:-Wall -Wno-unused-function -O2>
    $<$<C_COMPILER_ID:MSVC>:/W3 /O2>
)

if(APPLE)
    target_link_options(concat_tilde PRIVATE
        -bundle
        -undefined dynamic_lookup
        ${PLATFORM_LINK_FLAGS}
    )
endif()

# -----------------------------------------------------------------------
# Install target (copies .mxo into Max's Packages folder on macOS)
# Override by passing -DMAX_PACKAGES_PATH=/path/to/Packages to cmake
# -----------------------------------------------------------------------
if(APPLE)
    if(NOT DEFINED MAX_PACKAGES_PATH)
        # Search for the first existing Max version packages folder
        foreach(ver "Max 9" "Max 8")
            set(_candidate "$ENV{HOME}/Documents/${ver}/Packages")
            if(EXISTS "${_candidate}")
                set(MAX_PACKAGES_PATH "${_candidate}")
                break()
            endif()
        endforeach()
        # Fallback to a generic path if nothing was found
        if(NOT DEFINED MAX_PACKAGES_PATH)
            set(MAX_PACKAGES_PATH "$ENV{HOME}/Documents/Max/Packages")
        endif()
    endif()
    install(TARGETS concat_tilde
            LIBRARY DESTINATION "${MAX_PACKAGES_PATH}/concat/externals")
endif()
