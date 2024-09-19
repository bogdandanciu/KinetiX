cmake_minimum_required(VERSION 3.21)
project(kinetix LANGUAGES C CXX)

cmake_policy(SET CMP0079 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX $ENV{HOME}/.local/kinetix CACHE PATH "Default install path" FORCE)
ENDIF()


if(${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  MESSAGE(FATAL_ERROR "In-place builds are not supported. Please create a separate build directory")
endif(${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

set(ENABLE_METAL OFF CACHE BOOL "Enable OCCA Metal support")
set(ENABLE_OPENCL OFF CACHE BOOL "Enable OCCA OpenCL support")

find_package(MPI REQUIRED)
if(NOT MPI_FOUND)
  message(FATAL_ERROR "MPI is not found for one of the languages: CXX")
endif()

find_package(Python3 3.8 COMPONENTS Interpreter REQUIRED)
execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install ${_pip_args} --user -r ${CMAKE_CURRENT_SOURCE_DIR}/../kinetix/requirements.txt)

set(KINETIX_INSTALL_PREFIX "")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/occa")

if(NOT DEFINED AMReX_GPU_BACKEND)
  set(AMReX_GPU_BACKEND "NONE" CACHE STRING "AMReX GPU backend" FORCE)
endif()
if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
    enable_language(HIP)
elseif(AMReX_GPU_BACKEND STREQUAL "SYCL")
    enable_language(SYCL)
elseif(AMReX_GPU_BACKEND STREQUAL "NONE")
    message(STATUS "No GPU backend specified for AMReX. Defaulting to CPU.")
else()
    message(WARNING "Unsupported GPU backend specified for AMReX. Defaulting to CPU.")
endif()

set(KINETIX_INSTALL_PREFIX "")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/amrex")

add_library(kinetix SHARED
  src/kinetix.cpp
)
target_link_libraries (kinetix PUBLIC libocca amrex)
set(KINETIX_INSTALL_PREFIX ".")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${CMAKE_INSTALL_PREFIX}/lib")

target_include_directories (kinetix PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

install(TARGETS kinetix DESTINATION ${KINETIX_INSTALL_PREFIX}/lib)
install(DIRECTORY okl DESTINATION ${KINETIX_INSTALL_PREFIX}/.)
install(DIRECTORY ../kinetix DESTINATION ${KINETIX_INSTALL_PREFIX}/.)
install(PROGRAMS ../kinetix/__main__.py DESTINATION ${KINETIX_INSTALL_PREFIX}/kinetix)
install(DIRECTORY ../examples DESTINATION ${KINETIX_INSTALL_PREFIX}/.)
install(FILES ../README.md DESTINATION ${KINETIX_INSTALL_PREFIX})

add_executable(kinetix_bk src/bk.cpp)
target_link_libraries (kinetix_bk PRIVATE kinetix)
install(TARGETS kinetix_bk DESTINATION ${KINETIX_INSTALL_PREFIX}/bin)

target_compile_definitions(kinetix PUBLIC -DSTANDALONE)
install(FILES src/kinetix.hpp DESTINATION ${KINETIX_INSTALL_PREFIX}/include)
