cmake_minimum_required(VERSION 3.18)
project(nekRK LANGUAGES C CXX)

cmake_policy(SET CMP0079 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX $ENV{HOME}/.local/nekRK CACHE PATH "Default install path" FORCE)
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
execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install ${_pip_args} --user -r ${CMAKE_CURRENT_SOURCE_DIR}/generator/requirements.txt)

set(INSTALL_PREFIX "")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/occa")

set(AMReX_GPU_BACKEND "" CACHE STRING "AMReX GPU backend")
if(AMReX_GPU_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
elseif(AMReX_GPU_BACKEND STREQUAL "HIP")
    enable_language(HIP)
elseif(AMReX_GPU_BACKEND STREQUAL "SYCL")
    enable_language(SYCL)
else()
    message(STATUS "No GPU backend or unsupported backend specified for AMReX. Defaulting to CPU.")
endif()

set(INSTALL_PREFIX "")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/amrex")

add_library(nekRK SHARED
  src/nekRK.cpp
)
target_link_libraries (nekRK PUBLIC libocca)
target_link_libraries (nekRK PUBLIC amrex)
set(INSTALL_PREFIX ".")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${CMAKE_INSTALL_PREFIX}/lib")

target_include_directories (nekRK PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

install(TARGETS nekRK DESTINATION ${INSTALL_PREFIX}/lib)
install(DIRECTORY okl DESTINATION ${INSTALL_PREFIX}/.)
install(DIRECTORY mechanisms DESTINATION ${INSTALL_PREFIX}/.)
install(DIRECTORY generator DESTINATION ${INSTALL_PREFIX}/.)
install(DIRECTORY ciData DESTINATION ${INSTALL_PREFIX}/.) 
install(DIRECTORY tests DESTINATION ${INSTALL_PREFIX}/.)
install(FILES README.md DESTINATION ${INSTALL_PREFIX})

add_executable(nekrk_bk src/bk.cpp)
target_link_libraries (nekrk_bk PRIVATE nekRK)
install(TARGETS nekrk_bk DESTINATION ${INSTALL_PREFIX}/bin)

target_compile_definitions(nekRK PUBLIC -DSTANDALONE)
install(FILES src/nekRK.hpp DESTINATION ${INSTALL_PREFIX}/include)
