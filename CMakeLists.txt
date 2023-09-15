cmake_minimum_required(VERSION 3.18)
project(nekCRF LANGUAGES C CXX)

cmake_policy(SET CMP0079 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
  set(STANDALONE OFF)
else()
  set(STANDALONE ON)
endif()

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND ${STANDALONE})
  SET(CMAKE_INSTALL_PREFIX $ENV{HOME}/.local/nekCRF CACHE PATH "Default install path" FORCE)
ENDIF()


if(${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
  MESSAGE(FATAL_ERROR "In-place builds are not supported. Please create a separate build directory")
endif(${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

if(DEFINED OCCA_FOUND OR ${STANDALONE})
  set(ENABLE_METAL OFF CACHE BOOL "Enable OCCA Metal support")
  set(ENABLE_OPENCL OFF CACHE BOOL "Enable OCCA OpenCL support")
endif()

if(NEKRS_USE_DFLOAT_FLOAT)
  message(FATAL_ERROR "NEKRS_USE_DFLOAT_FLOAT=ON not supported!")
endif()

find_package(MPI REQUIRED)
if(NOT MPI_FOUND)
  message(FATAL_ERROR "MPI is not found for one of the languages: CXX")
endif()

find_package(Python3 3.8 COMPONENTS Interpreter REQUIRED)
execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install ${_pip_args} --user -r ${CMAKE_CURRENT_SOURCE_DIR}/generator/requirements.txt)

set(INSTALL_PREFIX "")
if(STANDALONE)
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rd_party/occa")

  add_library(nekCRF SHARED
    src/_nekCRF.cpp
  )
  target_link_libraries (nekCRF PUBLIC libocca)
  set(INSTALL_PREFIX ".")
else()
  add_library(nekCRF SHARED
    src/_nekCRF.cpp
    src/nekCRF.cpp
  )
  target_link_libraries (nekCRF PRIVATE nekrs-lib)
  target_include_directories (nekCRF PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_libraries (nekCRF PUBLIC libocca)
  set(INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/3rd_party/nekCRF")
endif()

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
if(STANDALONE)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${CMAKE_INSTALL_PREFIX}/lib")
else()
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${INSTALL_PREFIX}/lib")
endif()

target_include_directories (nekCRF PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

install(TARGETS nekCRF DESTINATION ${INSTALL_PREFIX}/lib)
install(DIRECTORY okl DESTINATION ${INSTALL_PREFIX}/.)
install(DIRECTORY mechanisms DESTINATION ${INSTALL_PREFIX}/.)
install(PROGRAMS generator/__init__.py DESTINATION ${INSTALL_PREFIX}/generator)
install(PROGRAMS generator/generate.py DESTINATION ${INSTALL_PREFIX}/generator)
install(PROGRAMS generator/constants.py DESTINATION ${INSTALL_PREFIX}/generator)
install(PROGRAMS generator/utils.py DESTINATION ${INSTALL_PREFIX}/generator)
install(DIRECTORY ciData DESTINATION ${INSTALL_PREFIX}/.) 
install(FILES README.md DESTINATION ${INSTALL_PREFIX})

add_executable(nekcrf_bk src/bk.cpp)
target_link_libraries (nekcrf_bk PRIVATE nekCRF)
install(TARGETS nekcrf_bk DESTINATION ${INSTALL_PREFIX}/bin)

if(STANDALONE)
  target_compile_definitions(nekCRF PUBLIC -DSTANDALONE)
  install(FILES include/_nekCRF.hpp DESTINATION ${INSTALL_PREFIX}/include)
else()
    install(DIRECTORY examples DESTINATION ${INSTALL_PREFIX})
    install(FILES include/nekCRF.hpp DESTINATION ${INSTALL_PREFIX}/include)
    set(ENABLE_CVODE ON PARENT_SCOPE) # compile CVODE with nekRS
endif()
