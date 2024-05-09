# CMakeLists.txt works for Clion.sh
cmake_minimum_required(VERSION 3.0)
project(SimpleMapping)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release) #Debug
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_PREFIX_PATH ~/libtorch)
set(CUDNN_LIBRARY /storage/software/cuda/cuda-11.1/cudnn/v8.2.1.32/cuda/lib64)
set(CUDNN_LIBRARY_PATH /storage/software/cuda/cuda-11.1/cudnn/v8.2.1.32/cuda/lib64)
set(CUDNN_INCLUDE_PATH /storage/software/cuda/cuda-11.1/cudnn/v8.2.1.32/cuda/include) 

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O3    -fPIC")#-Wall
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3     -fPIC")#-Wall
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4.2 REQUIRED HINTS /usr/include/)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 4.2 not found.")
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.3.9 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Torch REQUIRED)
find_package(Boost COMPONENTS system thread)

add_subdirectory(libdr/dr_mvsnet)
add_subdirectory(libdr/dr_fusion)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/MvsBackend.cpp
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
include/utils/MinimalImage.h
include/System.h
include/MvsBackend.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/ImuTypes.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Settings.h)

add_subdirectory(Thirdparty/g2o)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
-lboost_serialization
-lboost_system
-lboost_filesystem
-lcrypto
)

target_link_libraries(${PROJECT_NAME} dr-mvsnet) #new: mvsnet
target_link_libraries(${PROJECT_NAME} dr-fusion) #new: fusion
target_compile_definitions(${PROJECT_NAME} PUBLIC MVSNET) #new: mvsnet

# Build examples
# Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc
        Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME} ${OpenCV_LIBS})
if (MVSNET)
    target_link_libraries(mono_inertial_euroc dr-mvsnet)
    target_link_libraries(mono_inertial_euroc dr-fusion)
    target_compile_definitions(mono_inertial_euroc PUBLIC MVSNET)
endif ()

add_executable(mono_inertial_own
        Examples/Monocular-Inertial/mono_inertial_own.cc)
target_link_libraries(mono_inertial_own ${PROJECT_NAME} ${OpenCV_LIBS})
if (MVSNET)
    target_link_libraries(mono_inertial_own dr-mvsnet)
    target_link_libraries(mono_inertial_own dr-fusion)
    target_compile_definitions(mono_inertial_own PUBLIC MVSNET)
endif ()

