cmake_minimum_required(VERSION 3.5)
project(ORB_SLAM3)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
set(Torch_DIR "/home/fatih/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

include(CheckCXXCompilerFlag)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2)

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/CameraModels
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
)

add_compile_options(
    -Wno-unused-but-set-variable
    -Wno-deprecated-declarations
    -Wno-reorder
    -Wno-sign-compare
    -Wno-unused-variable
    -Wno-maybe-uninitialized
)

# SuperPoint modunu aktif etmek için ORBFEATURES makrosunu 0 yapıyoruz
add_definitions(-DUSE_ORBFEATURES=0)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
        src/System.cc
        src/Tracking.cc
        src/LocalMapping.cc
        src/LoopClosing.cc

        src/SPextractor.cc
        src/SPmatcher.cc
	    src/SuperPoint.cc

        src/FrameDrawer.cc
        src/Converter.cc
        src/MapPoint.cc
        src/KeyFrame.cc
        src/Atlas.cc
        src/Map.cc
        src/MapDrawer.cc
        src/Optimizer.cc
        src/PnPsolver.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Initializer.cc
        src/Viewer.cc
        src/ImuTypes.cc
        src/G2oTypes.cc
        src/CameraModels/Pinhole.cpp
        src/CameraModels/KannalaBrandt8.cpp
        src/OptimizableTypes.cpp
        src/MLPnPsolver.cpp
        src/Timestamp.cpp
        src/Random.cpp

        include/Random.h
        include/Timestamp.h
        include/System.h
        include/Tracking.h
        include/LocalMapping.h
        include/LoopClosing.h

        include/SPextractor.h
        include/SPmatcher.h
	    include/Old.h
	    include/SPVocabulary.h
	    include/SuperPoint.h

        include/FrameDrawer.h
        include/Converter.h
        include/MapPoint.h
        include/KeyFrame.h
        include/Atlas.h
        include/Map.h
        include/MapDrawer.h
        include/Optimizer.h
        include/PnPsolver.h
        include/Frame.h
        include/KeyFrameDatabase.h
        include/Sim3Solver.h
        include/Initializer.h
        include/Viewer.h
        include/ImuTypes.h
        include/G2oTypes.h
        include/CameraModels/GeometricCamera.h
        include/CameraModels/Pinhole.h
        include/CameraModels/KannalaBrandt8.h
        include/OptimizableTypes.h
        include/MLPnPsolver.h
        include/TwoViewReconstruction.h
        src/TwoViewReconstruction.cc)

add_subdirectory(Thirdparty/g2o)

target_link_libraries(${PROJECT_NAME}
	${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW3/lib/libDBoW3.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        -lboost_serialization
        -lcrypto
        )

# Build examples

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)


add_executable(mono_webcam
        Examples/Monocular/mono_webcam.cc)
target_link_libraries(mono_webcam ${PROJECT_NAME})
