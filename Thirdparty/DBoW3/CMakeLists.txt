cmake_minimum_required(VERSION 2.8)
project(DBoW3)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall  -O3 -march=native")

set(HDRS_DBOW3
        src/BowVector.h
        src/Database.h
        src/DBoW3.h
        src/DescManip.h
        src/exports.h
        src/QueryResults.h
        src/quicklz.h
        src/timers.h
        src/FeatureVector.h
        src/ScoringObject.h
        src/Vocabulary.h)
set(SRCS_DBOW3
        src/BowVector.cpp
        src/Database.cpp
        src/DescManip.cpp
        src/QueryResults.cpp
        src/quicklz.c
        src/FeatureVector.cpp
        src/ScoringObject.cpp
        src/Vocabulary.cpp)




set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(/usr/include/opencv4 /usr/include/opencv4/opencv)
add_library(DBoW3 SHARED ${SRCS_DBOW3})
target_link_libraries(DBoW3 ${OpenCV_LIBS})

