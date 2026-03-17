#ifndef OPENCV_COMPAT_H
#define OPENCV_COMPAT_H

#include <opencv2/opencv.hpp>

// OpenCV 4 removed old C-style constants. Map them to the new cv:: equivalents.
#ifndef CV_RGB2GRAY
#define CV_RGB2GRAY cv::COLOR_RGB2GRAY
#endif
#ifndef CV_BGR2GRAY
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#endif
#ifndef CV_GRAY2BGR
#define CV_GRAY2BGR cv::COLOR_GRAY2BGR
#endif
#ifndef CV_RGBA2GRAY
#define CV_RGBA2GRAY cv::COLOR_RGBA2GRAY
#endif
#ifndef CV_BGRA2GRAY
#define CV_BGRA2GRAY cv::COLOR_BGRA2GRAY
#endif
#ifndef CV_LOAD_IMAGE_UNCHANGED
#define CV_LOAD_IMAGE_UNCHANGED cv::IMREAD_UNCHANGED
#endif
#ifndef CV_REDUCE_SUM
#define CV_REDUCE_SUM cv::REDUCE_SUM
#endif

// OpenCV 4 C API support (CvMat, cvSVD, cvCreateMat, etc.)
#if __has_include(<opencv2/core/core_c.h>)
#include <opencv2/core/core_c.h>
#endif
#if __has_include(<opencv2/core/types_c.h>)
#include <opencv2/core/types_c.h>
#endif

#endif // OPENCV_COMPAT_H
