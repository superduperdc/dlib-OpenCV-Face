#pragma once

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
//declaration
int evaluatePOSE(const full_object_detection& shape, const dlib::rectangle face, double& LR, double& EyeLevel, double& Nod);