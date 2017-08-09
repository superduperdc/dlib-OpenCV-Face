#include "img_pose.h"

#include <iostream>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;

#define A_BIT (1 << 0)
#define B_BIT (1 << 1)
#define C_BIT (1 << 2)
#define D_BIT (1 << 3)

//extern full_object_detection shape;

//definition


int evaluatePOSE(const full_object_detection& shape, const dlib::rectangle face,
	double& LR, double& EyeLevel, double& Nod)
{
	double faceLeft;  //distance from left face to nose bridge
	double faceRight;
	double faceNose;  //distance from top nose to bottom nose
	double faceChin;  //distance from chin to bottom nose
	double nSqr = 2.0;
	double nSqRoot = 0.5;
	double eyeDiff;
	double bboxHeight;
	int facePOSEcode = 0;


	cv::Scalar color = (0, 255, 0);

	double LRRatio = 0.4;  //variance limit from facing camera
	double TiltRatio = 0.1;
	

	faceRight = std::pow((std::pow((shape.part(16).x() - shape.part(34).x()), nSqr) + std::pow((shape.part(16).y() - shape.part(34).y()), nSqr)), nSqRoot);
	faceLeft  = std::pow((std::pow((shape.part(0).x()  - shape.part(32).x()), nSqr) + std::pow((shape.part(0).y()  - shape.part(32).y()), nSqr)), nSqRoot);

	LR = faceLeft / faceRight;
	if (LR >= (1 + LRRatio) || LR <= (1 - LRRatio))
	{
		
		facePOSEcode += 1;
	}

	eyeDiff = std::abs(shape.part(36).y() - shape.part(45).y());
	bboxHeight = face.height();

	EyeLevel = eyeDiff / bboxHeight;
	if (EyeLevel >= TiltRatio)
	{ 
		
		facePOSEcode += 2;
	}

	faceNose = std::pow((std::pow((shape.part(27).x() - shape.part(33).x()), nSqr) + std::pow((shape.part(27).y() - shape.part(33).y()), nSqr)), nSqRoot);
	faceChin = std::pow((std::pow((shape.part(9).x() - shape.part(33).x()), nSqr) + std::pow((shape.part(9).y() - shape.part(33).y()), nSqr)), nSqRoot);

	Nod = faceNose / faceChin;
	if (Nod >= 1.1 || Nod <= 0.55)
	{
		
		facePOSEcode += 4;
	}

	if (!facePOSEcode)
	{
		cout << "Good" << endl;
	}
	if (facePOSEcode & A_BIT)
	{
		cout << "Left Right " << LR << endl;
	}
	if (facePOSEcode & B_BIT)
	{
		cout << "Eye Level  " << EyeLevel << endl;
	}
	if (facePOSEcode & C_BIT)
	{
		cout << "Nod        " << Nod << endl;
	}

	//cout << "From computePOSE LR " << faceLeft / faceRight << " TiltRatio " << eyeDiff / bboxHeight<< endl;

	//cout << "From computePOSE 16.x 34.x 16.y 34.y " << shape.part(16).x() << " "
	//	<< shape.part(34).x() << " "
	//	<< shape.part(16).y() << " "
	//	<< shape.part(34).y() << " "
	//	<< endl;

	return facePOSEcode;
}
