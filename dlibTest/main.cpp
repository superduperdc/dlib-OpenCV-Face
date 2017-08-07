/*  Camera input, dlib facial landmark detection, 1st attempt after OpenCV and dlib builds...*/
#include <dlib/opencv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
//#include <dlib/image_processing/render_eye_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <typeinfo>

#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/opencv.hpp>

#include "boost/filesystem.hpp"

#include <vector>
#include <opencv2/opencv.hpp>
#include "render_face.hpp"

#include <stdio.h>

using namespace dlib;
using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

#define FACE_DOWNSAMPLE_RATIO 4
#define SKIP_FRAMES 2
#define OPENCV_FACE_RENDER

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
//CascadeClassifier face_cascade;
//CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";
String camera_input_window = "Camera capture";

bool wroteAnImage = false;
bool processSingleFace = true;

// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
void get_all(const fs::path& root, const string& ext, std::vector<fs::path>& ret)
{
	if (!fs::exists(root) || !fs::is_directory(root)) return;

	fs::recursive_directory_iterator it(root);
	fs::recursive_directory_iterator endit;


	std::cout << "looking in path " << root.string() << '\n';

	while (it != endit)
	{
		if (fs::is_regular_file(*it) && (it->path().extension() == ".jpg" ||
										it->path().extension() == ".pgm" ||
										it->path().extension() == ".png" ||
										it->path().extension() == ".tif" ||
										it->path().extension() == ".tiff" )) ret.push_back(it->path());
		++it;

	}

}


std::vector<cv::Point3d> get_3d_model_points()
{
	std::vector<cv::Point3d> modelPoints;

	modelPoints.push_back(cv::Point3d(0.0f, 0.0f, 0.0f)); //The first must be (0,0,0) while using POSIT
	modelPoints.push_back(cv::Point3d(0.0f, -330.0f, -65.0f));
	modelPoints.push_back(cv::Point3d(-225.0f, 170.0f, -135.0f));
	modelPoints.push_back(cv::Point3d(225.0f, 170.0f, -135.0f));
	modelPoints.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f));
	modelPoints.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));

	return modelPoints;

}


std::vector<cv::Point2d> get_2d_image_points(full_object_detection &d)
{
	std::vector<cv::Point2d> image_points;
	image_points.push_back(cv::Point2d(d.part(30).x(), d.part(30).y()));    // Nose tip
	image_points.push_back(cv::Point2d(d.part(8).x(), d.part(8).y()));      // Chin
	image_points.push_back(cv::Point2d(d.part(36).x(), d.part(36).y()));    // Left eye left corner
	image_points.push_back(cv::Point2d(d.part(45).x(), d.part(45).y()));    // Right eye right corner
	image_points.push_back(cv::Point2d(d.part(48).x(), d.part(48).y()));    // Left Mouth corner
	image_points.push_back(cv::Point2d(d.part(54).x(), d.part(54).y()));    // Right mouth corner
	return image_points;

}

cv::Mat get_camera_matrix(float focal_length, cv::Point2d center)
{
	cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
	return camera_matrix;
}





int main(int argc, char** argv)
{
	try
	{
		std::vector<fs::path> v;
		fs::path p(argv[1]);
		get_all(p, ".pgm", v);
	
		//fs::recursive_directory_iterator begin(p), end;
		//std::vector<fs::directory_entry> v(begin, end);
		std::cout << "There are " << v.size() << " files: \n";

		image_window win_faces;
		image_window win;
		Mat frame;
		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
		int total_faces = 0;
		int skipped_faces = 0;

		std::ofstream myfile;
		myfile.open("data.txt");

		for (auto& f : v)
		{
			std::cout << f << '\n';
			cout << "processing image " << f << endl;

			// Read image using OpenCV
			cv::Mat im_bgr = cv::imread(f.string(), cv::IMREAD_COLOR);
			// Make the image larger so we can detect small faces.
			cv::resize(im_bgr, im_bgr, cv::Size(), 2.0, 2.0);

			cv::Size size = im_bgr.size();

			// Convert OpenCV's Mat to Dlib's cv_image
			cv_image<bgr_pixel> img(im_bgr);
			//cv_image<bgr_pixel> img(frame);

			std::vector<dlib::rectangle> faces = detector(img);
			//cout << "Number of faces detected: " << dets.size() << endl;
			if (faces.size() == 0)
			{
				skipped_faces++;
			}


			if (faces.size() >> 0)
			{

				//cout << "face box 1: " << dets[0] << endl;

				// Pose estimation
				std::vector<cv::Point3d> model_points = get_3d_model_points();

				std::vector<full_object_detection> shapes;
				for (unsigned long j = 0; j < faces.size(); ++j)
				{

					full_object_detection shape = pose_model(img, faces[j]);
					//cout << "number of parts: " << shape.num_parts() << endl;
					//cout << "pixel position of first part:  " << shape.part(0) << endl;
					//cout << "pixel position of first part: " << shape.part(0).x() << endl;
					cout << "bounding box for face: " << faces[0].left() << ", " << faces[0].top() << ", " << (faces[0].right() - faces[0].left()) << ", " << (faces[0].bottom() - faces[0].top()) << " " ;
					cout << "Faces found, skipped, total: " << total_faces++ << " " << skipped_faces << " " << v.size() <<endl;

					//Write shape coordinates to txt file
					//std::ofstream myfile;
					//myfile.open("frame.txt");
					//output bounding box for face
					//myfile << dets[0].left() << ", " << dets[0].top() << endl;
					//myfile << (dets[0].right() - dets[0].left()) << ", " << (dets[0].bottom() - dets[0].top()) << endl;
					////then left eye limits followed by right eye limits
					//myfile << shape.part(36).x() << ", " << shape.part(36).y() << endl;
					//myfile << shape.part(39).x() << ", " << shape.part(39).y() << endl;
					//myfile << shape.part(42).x() << ", " << shape.part(42).y() << endl;
					//myfile << shape.part(45).x() << ", " << shape.part(45).y() << endl;




					shapes.push_back(shape);  //This adds shape, which is the list of shapes of this face, to shapes.

					//render_face(im_bgr, shape);
					std::vector<cv::Point2d> image_points = get_2d_image_points(shape);
					double focal_length = im_bgr.cols;
					cv::Mat camera_matrix = get_camera_matrix(focal_length, cv::Point2d(im_bgr.cols / 2, im_bgr.rows / 2));
					cv::Mat rotation_vector;
					cv::Mat rotation_matrix;
					cv::Mat translation_vector;


					cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

					cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

					//cv::Rodrigues(rotation_vector, rotation_matrix);

					std::vector<cv::Point3d> nose_end_point3D;
					std::vector<cv::Point2d> nose_end_point2D;
					nose_end_point3D.push_back(cv::Point3d(0, 0, 1000.0));

					cv::projectPoints(nose_end_point3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs, nose_end_point2D);
					//                cv::Point2d projected_point = find_projected_point(rotation_matrix, translation_vector, camera_matrix, cv::Point3d(0,0,1000.0));
					cv::line(im_bgr, image_points[0], nose_end_point2D[0], cv::Scalar(255, 0, 0), 2);
					//                cv::line(im,image_points[0], projected_point, cv::Scalar(0,0,255), 2);

					//cv::putText(im_bgr, cv::format("nose %.3f %.3f", nose_end_point2D[0].x, nose_end_point2D[0].y), cv::Point(30, size.height - 30), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);

					//output file format:  fn, bbox, eye corners 
					//these XY coordinates are adjusted for face bounding box later
					int bboxWidth = faces[0].right() - faces[0].left();
					int bboxHeight = faces[0].bottom() - faces[0].top();
					myfile << f.string() << "," << faces[0].left() << "," << faces[0].top() << "," << bboxWidth << "," << bboxHeight << ","
						<< shape.part(36).x() << "," << shape.part(36).y() << ","
						<< shape.part(39).x() << "," << shape.part(39).y() << ","
						<< shape.part(42).x() << "," << shape.part(42).y() << ","
						<< shape.part(45).x() << "," << shape.part(45).y() << ","
						<< shape.part(30).x() << "," << shape.part(30).y() << ","					//nose tip
						<< nose_end_point2D[0].x << "," << nose_end_point2D[0].y << endl;           //projected nose line end point

					//cout << "nose end point x: " << nose_end_point2D[0].x - shape.part(30).x() << " y: " << nose_end_point2D[0].y - shape.part(30).y() << endl;
					//if (abs((nose_end_point2D[0].x - shape.part(30).x()) / (faces[0].right() - faces[0].left())) <= 0.8 &&
					//	abs((nose_end_point2D[0].y - shape.part(30).y()) / (faces[0].bottom() - faces[0].top())) <= 0.8)
					//{
					//	cout << "AngleX = good" << endl;
					//}
					//else
					//{
					//
					//	cout << "AngleX = bad" << endl;
					//}
				
					
					if (processSingleFace)
						break;
				}



				

				//frame = toMat(img);
				////save_png(img, "detected.jpg");
				//try {
				//	imwrite("frame.jpg", frame);
				//}
				//catch (runtime_error& ex) {
				//	fprintf(stderr, "Exception converting image to jpg format: %s\n", ex.what());
				//	return 1;
				//}

				//fprintf(stdout, "Saved jpg file with alpha data.\n");

				///*-----------------------------
				win.clear_overlay();
				win.set_image(img);
				win.add_overlay(render_face_detections(shapes));
				//-----------------------------*/


				dlib::array<array2d<rgb_pixel> > face_chips;

				extract_image_chips(img, get_face_chip_details(shapes), face_chips);

				win_faces.set_image(tile_images(face_chips));


				//save_png(tile_images(face_chips), "detected.jpg");

				cout << "Hit enter to process the next image..." << endl;
				cin.get();

			}
		}

		myfile.close();
		




		/*
		if (argc == 1)
		{
			cout << "Call this program like this:" << endl;
			cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
			cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
			cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl; return 0;
		} 
		frontal_face_detector detector = get_frontal_face_detector(); 
		shape_predictor sp; 
		deserialize(argv[1]) >> sp;

		image_window win, win_faces;

		//VideoCapture capture;

		Mat frame;
		*/
		//-- 1. Load the cascades
		//if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };
		//if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading eyes cascade\n"); return -1; };

		//-- 2. Read the video stream
		//capture.open(0);  //Modified to force 
		//if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }

		//capture.set(CAP_PROP_EXPOSURE, 1.0);
		//capture.set(CV_CAP_PROP_GAIN, 0.9);

//		while (capture.read(frame) ) //&& !wroteAnImage)
//		{

		//try
		//{

		//if (argc < 2) {
		//	std::cout << "Usage: " << argv[0] << " path\n";
		//	return 1;
		//}
		//fs::path p(argv[1]);
		//if (!exists(p) || !is_directory(p)) {
		//	std::cout << p << " is not a path\n";
		//	return 1;
		//}
		//fs::recursive_directory_iterator begin(p), end;
		//std::vector<fs::directory_entry> v(begin, end);
		//std::cout << "There are " << v.size() << " files: \n";
		//for (auto& f : v)
		//	std::cout << f << '\n';
		//}
		//catch (std::exception& e) {
		//	std::cout << e.what()  << "  boost exception"  << '\n';
		//}

		//return 0;


		/*
		for (int i = 2; i < argc; ++i)
		{
			cout << "processing image " << argv[i] <<  " filesize " << fs::file_size(argv[i]) << endl;
		


			// Read image using OpenCV
			cv::Mat im_bgr = cv::imread(argv[i], cv::IMREAD_COLOR);
			// Make the image larger so we can detect small faces.
			cv::resize(im_bgr, im_bgr, cv::Size(), 2.0, 2.0);
						
			// Convert OpenCV's Mat to Dlib's cv_image
			cv_image<bgr_pixel> img(im_bgr);
			//cv_image<bgr_pixel> img(frame);

			std::vector<dlib::rectangle> dets = detector(img);
			cout << "Number of faces detected: " << dets.size() << endl;

			if (dets.size() >> 0)
			{

				//cout << "face box 1: " << dets[0] << endl;

				std::vector<full_object_detection> shapes;
				for (unsigned long j = 0; j < dets.size(); ++j)
				{
					full_object_detection shape = sp(img, dets[j]);
					cout << "number of parts: " << shape.num_parts() << endl;
					cout << "pixel position of first part:  " << shape.part(0) << endl;
					cout << "pixel position of first part: " << shape.part(0).x() << endl; 
					cout << "bounding box for face: " << dets[0].left() << ", " << dets[0].top() << ", " << (dets[0].right()- dets[0].left()) << ", " << (dets[0].bottom() - dets[0].top()) << " " << endl;

					//Write shape coordinates to txt file
					std::ofstream myfile;
					myfile.open("frame.txt");
					//output bounding box for face
					myfile << dets[0].left() << ", " << dets[0].top() << endl;
					myfile << (dets[0].right() - dets[0].left()) << ", " << (dets[0].bottom() - dets[0].top()) << endl;
					//then left eye limits followed by right eye limits
					myfile << shape.part(36).x() << ", " << shape.part(36).y() << endl;
					myfile << shape.part(39).x() << ", " << shape.part(39).y() << endl;
					myfile << shape.part(42).x() << ", " << shape.part(42).y() << endl;
					myfile << shape.part(45).x() << ", " << shape.part(45).y() << endl;

					for (int i = 0; i < shape.num_parts(); i++)
					{
						myfile << shape.part(i).x() << ", " << shape.part(i).y() << endl;
					}
					myfile.close();

					shapes.push_back(shape);  //This adds shape, which is the list of shapes of this face, to shapes.
				}

				//if (!wroteAnImage)
				//{

				//	try {
				//		imwrite("frame.png", frame);
				//	}
				//	catch (runtime_error& ex) {
				//		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
				//		return 1;
				//	}

				//	fprintf(stdout, "Saved PNG file with alpha data.\n");
				//	wroteAnImage = true;





				//}



				win.clear_overlay();
				win.set_image(img);
				frame = toMat(img);
				//save_png(img, "detected.jpg");
				try {
					imwrite("frame.jpg", frame);
				}
				catch (runtime_error& ex) {
					fprintf(stderr, "Exception converting image to jpg format: %s\n", ex.what());
					return 1;
				}

				fprintf(stdout, "Saved jpg file with alpha data.\n");

				win.add_overlay(render_face_detections(shapes));

				dlib::array<array2d<rgb_pixel> > face_chips;
				
				extract_image_chips(img, get_face_chip_details(shapes), face_chips);
				
				//play with face_chips data


				//cout << "chip data " << face_chips[0].size << endl;






				win_faces.set_image(tile_images(face_chips));
				

				//save_png(tile_images(face_chips), "detected.jpg");

				cout << "Hit enter to process the next image..." << endl;
				cin.get();
			}
		} */
		
	}
	catch (exception& e)
	{
		cout << "\nexception thrown! in faces " << endl;
		cout << e.what() << endl;
		
	}

	return 0;
}