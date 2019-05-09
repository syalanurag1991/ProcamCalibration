#include <Windows.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <Kinect.h>

#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>

#include <fstream>

#include "KinectWrapper.h"
#include "CalibrationHandler.h"

using namespace cv;
using namespace std;

#define resolution_first_width 1920
#define resolution_first_height 1080

#define bytesPerPixel 4
#define downsampleSize 2

#define colorWidth 1920
#define colorHeight 1080
#define colorFrameArea colorWidth * colorHeight
#define colorDataLength colorFrameArea * bytesPerPixel

#define depthWidth 512
#define depthHeight 424
#define depthDataLength depthWidth * depthHeight
#define registeredColorDataLength depthDataLength * bytesPerPixel

// keep delay at 100 ms minimum for best aquisition of points
#define delay 100

void TestOpenCV(bool showImage)
{
	Mat img = imread("lena.png");
	if (showImage) {
		namedWindow("Test Image - Lena.png", WINDOW_NORMAL);
		imshow("Test Image - Lena.png", img);
		cv::waitKey(0);
	}
	std::cout << "\n\n OpenCV is working properly. \n\n";
	return;
}

void DisplayHomographicallyCalibratedFramesForProjectorAndKinect(KinectWrapper& kinectFrameData, CalibrationHandler& chessboardCalibrationData) {
	std::cout << "\n\n length in meters: " << chessboardCalibrationData.chessboardSquareLengthInMeters << endl;

	ofstream myfile;
	myfile.open("calibration.txt");

	std::cout << "\n\n Homography size: " << chessboardCalibrationData.homography.cols << " x " << chessboardCalibrationData.homography.rows << "\n\n";
	std::cout <<
		"[" << chessboardCalibrationData.homography.at<double>(0, 0) << ", " << chessboardCalibrationData.homography.at<double>(0, 1) << ", " << chessboardCalibrationData.homography.at<double>(0, 2) << "]\n" <<
		"[" << chessboardCalibrationData.homography.at<double>(1, 0) << ", " << chessboardCalibrationData.homography.at<double>(1, 1) << ", " << chessboardCalibrationData.homography.at<double>(1, 2) << "]\n" <<
		"[" << chessboardCalibrationData.homography.at<double>(2, 0) << ", " << chessboardCalibrationData.homography.at<double>(2, 1) << ", " << chessboardCalibrationData.homography.at<double>(2, 2) << "]\n";

	myfile << "Homography\n";
	myfile <<
		chessboardCalibrationData.homography.at<double>(0, 0) << " " << chessboardCalibrationData.homography.at<double>(0, 1) << " " << chessboardCalibrationData.homography.at<double>(0, 2) << " " <<
		chessboardCalibrationData.homography.at<double>(1, 0) << " " << chessboardCalibrationData.homography.at<double>(1, 1) << " " << chessboardCalibrationData.homography.at<double>(1, 2) << " " <<
		chessboardCalibrationData.homography.at<double>(2, 0) << " " << chessboardCalibrationData.homography.at<double>(2, 1) << " " << chessboardCalibrationData.homography.at<double>(2, 2) << endl;

	std::cout << "\n\n Inverted Homography size: " << chessboardCalibrationData.inverseHomography.cols << " x " << chessboardCalibrationData.inverseHomography.rows << "\n\n";
	std::cout <<
		"[" << chessboardCalibrationData.inverseHomography.at<double>(0, 0) << ", " << chessboardCalibrationData.inverseHomography.at<double>(0, 1) << ", " << chessboardCalibrationData.inverseHomography.at<double>(0, 2) << "]\n" <<
		"[" << chessboardCalibrationData.inverseHomography.at<double>(1, 0) << ", " << chessboardCalibrationData.inverseHomography.at<double>(1, 1) << ", " << chessboardCalibrationData.inverseHomography.at<double>(1, 2) << "]\n" <<
		"[" << chessboardCalibrationData.inverseHomography.at<double>(2, 0) << ", " << chessboardCalibrationData.inverseHomography.at<double>(2, 1) << ", " << chessboardCalibrationData.inverseHomography.at<double>(2, 2) << "]\n";

	myfile << "Inverse Homography\n";
	myfile <<
		chessboardCalibrationData.inverseHomography.at<double>(0, 0) << " " << chessboardCalibrationData.inverseHomography.at<double>(0, 1) << " " << chessboardCalibrationData.inverseHomography.at<double>(0, 2) << " " <<
		chessboardCalibrationData.inverseHomography.at<double>(1, 0) << " " << chessboardCalibrationData.inverseHomography.at<double>(1, 1) << " " << chessboardCalibrationData.inverseHomography.at<double>(1, 2) << " " <<
		chessboardCalibrationData.inverseHomography.at<double>(2, 0) << " " << chessboardCalibrationData.inverseHomography.at<double>(2, 1) << " " << chessboardCalibrationData.inverseHomography.at<double>(2, 2) << endl;

	cout << "\nPress 'ESC' to stop displaying kinect streams and start calibration\n\n";

	myfile.close();

	while (1)
	{
		kinectFrameData.GetLatestColorDataFromKinect() & kinectFrameData.GetLatestDepthDataFromKinect();
		kinectFrameData.UpdateDepthAndCameraSpaceMapping();
		warpPerspective(kinectFrameData.registeredColorFrame, chessboardCalibrationData.projectorFrame, chessboardCalibrationData.homography, chessboardCalibrationData.projectorFrame.size());

		kinectFrameData.DisplayFrames(true, true, true, false, false);
		chessboardCalibrationData.DisplayFrames(true, false, false);

		char key = cv::waitKey(30);
		if (key == 27)
		{
			std::cout << "\nExiting streams...\n\n";
			break;
		}
	}

	return;
}

void DisplayStereoCalibratedFramesForProjectorAndKinect(KinectWrapper& kinectFrameData, CalibrationHandler& chessboardCalibrationData)
{
	cout << "\nPress 'ESC' to stop displaying streams and exit program \n\n";

	while (1)
	{
		bool updatedFrames = kinectFrameData.GetLatestColorDataFromKinect() & kinectFrameData.GetLatestDepthDataFromKinect();
		kinectFrameData.UpdateDepthAndCameraSpaceMapping();
		warpPerspective(kinectFrameData.registeredColorFrame, chessboardCalibrationData.projectorFrame, chessboardCalibrationData.homography, chessboardCalibrationData.projectorFrame.size());

		undistort(kinectFrameData.registeredColorFrame, kinectFrameData.undistortedRegisteredColorFrame, chessboardCalibrationData.cameraCalibrationMatrix, chessboardCalibrationData.cameraDistortionCoefficients);
		undistort(chessboardCalibrationData.projectorFrame, chessboardCalibrationData.undistortedProjectorFrame, chessboardCalibrationData.projectorCalibrationMatrix, chessboardCalibrationData.projectorDistortionCoefficients);

		remap(kinectFrameData.undistortedRegisteredColorFrame, kinectFrameData.rectifiedRegisteredColorFrame, chessboardCalibrationData.map1x, chessboardCalibrationData.map1y, INTER_LINEAR, BORDER_CONSTANT);
		//warpPerspective(kinectFrameData.registeredColorFrame, chessboardCalibrationData.projectorFrame, chessboardCalibrationData.homography, chessboardCalibrationData.projectorFrame.size());
		remap(chessboardCalibrationData.undistortedProjectorFrame, chessboardCalibrationData.rectifiedProjectorFrame, chessboardCalibrationData.map2x, chessboardCalibrationData.map2y, INTER_LINEAR, BORDER_CONSTANT);

		kinectFrameData.DisplayFrames();
		chessboardCalibrationData.DisplayFrames();

		char key = cv::waitKey(30);
		if (key == 27)
		{
			std::cout << "\nExiting streams...\n\n";
			break;
		}
	}

	return;
}

int main(int argc, char* argv[]) {

	// Camera setup
	Size cameraFrameSize = Size(512, 424);

	// Projector setup
	UINT firstScreenResolution = 1920;
	Size chessboardSize = Size(8, 5);
	UINT minChessboardSquareLengthInPixels = 40;
	Size primaryScreenResolution = Size(1920, 1080);
	Size projectorFrameSize = Size(1024, 768);
	Size ProjectorAspectRatio = Size(4, 3);
	
	//CalibrationPatternData chessboardCalibrationData = CalibrationPatternData(cameraFrameSize, primaryScreenResolution, projectorFrameSize, ProjectorAspectRatio, chessboardSize, minChessboardSquareDimension, false, false, false, true, true);
	
	CalibrationHandler chessboardCalibrationData = CalibrationHandler(cameraFrameSize, primaryScreenResolution, projectorFrameSize, ProjectorAspectRatio, chessboardSize, minChessboardSquareLengthInPixels, false, false, false, true, true);
	

	std::cout << "Chessboard size = " << chessboardCalibrationData.calibrationPatternSize.width << " x " << chessboardCalibrationData.calibrationPatternSize.height << endl;
	std::cout << "Chessboard square length (pixels) = " << chessboardCalibrationData.chessboardSquareLengthInPixels << endl;

	// Initialize Kinect
	KinectWrapper kinectFrameData = KinectWrapper(colorWidth, colorHeight, depthWidth, depthHeight, bytesPerPixel, downsampleSize);
	if (!kinectFrameData.Initialize()) return 1;
	TestOpenCV(false);

	kinectFrameData.CreateWindowsForDisplayingFrames();
	chessboardCalibrationData.CreateWindowsForDisplayingFrames();
	chessboardCalibrationData.CreateChessBoardPatternImages();
	chessboardCalibrationData.CollectCalibrationPatternPointsFromProjector(kinectFrameData);
	chessboardCalibrationData.CreateWorldCoordinatesForChessBoardCornerPositions();

	// Diaplay calibrated streams - Homography
	DisplayHomographicallyCalibratedFramesForProjectorAndKinect(kinectFrameData, chessboardCalibrationData);

	// Camera calibration
	chessboardCalibrationData.StartCalibration(true);

	// Projector calibration
	chessboardCalibrationData.StartCalibration(false);

	// Procam calibration
	//chessboardCalibrationData.StartProcamCalibration();
	
	// Display calibrated streams - stereo calibration
	//DisplayStereoCalibratedFramesForProjectorAndKinect(kinectFrameData, chessboardCalibrationData);

	kinectFrameData.ShutDown();
	std::cout << "\nPress any key to exit...";
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}