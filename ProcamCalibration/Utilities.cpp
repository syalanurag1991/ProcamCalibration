#include "Utilities.h"

void Utilities::TestOpenCV(bool showImage)
{
	cv::Mat img = cv::imread("lena.png");
	if (showImage) {
		cv::namedWindow("Test Image - Lena.png", cv::WINDOW_NORMAL);
		cv::imshow("Test Image - Lena.png", img);
	}
	std::cout << "\n\n OpenCV is working properly. Press any key to continue ...\n\n";
	cv::waitKey(0);
	cv::destroyWindow("Test Image - Lena.png");
	return;
}

void Utilities::DisplayHomographicallyCalibratedFramesForProjectorAndKinect(KinectWrapper& kinectFrameData, CalibrationHandler& chessboardCalibrationData) {
	std::cout << "\n\n length in meters: " << chessboardCalibrationData.chessboardSquareLengthInMeters << std::endl;

	std::ofstream myfile;
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
		chessboardCalibrationData.homography.at<double>(2, 0) << " " << chessboardCalibrationData.homography.at<double>(2, 1) << " " << chessboardCalibrationData.homography.at<double>(2, 2) << std::endl;

	std::cout << "\n\n Inverted Homography size: " << chessboardCalibrationData.inverseHomography.cols << " x " << chessboardCalibrationData.inverseHomography.rows << "\n\n";
	std::cout <<
		"[" << chessboardCalibrationData.inverseHomography.at<double>(0, 0) << ", " << chessboardCalibrationData.inverseHomography.at<double>(0, 1) << ", " << chessboardCalibrationData.inverseHomography.at<double>(0, 2) << "]\n" <<
		"[" << chessboardCalibrationData.inverseHomography.at<double>(1, 0) << ", " << chessboardCalibrationData.inverseHomography.at<double>(1, 1) << ", " << chessboardCalibrationData.inverseHomography.at<double>(1, 2) << "]\n" <<
		"[" << chessboardCalibrationData.inverseHomography.at<double>(2, 0) << ", " << chessboardCalibrationData.inverseHomography.at<double>(2, 1) << ", " << chessboardCalibrationData.inverseHomography.at<double>(2, 2) << "]\n";

	myfile << "Inverse Homography\n";
	myfile <<
		chessboardCalibrationData.inverseHomography.at<double>(0, 0) << " " << chessboardCalibrationData.inverseHomography.at<double>(0, 1) << " " << chessboardCalibrationData.inverseHomography.at<double>(0, 2) << " " <<
		chessboardCalibrationData.inverseHomography.at<double>(1, 0) << " " << chessboardCalibrationData.inverseHomography.at<double>(1, 1) << " " << chessboardCalibrationData.inverseHomography.at<double>(1, 2) << " " <<
		chessboardCalibrationData.inverseHomography.at<double>(2, 0) << " " << chessboardCalibrationData.inverseHomography.at<double>(2, 1) << " " << chessboardCalibrationData.inverseHomography.at<double>(2, 2) << std::endl;

	std::cout << "\nPress 'ESC' to stop displaying kinect streams and start calibration\n\n";

	myfile.close();

	while (1)
	{
		kinectFrameData.GetLatestColorDataFromKinect() & kinectFrameData.GetLatestDepthDataFromKinect();
		kinectFrameData.UpdateDepthAndCameraSpaceMapping();
		warpPerspective(kinectFrameData.registeredColorFrame, chessboardCalibrationData.projectorFrame, chessboardCalibrationData.homography, chessboardCalibrationData.projectorFrame.size());

		kinectFrameData.DisplayFrames();
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

void Utilities::DisplayStereoCalibratedFramesForProjectorAndKinect(KinectWrapper& kinectFrameData, CalibrationHandler& chessboardCalibrationData)
{
	std::cout << "\nPress 'ESC' to stop displaying streams and exit program \n\n";

	while (1)
	{
		bool updatedFrames = kinectFrameData.GetLatestColorDataFromKinect() & kinectFrameData.GetLatestDepthDataFromKinect();
		kinectFrameData.UpdateDepthAndCameraSpaceMapping();
		warpPerspective(kinectFrameData.registeredColorFrame, chessboardCalibrationData.projectorFrame, chessboardCalibrationData.homography, chessboardCalibrationData.projectorFrame.size());

		undistort(kinectFrameData.registeredColorFrame, kinectFrameData.undistortedRegisteredColorFrame, chessboardCalibrationData.cameraCalibrationMatrix, chessboardCalibrationData.cameraDistortionCoefficients);
		undistort(chessboardCalibrationData.projectorFrame, chessboardCalibrationData.undistortedProjectorFrame, chessboardCalibrationData.projectorCalibrationMatrix, chessboardCalibrationData.projectorDistortionCoefficients);

		remap(kinectFrameData.undistortedRegisteredColorFrame, kinectFrameData.rectifiedRegisteredColorFrame, chessboardCalibrationData.map1x, chessboardCalibrationData.map1y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		//warpPerspective(kinectFrameData.registeredColorFrame, chessboardCalibrationData.projectorFrame, chessboardCalibrationData.homography, chessboardCalibrationData.projectorFrame.size());
		remap(chessboardCalibrationData.undistortedProjectorFrame, chessboardCalibrationData.rectifiedProjectorFrame, chessboardCalibrationData.map2x, chessboardCalibrationData.map2y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

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
