#include "KinectWrapper.h"
#include "CalibrationHandler.h"
#include "Utilities.h";

int main(int argc, char* argv[]) {

	Utilities procamUtils;
	//procamUtils.TestOpenCV(true);

	// Calibration setup
	std::cout << "\n\nSetup calibration parameters\n\n";
	
	std::cout << "Enter primary screen resolution (e.g. enter 1920 1080 for 1920 x 1080) : ";
	cv::Size primaryScreenResolution;
	std::cin >> primaryScreenResolution.width >> primaryScreenResolution.height;

	std::cout << "Enter projector's aspect ratio (e.g. enter 4 3 for 4:3)                : ";
	cv::Size projectorAspectRatio;
	std::cin >> projectorAspectRatio.width >> projectorAspectRatio.height;

	std::cout << "Enter projector width (e.g. enter 1024 for 1024 x 768 @ 4:3 ratio)     : ";
	cv::Size projectorFrameSize;
	std::cin >> projectorFrameSize.width;
	projectorFrameSize.height = projectorFrameSize.width * projectorAspectRatio.height / projectorAspectRatio.width;

	std::cout << "Enter chessboard size	(e.g. enter 8 5 for an 8 x 5 board)            : ";
	cv::Size chessboardSize;
	std::cin >> chessboardSize.width >> chessboardSize.height;

	std::cout << "Enter minimum chessboard square length in pixels (e.g. 40)             : ";
	UINT minChessboardSquareLengthInPixels;
	std::cin >> minChessboardSquareLengthInPixels;

	// Initialize Kinect
	KinectWrapper kinectFrameData = KinectWrapper();
	if (!kinectFrameData.Initialize()) return 1;
	kinectFrameData.CreateWindowsForDisplayingFrames(false, false);

	// Initialize calibration object
	CalibrationHandler chessboardCalibrationData = CalibrationHandler(kinectFrameData.registeredColorFrameSize, primaryScreenResolution, projectorFrameSize, projectorAspectRatio, chessboardSize, minChessboardSquareLengthInPixels, false, false, false, true, true);
	chessboardCalibrationData.CreateWindowsForDisplayingFrames(false, false);
	chessboardCalibrationData.CreateChessBoardPatternImages();
	chessboardCalibrationData.CollectCalibrationPatternPointsFromProjector(kinectFrameData);
	chessboardCalibrationData.CreateWorldCoordinatesForChessBoardCornerPositions();

	// Display calibrated streams - Homography
	procamUtils.DisplayHomographicallyCalibratedFramesForProjectorAndKinect(kinectFrameData, chessboardCalibrationData);

	// Camera calibration
	//chessboardCalibrationData.StartCalibration(true);

	// Projector calibration
	//chessboardCalibrationData.StartCalibration(false);

	// Procam calibration
	//chessboardCalibrationData.StartProcamCalibration();
	
	// Display calibrated streams - stereo calibration
	//DisplayStereoCalibratedFramesForProjectorAndKinect(kinectFrameData, chessboardCalibrationData);

	kinectFrameData.ShutDown();
	std::cout << "\nCalibration complete!!";
	//std::cout << "\nPress any key to exit...";
	//cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}