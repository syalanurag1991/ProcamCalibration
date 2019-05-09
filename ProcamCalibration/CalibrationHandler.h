#pragma once
#include <Windows.h>
#include <iostream>
#include <iomanip>
#include <exception>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "KinectWrapper.h"

class CalibrationHandler
{
	struct Measurements
	{
		float rmsLength;
		float meanLengthX;
		float meanLengthY;
		float stdDev;
		float stdDevX;
		float stdDevY;
	};

private:
	cv::Mat grayRegisteredColorFrame;

	float current, delta, fpsOfDetection;

public:
	// Calibration pattern variables
	cv::Size calibrationPatternSize;
	cv::Size minOffset;
	cv::Size maxOffset;
	UINT chessboardSquareLengthInPixels;
	UINT minChessboardSquareLengthInPixels = 40;
	float chessboardSquareLengthInMeters;
	Measurements measurements;

	// Calibration helper variables
	UINT maxCaptureAttempts = 5;
	UINT delay = 100;
	bool downsampleFrame, fastCheck, manualMode, showResults, showLogs;

	// Calibration process vairables
	std::vector<cv::Mat> collectionOfCalibrationPatterns;
	std::vector<std::vector<cv::Point2f>> generatedCalibrationPatternPoints;
	std::vector<std::vector<cv::Point2f>> detectedCalibrationPatternPointsInCameraFrame;
	std::vector<std::vector<cv::Point2f>> detectedCalibrationPatternPointsInProjectorFrame;
	std::vector<std::vector<cv::Point3f>> worldCoordinatesForGeneratedCalibrationPatternPoints;
	std::vector<bool> worldPointsToBeConsidered;

	// Projector display parameters
	cv::Mat undistortedProjectorFrame;
	cv::Mat rectifiedProjectorFrame;
	cv::Mat projectorFrame;
	cv::Size primaryScreenResolution;

	// Projector calibration parameters
	cv::Size projectorAspectRatio;
	cv::Size projectorFrameSize;
	cv::Mat projectorCalibrationMatrix = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat inverseProjectorCalibrationMatrix = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat projectorDistortionCoefficients = cv::Mat::zeros(8, 1, CV_64F);
	std::vector<cv::Mat> projectorRVectors, projectorTVectors;
	double projectorApertureWidth = 0, projectorApertureHeight = 0, projectorFocalLengthAspectRatio = 0;
	double projectorFoVX = 0, projectorFoVY = 0, projectorFocalLength = 0;
	cv::Point2d projectorFocalPoint = cv::Point2d();

	// Camera calibration parameters
	cv::Size cameraFrameSize;
	cv::Mat cameraCalibrationMatrix = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat inverseCameraCalibrationMatrix = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat cameraDistortionCoefficients = cv::Mat::zeros(8, 1, CV_64F);
	std::vector<cv::Mat> cameraRVectors, cameraTVectors;
	double cameraApertureWidth = 0, cameraApertureHeight = 0, cameraFocalLengthAspectRatio = 0;
	double cameraFoVX = 0, cameraFoVY = 0, cameraFocalLength = 0;
	cv::Point2d cameraFocalPoint = cv::Point2d();

	// Calibration result variables
	cv::Mat homography = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat inverseHomography = cv::Mat::eye(3, 3, CV_64F);

	// Stereo calibration parameters
	cv::Mat H1, H2;
	cv::Mat R1, R2, P1, P2, Q;
	cv::Mat map1x, map1y, map2x, map2y;

	CalibrationHandler(cv::Size cameraFrameSize, cv::Size primaryScreenResolution, cv::Size projectorFrameSize, cv::Size projectorAspectRatio, cv::Size chessboardSize, UINT minChessboardSquareLengthInPixels, bool downsampleFrame, bool fastCheck, bool manualMode, bool showResults, bool showLogs);
	~CalibrationHandler();

	void CalculateAndSetChessboardSquareEdgeLengthInpixels();
	void CreateWindowsForDisplayingFrames();
	void DisplayFrames();
	void DisplayFrames(bool displayProjectorFrame, bool displayUndistortedProjectorFrame, bool displayRectifiedProjectorFrame);
	void CreateChessBoardPatternImages();
	void CreateWorldCoordinatesForChessBoardCornerPositions();
	bool GetCalibrationPatternPointsInCurrentFrame(KinectWrapper& in_out_kinectFrameData, UINT frameNumber);
	void ApplyInverseHomographyAndWarpImage(cv::Mat& in_source, cv::Mat& out_destination);
	void ApplyHomographyAndWarpPoints(std::vector<std::vector<cv::Point2f>>& in_collectionOfSourcePoints, std::vector<std::vector<cv::Point2f>>& out_collectionOfDestinationPoints);
	void CollectCalibrationPatternPointsFromProjector(KinectWrapper& in_out_kinectFrameData);
	void StartCalibration(bool isCamera);
	void StartProcamCalibration();
};

