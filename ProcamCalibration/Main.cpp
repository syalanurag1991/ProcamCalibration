#include <Windows.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <Kinect.h>

#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

#define bytesPerPixel 4
#define downsampleSize 2

#define calibrationSquareDimension 0.015f													// Checkerboard square edge in meters

#define colorWidth 1920
#define colorHeight 1080
#define colorFrameArea colorWidth * colorHeight
#define colorDataLength colorFrameArea * bytesPerPixel

#define depthWidth 512
#define depthHeight 424
#define depthDataLength depthWidth * depthHeight

BYTE   colorData[colorDataLength];															// BGRA array containing color stream data
BYTE   vizDepthData[depthDataLength];														// BYTE array containing depth stream data for visualization
USHORT rawDepthData[depthDataLength];														// USHORT array containing depth stream data
USHORT maxDepth = 0;

Mat colorFrame;
Mat depthFrame;
Size colorFrameSize = Size(colorWidth, colorHeight);
Size depthFrameSize = Size(depthWidth, depthHeight);

// Kinect variables
IKinectSensor* sensor;																		// kinect sensor
IColorFrameReader* colorFrameReader;														// kinect color data source
IDepthFrameReader* depthFrameReader;														// kinect depth data source

// Calibration processing parameters
Mat grayFrame;																				// buffer for converting color frame to gray
bool found = false;																			// if pattern is detected by opencv function
float fpsOfDetection;																		// detection performance
clock_t current, delta;
Size chessboardSize = Size(9, 6);
vector<Mat> framesWithDetectedPattern;
vector<vector<Point2f>> allDetectedPatternPoints;


// Camera calibration parameters
Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
Mat distanceCoefficients = Mat::zeros(8, 1, CV_64F);

bool initKinect() {
	if (FAILED(GetDefaultKinectSensor(&sensor))) {
		return false;
	}
	if (sensor) {
		sensor->Open();
		IColorFrameSource* colorFrameSource = NULL;
		sensor->get_ColorFrameSource(&colorFrameSource);
		colorFrameSource->OpenReader(&colorFrameReader);
		if (colorFrameSource) {
			colorFrameSource->Release();
			colorFrameSource = NULL;
		}

		IDepthFrameSource* framesource = NULL;
		sensor->get_DepthFrameSource(&framesource);
		framesource->OpenReader(&depthFrameReader);
		if (framesource) {
			framesource->Release();
			framesource = NULL;
		}
		return true;
	} else
		return false;
}

void getColorData(BYTE* colorDataBuffer) {
	IColorFrame* colorFrame = NULL;
	if (SUCCEEDED(colorFrameReader->AcquireLatestFrame(&colorFrame)))
		colorFrame->CopyConvertedFrameDataToArray(colorDataLength, colorDataBuffer, ColorImageFormat_Bgra);
	if (colorFrame) colorFrame->Release();
}

void getDepthData(USHORT* depthDataBuffer, BYTE* vizDepthDataBuffer) {
	IDepthFrame* depthFrame = NULL;
	if (SUCCEEDED(depthFrameReader->AcquireLatestFrame(&depthFrame))) {
		depthFrame->CopyFrameDataToArray(depthDataLength, depthDataBuffer);
		for (int i = 0; i < depthDataLength; i++) {
			vizDepthDataBuffer[i] = (BYTE)(depthDataBuffer[i] >> 5);
			if (maxDepth < vizDepthDataBuffer[i])
				maxDepth = vizDepthDataBuffer[i];
		}
	}
	if (depthFrame) depthFrame->Release();
}

void TestOpenCV(bool showImage) {
	Mat img = imread("lena.png");
	if (showImage) {
		namedWindow("Test Image - Lena.png", WINDOW_NORMAL);
		imshow("Test Image - Lena.png", img);
		waitKey(0);
	}
	cout << "\n\n OpenCV is working properly. \n\n";
	return;
}

void CreateKnownBoardPositions(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
	for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
}

void GetCalibrationPatternPointsInFrame(Mat& in_out_frame, vector<vector<Point2f>>& out_allFoundCorners, vector<Mat>& out_savedFramesWithSuccessfulDetection, bool downsampleFrame, bool fastCheck, bool showResults)
{
	vector<Point2f> foundPoints, foundPoints_upsampled;
	Mat in_out_frame_downsampled;

	if (downsampleFrame)
	{
		resize(in_out_frame, in_out_frame_downsampled, Size(), 1 / (float)downsampleSize, 1 / (float)downsampleSize, CV_INTER_AREA);
		cvtColor(in_out_frame_downsampled, grayFrame, COLOR_BGR2GRAY);
	}
	else
		cvtColor(in_out_frame, grayFrame, COLOR_BGR2GRAY);

	try
	{
		if (fastCheck)
		{
			found = findChessboardCorners(grayFrame, chessboardSize, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE | CV_CALIB_CB_FILTER_QUADS);
			if (foundPoints.size() > 0)
				found = find4QuadCornerSubpix(grayFrame, foundPoints, Size(50, 50));
		}
		else
			found = findChessboardCorners(grayFrame, chessboardSize, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CV_CALIB_CB_FILTER_QUADS);
	
		if (found)
		{
			cout << "\nPattern detected ... press 'c' to capture or 'esc to reject'\n";
			if (downsampleFrame)
			{
				for (int i = 0; i < foundPoints.size(); i++)
					foundPoints_upsampled.push_back(Vec2f(foundPoints[i].x * (float)downsampleSize, foundPoints[i].y * (float)downsampleSize));
				if (showResults)
					drawChessboardCorners(in_out_frame, chessboardSize, foundPoints_upsampled, found);
			}
			else
			{
				if (showResults)
					drawChessboardCorners(in_out_frame, chessboardSize, foundPoints, found);
			}

			char key = waitKey(0);
			if (key == 'r')
				return;
			else if (key == 'c')
			{
				if (downsampleFrame)
					out_allFoundCorners.push_back(foundPoints_upsampled);
				else
					out_allFoundCorners.push_back(foundPoints);
				Mat saveThisFrame = in_out_frame.clone();
				out_savedFramesWithSuccessfulDetection.push_back(saveThisFrame);
				return;
			}
		}
	}
	catch (Exception e)
	{
		cout << "Caught exception";
	}
}

void CreateKnownChessBoardCornerPositions(Size boardSize, float squareEdgeLength, vector<Point3f>& out_knownChessBoardCorners)
{
	for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			out_knownChessBoardCorners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
}

void StartCalibration(vector<vector<Point2f>> in_allFoundCorners, vector<Mat>& calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraCalibrationMatrix)
{
	cout << "\n\n\n!!!! STARTING CALIBRATION !!!!\n\n\n";
	vector<vector<Point3f>> worldSpaceCornerPoints(1);
	CreateKnownChessBoardCornerPositions(chessboardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
	worldSpaceCornerPoints.resize(in_allFoundCorners.size(), worldSpaceCornerPoints[0]);
	vector<Mat> rVectors, tVectors;
	calibrateCamera(worldSpaceCornerPoints, in_allFoundCorners, colorFrameSize, cameraCalibrationMatrix, distanceCoefficients, rVectors, tVectors);
}

int main(int argc, char* argv[]) {

	/*VideoCapture vid(0);
	if (!vid.isOpened())
		return -1;
	vid.read(frame);*/

	if (!initKinect()) return 1;
	TestOpenCV(false);

	colorFrame = Mat(colorHeight, colorWidth, CV_8UC4, &colorData[0]);
	//depthFrame = Mat(depthHeight, depthWidth, CV_16UC1, &rawDepthData[0]);
	depthFrame = Mat(depthHeight, depthWidth, CV_8UC1, &vizDepthData[0]);

	namedWindow("Kinect COLOR", WINDOW_NORMAL);													// Create windows for display
	resizeWindow("Kinect COLOR", colorWidth / downsampleSize, colorHeight / downsampleSize);
	namedWindow("Kinect DEPTH", WINDOW_AUTOSIZE);

	int countTotalFrames = 0;
	while (1)
	{
		current = clock();

		getColorData(colorData);
		getDepthData(rawDepthData, vizDepthData);
		memcpy(colorFrame.data, colorData, colorDataLength);
		
		//vid.read(frame);
		
		bool downsampleFrame = true;
		GetCalibrationPatternPointsInFrame(colorFrame, allDetectedPatternPoints, framesWithDetectedPattern, downsampleFrame, true, true);

		memcpy(depthFrame.data, vizDepthData, depthDataLength);
		imshow("Kinect COLOR", colorFrame);														// Show color image
		imshow("Kinect DEPTH", depthFrame);														// Show depth image

		delta = clock() - current;
		fpsOfDetection = CLOCKS_PER_SEC / (float)delta;
		cout << setw(2) << setprecision(2)
			<< "Frame count: " << countTotalFrames++
			<< "\t FPS = " << fpsOfDetection
			<< "\t # of saved frames = " << framesWithDetectedPattern.size()
			<< "\t # of correspondences = " << allDetectedPatternPoints.size()
			<< "\t Max depth = " << maxDepth << endl;
		
		if (framesWithDetectedPattern.size() > 15)
			break;

		int key = waitKey(1);
		if (key == 27)
			break;
	}

	sensor->Close();
	sensor->Release();

	StartCalibration(allDetectedPatternPoints, framesWithDetectedPattern, chessboardSize, calibrationSquareDimension, cameraMatrix);
	double apertureWidth = 0, apertureHeight = 0, aspectRatio = 0;
	double fovX = 0, fovY = 0, focalLength = 0;
	Point2d focalPoint = Point2d();
	try
	{
		calibrationMatrixValues(cameraMatrix, colorFrameSize, apertureWidth, apertureHeight, fovX, fovY, focalLength, focalPoint, aspectRatio);
	}
	catch (Exception e)
	{
		cout << "\n" << e.msg << endl;
	}

	cout << "\n\n";

	for (int i = 0; i < cameraMatrix.rows; i++)
		for (int j = 0; j < cameraMatrix.cols; j++)
			cout << cameraMatrix.at<double>(i, j) << endl;
	
	cout << "\n\n";

	for (int i = 0; i < distanceCoefficients.rows; i++)
		for (int j = 0; j < distanceCoefficients.cols; j++)
			cout << distanceCoefficients.at<double>(i, j) << endl;

	cout << "\n\n";

	cout
		<< "Image width = " << colorFrameSize.width << endl
		<< "Image height = " << colorFrameSize.height << endl
		<< "Aperture width = " << apertureWidth << endl
		<< "Aperture height = " << apertureHeight << endl
		<< "Aspect ratio = " << aspectRatio << endl
		<< "fovX = " << fovX << endl
		<< "fovY = " << fovY << endl
		<< "focal length = " << focalLength << endl
		<< "focal point X = " << focalPoint.x << endl
		<< "focal point Y = "  << focalPoint.y << endl;

	cout << "\n\n";
	cout << "Press any key to exit...";
	waitKey(0);
	destroyAllWindows();

	return 0;
}