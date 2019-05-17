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

class KinectFrameData
{
public:
	UINT colorFrameDataLength;
	UINT depthFrameDataLength;
	UINT registeredColorFrameDataLength;
	UINT BPP;

	BYTE*   colorData;																// BGRA array containing color stream data
	BYTE*   vizDepthData;											// BYTE array containing depth stream data for visualization
	USHORT* rawDepthData;															// USHORT array containing depth stream data
	BYTE*   registeredColorData;									// BGRA array containing color stream data

	Mat colorStream;
	Mat depthStream;
	Mat registeredColorFrame;
	Mat registeredColorFrameUndistorted;
	Size colorFrameSize;
	Size depthFrameSize;
	Size registeredColorFrameSize;

	// Kinect variables
	IKinectSensor* sensor;																			// kinect sensor
	IColorFrameReader* colorFrameReader;															// kinect color data source
	IDepthFrameReader* depthFrameReader;															// kinect depth data source
	ICoordinateMapper* coordinateMapper;

	// Coordinate mapping parameters
	ColorSpacePoint* colorSpacePoints;
	CameraSpacePoint* cameraSpacePoints;

	// Calibration processing parameters
	Mat grayRegisteredColorFrame;																					// buffer for converting color frame to gray
	Mat downsampledRegisteredColorFrame;

	KinectFrameData(UINT colorFrameWidth, UINT colorFrameHeight, UINT depthFrameWidth, UINT depthFrameHeight, UINT bpp)
	{
		colorFrameDataLength = colorFrameWidth * colorFrameHeight * bpp;
		depthFrameDataLength = depthFrameWidth * depthFrameHeight;
		registeredColorFrameDataLength = depthFrameDataLength * bpp;
		BPP = bpp;

		colorData = new BYTE[colorFrameDataLength];
		vizDepthData = new BYTE[registeredColorFrameDataLength];
		rawDepthData = new USHORT[depthFrameDataLength];
		registeredColorData = new BYTE[registeredColorFrameDataLength];

		colorStream = Mat(colorFrameHeight, colorFrameWidth, CV_8UC4, &colorData[0]);
		depthStream = Mat(depthFrameHeight, depthFrameWidth, CV_8UC4, &vizDepthData[0]);
		registeredColorFrame = Mat(depthFrameHeight, depthFrameWidth, CV_8UC4, &registeredColorData[0]);

		colorFrameSize = Size(colorFrameWidth, colorFrameHeight);
		depthFrameSize = Size(depthFrameWidth, depthFrameHeight);
		registeredColorFrameSize = Size(depthFrameWidth, depthFrameHeight);

		colorSpacePoints = new ColorSpacePoint[depthFrameDataLength];
		cameraSpacePoints = new CameraSpacePoint[depthFrameDataLength];
	}
};

// Calibration processing parameters
float fpsOfDetection;																			// detection performance
clock_t current, delta;


// Camera calibration parameters
Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
Mat projectorMatrix = Mat::eye(3, 3, CV_64F);
Mat cameraDistortionCoefficients = Mat::zeros(8, 1, CV_64F);
Mat projectorDistortionCoefficients = Mat::zeros(8, 1, CV_64F);
float chessboardSquareLengthInMeters = 0.01f;													// Checkerboard square edge in meters

																								// Create chessboard
Mat chessboard;
int chessboardSquareLengthInPixels;
Size chessboardSize = Size(8, 5);
int minChessboardSquareDimension = 40;
int maxChessboardSquareDimension = 80;
int chessboardWidth;
int chessboardHeight;

// Projector variables
enum AspectRatio
{
	AspectRatio_4x3 = 0,
	AspectRatio_16x9 = 1
};

struct Measurements
{
	float rmsLength;
	float meanLengthX;
	float meanLengthY;
	float stdDev;
	float stdDevX;
	float stdDevY;
};

struct CalibrationPatternData
{
	Mat searchPattern;
	Mat testPattern;
	vector<Point2f> generatedCalibrationPatternPoints;

	vector<Mat> collectionOfSearchPatterns;
	vector<Mat> collectionOfTestPatterns;

	vector<vector<Point2f>> generatedCalibrationPatternPointsForImage;
	vector<vector<Point2f>> detectedCalibrationPatternPointsForImage;
	vector<vector<Point2f>> detectedCalibrationPatternPointsWarped;
	vector<vector<Point3f>> worldCoordinatesForGeneratedCalibrationPatternPoints;

	Size searchPatternSize;
	Size chessboardSize;
	Size minOffset;
	Size maxOffset;
	int chessboardSquareLengthInPixels;
	float chessboardSquareLengthInMeters;

	Measurements measurements;
	bool downsampleFrame, fastCheck, manualMode, showResults, showLogs;
};

Size ProjectorResolution;
AspectRatio ProjectorAspectRatio;

bool initKinect(KinectFrameData& in_kinectFrameData) {
	if (FAILED(GetDefaultKinectSensor(&(in_kinectFrameData.sensor)))) {
		return false;
	}
	if (in_kinectFrameData.sensor) {
		in_kinectFrameData.sensor->Open();
		IColorFrameSource* colorFrameSource = NULL;
		in_kinectFrameData.sensor->get_ColorFrameSource(&colorFrameSource);
		colorFrameSource->OpenReader(&(in_kinectFrameData.colorFrameReader));
		if (colorFrameSource) {
			colorFrameSource->Release();
			colorFrameSource = NULL;
		}

		IDepthFrameSource* framesource = NULL;
		in_kinectFrameData.sensor->get_DepthFrameSource(&framesource);
		framesource->OpenReader(&(in_kinectFrameData.depthFrameReader));
		if (framesource) {
			framesource->Release();
			framesource = NULL;
		}

		in_kinectFrameData.sensor->get_CoordinateMapper(&(in_kinectFrameData.coordinateMapper));

		return true;
	}
	else
		return false;
}

bool GetLatestColorDataFromKinect(KinectFrameData& in_kinectFrameData) {
	IColorFrame* colorFrame = NULL;

	if (SUCCEEDED(in_kinectFrameData.colorFrameReader->AcquireLatestFrame(&colorFrame)))
	{
		if (colorFrame)
		{
			colorFrame->CopyConvertedFrameDataToArray(in_kinectFrameData.colorFrameDataLength, in_kinectFrameData.colorData, ColorImageFormat_Bgra);
			colorFrame->Release();
			return true;
		}
		colorFrame->Release();
		return false;
	}
	return false;
}

bool GetLatestDepthDataFromKinect(KinectFrameData& in_kinectFrameData) {
	IDepthFrame* depthFrame = NULL;

	if (SUCCEEDED(in_kinectFrameData.depthFrameReader->AcquireLatestFrame(&depthFrame)))
	{
		if (depthFrame)
		{
			depthFrame->CopyFrameDataToArray(in_kinectFrameData.depthFrameDataLength, in_kinectFrameData.rawDepthData);
			depthFrame->Release();
			return true;
		}
		depthFrame->Release();
		return false;
	}
	return false;
}

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

void CreateKnownBoardPositions(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
	for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
}

bool GetCalibrationPatternPointsInFrame(
	KinectFrameData& in_out_kinectFrameData,
	CalibrationPatternData& in_out_chessboardCalibrationPatternData,
	UINT frameNumber)
{
	vector<Point2f> foundPoints, foundPoints_upsampled;
	Mat in_out_frame_downsampled;

	if (in_out_chessboardCalibrationPatternData.downsampleFrame)
	{
		resize(in_out_kinectFrameData.registeredColorFrame, in_out_kinectFrameData.downsampledRegisteredColorFrame, Size(), 1 / (float)downsampleSize, 1 / (float)downsampleSize, CV_INTER_AREA);
		cvtColor(in_out_kinectFrameData.downsampledRegisteredColorFrame, in_out_kinectFrameData.grayRegisteredColorFrame, COLOR_BGR2GRAY);
	}
	else
		cvtColor(in_out_kinectFrameData.registeredColorFrame, in_out_kinectFrameData.grayRegisteredColorFrame, COLOR_BGR2GRAY);

	bool found = false;																										// if pattern is detected by opencv function
	try
	{
		if (in_out_chessboardCalibrationPatternData.fastCheck)
		{
			found = findChessboardCorners(in_out_kinectFrameData.grayRegisteredColorFrame, in_out_chessboardCalibrationPatternData.chessboardSize, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK | CALIB_CB_NORMALIZE_IMAGE | CV_CALIB_CB_FILTER_QUADS);
			if (foundPoints.size() > 0)
				found = find4QuadCornerSubpix(in_out_kinectFrameData.grayRegisteredColorFrame, foundPoints, Size(50, 50));
		}
		else
			found = findChessboardCorners(in_out_kinectFrameData.grayRegisteredColorFrame, in_out_chessboardCalibrationPatternData.chessboardSize, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CV_CALIB_CB_FILTER_QUADS);

		if (found)
		{
			int pointIndex = 0;
			float sumX = 0;
			float sumY = 0;
			float squareSumX = 0;
			float squareSumY = 0;

			if (in_out_chessboardCalibrationPatternData.downsampleFrame)
			{
				float previousY = 0;
				for (int i = 0; i < in_out_chessboardCalibrationPatternData.chessboardSize.height; i++)
				{
					float differenceY = foundPoints[pointIndex].y - previousY;
					sumY += differenceY;
					squareSumY += differenceY * differenceY;
					previousY = foundPoints[pointIndex].y;
					float previousX = 0;
					for (int j = 0; j < in_out_chessboardCalibrationPatternData.chessboardSize.width; j++)
					{
						float differenceX = foundPoints[pointIndex].x - previousX;
						sumX += differenceX;
						squareSumX += differenceX * differenceX;
						previousX = foundPoints[pointIndex].x;

						float foundPointX = foundPoints[pointIndex].x * (float)downsampleSize;
						float foundPointY = foundPoints[pointIndex].y * (float)downsampleSize;
						int foundPointX_int = (int)foundPointX;
						int foundPointY_int = (int)foundPointY;
						int colorSpaceIndex = foundPointY_int * colorWidth + foundPointX_int;
						if (in_out_chessboardCalibrationPatternData.showLogs)
						{
							std::cout
								<< "#" << i << ": "
								<< " foundPointX = " << foundPointX_int
								<< " foundPointY = " << foundPointY_int
								<< " colorSpaceIndex = " << colorSpaceIndex
								<< " Camera Space X = " << in_out_kinectFrameData.cameraSpacePoints[colorSpaceIndex].X
								<< " Camera Space Y = " << in_out_kinectFrameData.cameraSpacePoints[colorSpaceIndex].Y
								<< " Camera Space Z = " << in_out_kinectFrameData.cameraSpacePoints[colorSpaceIndex].Z
								<< endl;
						}
						foundPoints_upsampled.push_back(Vec2f(foundPoints[pointIndex].x * (float)downsampleSize, foundPoints[pointIndex].y * (float)downsampleSize));
						pointIndex++;
					}
				}

				foundPoints.clear();
				foundPoints = foundPoints_upsampled;
			}
			else
			{
				for (int i = 0; i < in_out_chessboardCalibrationPatternData.chessboardSize.height; i++)
				{
					float previousX = 0;
					for (int j = 0; j < in_out_chessboardCalibrationPatternData.chessboardSize.width; j++)
					{
						int foundPointX_int = (int)foundPoints[pointIndex].x;
						int foundPointY_int = (int)foundPoints[pointIndex].y;
						int depthSpaceIndexForCurrentFoundPoint = foundPointY_int * depthWidth + foundPointX_int;
						float differenceX = 0, differenceY = 0;

						if (j != 0)
						{
							int previousFoundPointX_int = (int)foundPoints[pointIndex - 1].x;
							int depthSpaceIndexForPreviousFoundPointInX = foundPointY_int * depthWidth + previousFoundPointX_int;
							differenceX = std::abs(in_out_kinectFrameData.cameraSpacePoints[depthSpaceIndexForCurrentFoundPoint].X - in_out_kinectFrameData.cameraSpacePoints[depthSpaceIndexForPreviousFoundPointInX].X);
							sumX += differenceX;
							squareSumX += differenceX * differenceX;
						}
						previousX = in_out_kinectFrameData.cameraSpacePoints[depthSpaceIndexForCurrentFoundPoint].X;

						if (i != 0)
						{
							int previousFoundPointY_int = (int)foundPoints[pointIndex - in_out_chessboardCalibrationPatternData.chessboardSize.width].y;
							int depthSpaceIndexForPreviousFoundPointInY = previousFoundPointY_int * depthWidth + foundPointX_int;
							differenceY = std::abs(in_out_kinectFrameData.cameraSpacePoints[depthSpaceIndexForCurrentFoundPoint].Y - in_out_kinectFrameData.cameraSpacePoints[depthSpaceIndexForPreviousFoundPointInY].Y);
							sumY += differenceY;
							squareSumY += differenceY * differenceY;
						}

						if (in_out_chessboardCalibrationPatternData.showLogs)
						{
							std::cout
								<< "#" << pointIndex << " (" << j << ", " << i << ") : "
								<< " foundPointX = " << foundPointX_int
								<< " foundPointY = " << foundPointY_int
								<< " depthSpaceIndex = " << depthSpaceIndexForCurrentFoundPoint
								<< " Camera Space X = " << in_out_kinectFrameData.cameraSpacePoints[depthSpaceIndexForCurrentFoundPoint].X
								<< " Camera Space Y = " << in_out_kinectFrameData.cameraSpacePoints[depthSpaceIndexForCurrentFoundPoint].Y
								<< " Camera Space Z = " << in_out_kinectFrameData.cameraSpacePoints[depthSpaceIndexForCurrentFoundPoint].Z
								<< " difference X = " << differenceX
								<< " difference Y = " << differenceY
								<< endl;
						}
						pointIndex++;
					}
				}
			}

			in_out_chessboardCalibrationPatternData.measurements.meanLengthX = sumX / ((in_out_chessboardCalibrationPatternData.chessboardSize.width - 1) * in_out_chessboardCalibrationPatternData.chessboardSize.height);
			in_out_chessboardCalibrationPatternData.measurements.meanLengthY = sumY / (in_out_chessboardCalibrationPatternData.chessboardSize.width * (in_out_chessboardCalibrationPatternData.chessboardSize.height - 1));
			float expectationOfSquareOfDifferenceX = squareSumX / ((in_out_chessboardCalibrationPatternData.chessboardSize.width - 1) * in_out_chessboardCalibrationPatternData.chessboardSize.height);
			float expectationOfSquareOfDifferenceY = squareSumY / (in_out_chessboardCalibrationPatternData.chessboardSize.width * (in_out_chessboardCalibrationPatternData.chessboardSize.height - 1));
			float squareMeanMeasureX = in_out_chessboardCalibrationPatternData.measurements.meanLengthX * in_out_chessboardCalibrationPatternData.measurements.meanLengthX;
			float squareMeanMeasureY = in_out_chessboardCalibrationPatternData.measurements.meanLengthY * in_out_chessboardCalibrationPatternData.measurements.meanLengthY;
			float varianceOfDifferenceX = expectationOfSquareOfDifferenceX - squareMeanMeasureX;
			float varianceOfDifferenceY = expectationOfSquareOfDifferenceY - squareMeanMeasureY;
			in_out_chessboardCalibrationPatternData.measurements.stdDevX = sqrt(varianceOfDifferenceX);
			in_out_chessboardCalibrationPatternData.measurements.stdDevY = sqrt(varianceOfDifferenceY);
			in_out_chessboardCalibrationPatternData.measurements.rmsLength = sqrt(squareMeanMeasureX + squareMeanMeasureY) / sqrt(2);
			in_out_chessboardCalibrationPatternData.measurements.stdDev = sqrt(in_out_chessboardCalibrationPatternData.measurements.stdDevX * in_out_chessboardCalibrationPatternData.measurements.stdDevX + in_out_chessboardCalibrationPatternData.measurements.stdDevY * in_out_chessboardCalibrationPatternData.measurements.stdDevY) / sqrt(2);


			if (in_out_chessboardCalibrationPatternData.showResults)
			{
				drawChessboardCorners(in_out_kinectFrameData.registeredColorFrame, in_out_chessboardCalibrationPatternData.chessboardSize, foundPoints, found);
				std::cout << endl
					<< "\nsumX = " << sumX << endl
					<< "sumY = " << sumY << endl
					<< "squareSumX = " << squareSumX << endl
					<< "squareSumY = " << squareSumY << endl
					<< "# of points = " << foundPoints.size() << endl
					<< "meanMeasureX = " << in_out_chessboardCalibrationPatternData.measurements.meanLengthX << endl
					<< "meanMeasureY = " << in_out_chessboardCalibrationPatternData.measurements.meanLengthY << endl
					<< "meanMeasure = " << in_out_chessboardCalibrationPatternData.measurements.rmsLength << endl
					<< "standardDeviationX = " << in_out_chessboardCalibrationPatternData.measurements.stdDevX << endl
					<< "standardDeviationY = " << in_out_chessboardCalibrationPatternData.measurements.stdDevY << endl
					<< "standardDeviation = " << in_out_chessboardCalibrationPatternData.measurements.stdDev << endl;
			}

			if (in_out_chessboardCalibrationPatternData.manualMode)
			{
				std::cout << "\nPattern detected ... press 'c' to capture or 'r' to reject, frame # " << frameNumber << "\n";
				char key = cv::waitKey(0);
				if (key == 'r')
					return false;
				else if (key == 'c')
				{
					in_out_chessboardCalibrationPatternData.detectedCalibrationPatternPointsForImage.push_back(foundPoints);
					return true;
				}
				return false;
			}
			else
			{
				std::cout << "\nPattern detected ... standard deviation = " << in_out_chessboardCalibrationPatternData.measurements.stdDev << ", frame # " << frameNumber << "\n";
				cv::waitKey(delay);
				if (in_out_chessboardCalibrationPatternData.measurements.stdDev < 0.01)
				{
					in_out_chessboardCalibrationPatternData.detectedCalibrationPatternPointsForImage.push_back(foundPoints);
					return true;
				}
				return false;
			}

		}
		return false;
	}
	catch (Exception e)
	{
		std::cout << "Caught exception";
		return false;
	}

	return false;
}

void CreateWorldCoordinatesForChessBoardCornerPositions(CalibrationPatternData& calibrationPatternData)
{
	std::cout << "\n\nCreating world coordinates: \n\n";
	int pointNumber = 1;
	for (int yOffset = calibrationPatternData.minOffset.height; yOffset < calibrationPatternData.maxOffset.height; yOffset++)
	{
		for (int xOffset = calibrationPatternData.minOffset.width; xOffset < calibrationPatternData.maxOffset.width; xOffset++)
		{
			bool storeInReverseOrder = false;
			if ((xOffset + yOffset) % 2 != 0)
				storeInReverseOrder = true;

			vector<Point3f> newWorldCoordinatesForCalibrationPatternPoints;
			for (int y = yOffset; y <= yOffset + calibrationPatternData.chessboardSize.height; y++) {
				for (int x = xOffset; x <= xOffset + calibrationPatternData.chessboardSize.width; x++) {
					float xPosition = (float)x * calibrationPatternData.chessboardSquareLengthInMeters;
					float yPosition = (float)y * calibrationPatternData.chessboardSquareLengthInMeters;
					// don't include points at boundaries
					if (x != xOffset && y != yOffset)
					{
						std::cout << pointNumber++ << "# " << x << ", " << y << " = " << xPosition << ", " << yPosition << endl;
						Point3f chessboardPoint = Point3f((float)xPosition, (float)yPosition, 0.0f);
						if (!storeInReverseOrder)
							newWorldCoordinatesForCalibrationPatternPoints.push_back(chessboardPoint);
						else
							newWorldCoordinatesForCalibrationPatternPoints.insert(newWorldCoordinatesForCalibrationPatternPoints.begin(), chessboardPoint);
					}
				}
			}
			std::cout << endl;
			calibrationPatternData.worldCoordinatesForGeneratedCalibrationPatternPoints.push_back(newWorldCoordinatesForCalibrationPatternPoints);
		}
	}
}

void StartCalibration(bool isCamera, CalibrationPatternData& in_calibrationPatternData, Size& in_frameSize, Mat& out_cameraCalibrationMatrix, Mat& out_cameraDistanceCoefficients)
{
	std::cout << "\n\n\n!!!! STARTING CALIBRATION !!!!\n\n\n";
	std::cout << "Length of square in meters = " << in_calibrationPatternData.chessboardSquareLengthInMeters << endl;
	std::cout << "Found points vector size = " << in_calibrationPatternData.detectedCalibrationPatternPointsForImage.size() << " x " << in_calibrationPatternData.detectedCalibrationPatternPointsForImage[0].size() << endl;
	std::cout << "World coord. vector size = " << in_calibrationPatternData.worldCoordinatesForGeneratedCalibrationPatternPoints.size() << " x " << in_calibrationPatternData.worldCoordinatesForGeneratedCalibrationPatternPoints[0].size() << "\n\n";

	vector<Mat> rVectors, tVectors;
	if (isCamera) {
		std::cout << "\nCalibrating camera\n";
		double rms = calibrateCamera(in_calibrationPatternData.worldCoordinatesForGeneratedCalibrationPatternPoints, in_calibrationPatternData.detectedCalibrationPatternPointsForImage, in_frameSize, out_cameraCalibrationMatrix, out_cameraDistanceCoefficients, rVectors, tVectors);
		std::cout << "RMS error for calibration = " << rms << endl;
	}
	else
	{
		std::cout << "\nCalibrating projector\n";
		double rms = calibrateCamera(in_calibrationPatternData.worldCoordinatesForGeneratedCalibrationPatternPoints, in_calibrationPatternData.detectedCalibrationPatternPointsWarped, in_frameSize, out_cameraCalibrationMatrix, out_cameraDistanceCoefficients, rVectors, tVectors);
		std::cout << "RMS error for calibration = " << rms << endl;
	}
}

// to apply a shader just change the depth parameters from the kinects point of view
void UpdateDepthAndCameraSpaceMapping(KinectFrameData& in_out_kinectFrameData)
{
	in_out_kinectFrameData.coordinateMapper->MapDepthFrameToColorSpace(in_out_kinectFrameData.depthFrameDataLength, in_out_kinectFrameData.rawDepthData, in_out_kinectFrameData.depthFrameDataLength, in_out_kinectFrameData.colorSpacePoints);
	in_out_kinectFrameData.coordinateMapper->MapDepthFrameToCameraSpace(in_out_kinectFrameData.depthFrameDataLength, in_out_kinectFrameData.rawDepthData, in_out_kinectFrameData.depthFrameDataLength, in_out_kinectFrameData.cameraSpacePoints);
	for (int i = 0; i < in_out_kinectFrameData.depthFrameDataLength; i++) {
		in_out_kinectFrameData.vizDepthData[4 * i] = (BYTE)(in_out_kinectFrameData.rawDepthData[i] >> 5);
		in_out_kinectFrameData.vizDepthData[4 * i + 1] = in_out_kinectFrameData.vizDepthData[4 * i];
		in_out_kinectFrameData.vizDepthData[4 * i + 2] = in_out_kinectFrameData.vizDepthData[4 * i];
		in_out_kinectFrameData.vizDepthData[4 * i + 3] = 255;
		//out_vizDepthData[y] = (BYTE)(in_depthFrameData[y] >> 5);
		int colorSpacePointX = (int)in_out_kinectFrameData.colorSpacePoints[i].X;
		int colorSpacePointY = (int)in_out_kinectFrameData.colorSpacePoints[i].Y;
		UINT colorSpaceIndex = 0;
		if ((colorSpacePointX >= 0) && (colorSpacePointX < in_out_kinectFrameData.colorFrameSize.width) && (colorSpacePointY >= 0) && (colorSpacePointY < in_out_kinectFrameData.colorFrameSize.height))
		{
			colorSpaceIndex = in_out_kinectFrameData.BPP * (colorSpacePointY * in_out_kinectFrameData.colorFrameSize.width + colorSpacePointX);
			in_out_kinectFrameData.registeredColorData[in_out_kinectFrameData.BPP * i] = in_out_kinectFrameData.colorData[colorSpaceIndex];
			in_out_kinectFrameData.registeredColorData[in_out_kinectFrameData.BPP * i + 1] = in_out_kinectFrameData.colorData[colorSpaceIndex + 1];
			in_out_kinectFrameData.registeredColorData[in_out_kinectFrameData.BPP * i + 2] = in_out_kinectFrameData.colorData[colorSpaceIndex + 2];
			in_out_kinectFrameData.registeredColorData[in_out_kinectFrameData.BPP * i + 3] = 255;
		}
	}

	memcpy(in_out_kinectFrameData.colorStream.data, in_out_kinectFrameData.colorData, in_out_kinectFrameData.colorFrameDataLength);
	//memcpy(in_out_kinectFrameData.depthFrame.data, in_out_kinectFrameData.vizDepthData, in_out_kinectFrameData.depthDataLength);
	memcpy(in_out_kinectFrameData.depthStream.data, in_out_kinectFrameData.vizDepthData, in_out_kinectFrameData.registeredColorFrameDataLength);
	memcpy(in_out_kinectFrameData.registeredColorFrame.data, in_out_kinectFrameData.registeredColorData, in_out_kinectFrameData.registeredColorFrameDataLength);
	flip(in_out_kinectFrameData.registeredColorFrame, in_out_kinectFrameData.registeredColorFrame, 1);
}

void CreateChessBoardImage(Size in_chessboardSize, UINT in_chessboardSquareLengthInPixels, Size in_ProjectorResolution, Size in_offset, CalibrationPatternData& out_chessboardCalibrationData)
{
	// Setup chessboard
	Scalar colorBlack = Scalar(0, 0, 0, 255);
	Scalar colorWhite = Scalar(255, 255, 255, 255);
	Scalar colorRed = Scalar(255, 0, 0, 255);
	Scalar colorBlue = Scalar(0, 0, 255, 255);
	UINT chessboardWidth = (in_chessboardSize.width + 1) * in_chessboardSquareLengthInPixels;
	UINT chessboardHeight = (in_chessboardSize.height + 1) * in_chessboardSquareLengthInPixels;

	out_chessboardCalibrationData.searchPattern = Mat(in_ProjectorResolution.height, in_ProjectorResolution.width, CV_8UC4, colorWhite);
	out_chessboardCalibrationData.testPattern = Mat(chessboardHeight, chessboardWidth, CV_8UC4, colorWhite);

	bool storeInReverseOrder = false;
	if ((in_offset.height + in_offset.width) % 2 != 0)
		storeInReverseOrder = true;

	vector<Point2f> newCalibrationPatternPoints;
	for (int y = in_offset.height, y_searchPattern = 0; y <= in_offset.height + in_chessboardSize.height; y++, y_searchPattern += in_chessboardSquareLengthInPixels) {
		for (int x = in_offset.width, x_searchPattern = 0; x <= in_offset.width + in_chessboardSize.width; x++, x_searchPattern += in_chessboardSquareLengthInPixels) {
			int xPosition = x * in_chessboardSquareLengthInPixels;
			int yPosition = y * in_chessboardSquareLengthInPixels;
			// don't include points at boundaries
			if (x != in_offset.width && y != in_offset.height)
			{
				Point2f chessboardPoint = Point2f((float)xPosition, (float)yPosition);
				if (!storeInReverseOrder)
				{
					out_chessboardCalibrationData.generatedCalibrationPatternPoints.push_back(chessboardPoint);
					newCalibrationPatternPoints.push_back(chessboardPoint);
				}
				else
				{
					out_chessboardCalibrationData.generatedCalibrationPatternPoints.insert(out_chessboardCalibrationData.generatedCalibrationPatternPoints.begin(), chessboardPoint);
					newCalibrationPatternPoints.insert(newCalibrationPatternPoints.begin(), chessboardPoint);
				}
			}

			Rect chessboardBlock(xPosition, yPosition, in_chessboardSquareLengthInPixels, in_chessboardSquareLengthInPixels);
			Rect chessboardBlock_searchPattern(x_searchPattern, y_searchPattern, in_chessboardSquareLengthInPixels, in_chessboardSquareLengthInPixels);
			if ((x + y) % 2 == 0)
			{
				rectangle(out_chessboardCalibrationData.searchPattern, chessboardBlock, colorBlack, -1, 8);
				rectangle(out_chessboardCalibrationData.testPattern, chessboardBlock_searchPattern, colorBlue, -1, 8);
			}
			else
			{
				rectangle(out_chessboardCalibrationData.searchPattern, chessboardBlock, colorWhite, -1, 8);
				rectangle(out_chessboardCalibrationData.testPattern, chessboardBlock_searchPattern, colorRed, -1, 8);
			}
		}
	}

	out_chessboardCalibrationData.generatedCalibrationPatternPointsForImage.push_back(newCalibrationPatternPoints);
}

void CreateChessBoardPatternImages(CalibrationPatternData& in_out_chessboardCalibrationData)
{
	// Setup chessboard colors and dimensions
	Scalar colorBlack = Scalar(0, 0, 0, 255);
	Scalar colorWhite = Scalar(255, 255, 255, 255);
	Scalar colorRed = Scalar(255, 0, 0, 255);
	Scalar colorBlue = Scalar(0, 0, 255, 255);
	UINT chessboardWidth = (in_out_chessboardCalibrationData.chessboardSize.width + 1) * in_out_chessboardCalibrationData.chessboardSquareLengthInPixels;
	UINT chessboardHeight = (in_out_chessboardCalibrationData.chessboardSize.height + 1) * in_out_chessboardCalibrationData.chessboardSquareLengthInPixels;

	int cornerCount = 1;
	for (int yOffset = in_out_chessboardCalibrationData.minOffset.height; yOffset < in_out_chessboardCalibrationData.maxOffset.height; yOffset++)
		//for (int yOffset = in_out_chessboardCalibrationData.minOffset.height; yOffset < in_out_chessboardCalibrationData.minOffset.height + 1; yOffset++)
	{
		for (int xOffset = in_out_chessboardCalibrationData.minOffset.width; xOffset < in_out_chessboardCalibrationData.maxOffset.width; xOffset++)
			//for (int xOffset = in_out_chessboardCalibrationData.minOffset.width; xOffset < in_out_chessboardCalibrationData.minOffset.width + 1; xOffset++)
		{
			std::cout << "\n Offset = (" << xOffset << ", " << yOffset << ")" << endl << "Corner locations:" << endl;

			Mat searchPattern = Mat(in_out_chessboardCalibrationData.searchPatternSize.height, in_out_chessboardCalibrationData.searchPatternSize.width, CV_8UC4, colorWhite);
			Mat testPattern = Mat(chessboardHeight, chessboardWidth, CV_8UC4, colorWhite);

			bool storeInReverseOrder = false;
			if ((xOffset + yOffset) % 2 != 0)
				storeInReverseOrder = true;

			vector<Point2f> newCalibrationPatternPoints;
			for (int y = yOffset, y_searchPattern = 0; y <= yOffset + in_out_chessboardCalibrationData.chessboardSize.height; y++, y_searchPattern += in_out_chessboardCalibrationData.chessboardSquareLengthInPixels) {
				for (int x = xOffset, x_searchPattern = 0; x <= xOffset + in_out_chessboardCalibrationData.chessboardSize.width; x++, x_searchPattern += in_out_chessboardCalibrationData.chessboardSquareLengthInPixels) {
					int xPosition = x * in_out_chessboardCalibrationData.chessboardSquareLengthInPixels;
					int yPosition = y * in_out_chessboardCalibrationData.chessboardSquareLengthInPixels;
					std::cout << cornerCount++ << " @ (" << xPosition << ", " << yPosition << ")" << endl;
					// don't include points at boundaries
					if (x != xOffset && y != yOffset)
					{
						Point2f chessboardPoint = Point2f((float)xPosition, (float)yPosition);
						if (!storeInReverseOrder)
							newCalibrationPatternPoints.push_back(chessboardPoint);
						else
							newCalibrationPatternPoints.insert(newCalibrationPatternPoints.begin(), chessboardPoint);
					}

					Rect chessboardBlock(xPosition, yPosition, in_out_chessboardCalibrationData.chessboardSquareLengthInPixels, in_out_chessboardCalibrationData.chessboardSquareLengthInPixels);
					Rect chessboardBlock_searchPattern(x_searchPattern, y_searchPattern, in_out_chessboardCalibrationData.chessboardSquareLengthInPixels, in_out_chessboardCalibrationData.chessboardSquareLengthInPixels);
					if ((x + y) % 2 == 0)
					{
						rectangle(searchPattern, chessboardBlock, colorBlack, -1, 8);
						rectangle(testPattern, chessboardBlock_searchPattern, colorBlue, -1, 8);
					}
					else
					{
						rectangle(searchPattern, chessboardBlock, colorWhite, -1, 8);
						rectangle(testPattern, chessboardBlock_searchPattern, colorRed, -1, 8);
					}
				}
			}
			in_out_chessboardCalibrationData.collectionOfSearchPatterns.push_back(searchPattern);
			in_out_chessboardCalibrationData.collectionOfTestPatterns.push_back(testPattern);
			in_out_chessboardCalibrationData.generatedCalibrationPatternPointsForImage.push_back(newCalibrationPatternPoints);

			imshow("PROJECTOR", searchPattern);
			cv::waitKey(delay);
		}
	}
}

void ApplyInverseHomographyAndWarpImage(Mat& in_inv_homography, Mat& in_source, Mat& out_destination)
{
	double denominator = 1;
	int newX = 0, newY = 0;
	for (int y = 0; y < out_destination.rows; y++)
	{
		for (int x = 0; x < out_destination.cols; x++)
		{
			denominator = in_inv_homography.at<double>(2, 0) * x + in_inv_homography.at<double>(2, 1) * y + in_inv_homography.at<double>(2, 2);
			newX = (int)((in_inv_homography.at<double>(0, 0) * x + in_inv_homography.at<double>(0, 1) * y + in_inv_homography.at<double>(0, 2)) / denominator);
			newY = (int)((in_inv_homography.at<double>(1, 0) * x + in_inv_homography.at<double>(1, 1) * y + in_inv_homography.at<double>(1, 2)) / denominator);

			if (newX >= 0 && newX < in_source.cols && newY >= 0 && newY < in_source.rows)
				out_destination.at<Vec4b>(y, x) = in_source.at<Vec4b>(newY, newX);
		}
	}
}

void ApplyHomographyAndWarpPoints(Mat& in_homography, vector<vector<Point2f>> in_collectionOfSourcePoints, vector<vector<Point2f>>& out_collectionOfDestinationPoints)
{
	float x, y;
	float denominator = 1, newX = 0, newY = 0;
	vector<Point2f> latestSourcePoints = in_collectionOfSourcePoints[in_collectionOfSourcePoints.size() - 1];
	vector<Point2f> latestDestinationPoints;
	for (int i = 0; i < latestSourcePoints.size(); i++)
	{
		x = latestSourcePoints[i].x;
		y = latestSourcePoints[i].y;
		denominator = ((float)in_homography.at<double>(2, 0) * x) + ((float)in_homography.at<double>(2, 1) * y) + (float)in_homography.at<double>(2, 2);
		newX = (((float)in_homography.at<double>(0, 0) * x) + ((float)in_homography.at<double>(0, 1) * y) + (float)in_homography.at<double>(0, 2)) / denominator;
		newY = (((float)in_homography.at<double>(1, 0) * x) + ((float)in_homography.at<double>(1, 1) * y) + (float)in_homography.at<double>(1, 2)) / denominator;
		Point2f newDestinationPoint = Point2f(newX, newY);
		latestDestinationPoints.push_back(newDestinationPoint);
	}

	out_collectionOfDestinationPoints.push_back(latestDestinationPoints);
	std::cout << "\n\nsize of latest destination points collection = " << out_collectionOfDestinationPoints.size() << " x " << latestDestinationPoints.size() << "\n\n";
}

void CollectCalibrationPatternPointsFromProjector(
	Mat& homography, Mat& tempResultPlaceholder,
	KinectFrameData& in_out_kinectFrameData,
	CalibrationPatternData& in_calibrationPatternData)
{
	int countTotalFrames = 0;
	int key = -1;
	bool firstAttempt = true;
	bool patternCaptured = false;
	bool gotLatestData = false;

	Mat inv_homography;
	std::cout << "tempResult placeholder stream resolution              = " << tempResultPlaceholder.cols << " x " << tempResultPlaceholder.rows << endl;

	for (int i = 0; i < in_calibrationPatternData.collectionOfSearchPatterns.size();)
	{
		current = clock();

		std::cout << "\n\nFrame count: " << countTotalFrames << endl;
		imshow("PROJECTOR", in_calibrationPatternData.collectionOfSearchPatterns[i]);														// Show chessboard image
		cv::waitKey(delay);

		gotLatestData = GetLatestColorDataFromKinect(in_out_kinectFrameData) & GetLatestDepthDataFromKinect(in_out_kinectFrameData);

		if (gotLatestData)
		{
			UpdateDepthAndCameraSpaceMapping(in_out_kinectFrameData);

			imshow("Kinect COLOR", in_out_kinectFrameData.colorStream);																		// Show color image
			imshow("Kinect DEPTH", in_out_kinectFrameData.depthStream);																		// Show depth image
			imshow("Kinect REGISTERED COLOR", in_out_kinectFrameData.registeredColorFrame);													// Show registered color image

																																			//patternCaptured = GetCalibrationPatternPointsInFrame(in_out_kinectFrameData.registeredColorFrame, in_calibrationPatternData.chessboardSize, countTotalFrames, in_calibrationPatternData.detectedCalibrationPatternPointsForImage, chessboardSquareEdgeMeasurements, downsampleFrame, fastCheck, manualMode, showResults, showLogs);
			patternCaptured = GetCalibrationPatternPointsInFrame(in_out_kinectFrameData, in_calibrationPatternData, countTotalFrames);
			std::cout << "chessboard size = " << in_calibrationPatternData.chessboardSize.width << " x " << in_calibrationPatternData.chessboardSize.height << endl;
			//drawChessboardCorners(chessboardImage, in_chessboardSize, chessboardPointsProjectedOnProjector, true);
			drawChessboardCorners(in_calibrationPatternData.collectionOfSearchPatterns[i], in_calibrationPatternData.chessboardSize, in_calibrationPatternData.generatedCalibrationPatternPointsForImage[i], true);
			//imshow("PROJECTOR", chessboardImage);																						// Show chessboard image
			imshow("PROJECTOR", in_calibrationPatternData.collectionOfSearchPatterns[i]);													// Show chessboard image

																																			// delay for view
			cv::waitKey(delay);

			vector<Point2f> sourcePoints, destinationPoints;

			if (patternCaptured && in_calibrationPatternData.detectedCalibrationPatternPointsForImage.size() > 0)
			{
				std::cout << "\n # of points detected = " << in_calibrationPatternData.detectedCalibrationPatternPointsForImage.size() << " x " << in_calibrationPatternData.detectedCalibrationPatternPointsForImage[in_calibrationPatternData.detectedCalibrationPatternPointsForImage.size() - 1].size() << ", # of points projected = " << in_calibrationPatternData.generatedCalibrationPatternPointsForImage[i].size() << endl;
				if (in_calibrationPatternData.detectedCalibrationPatternPointsForImage[in_calibrationPatternData.detectedCalibrationPatternPointsForImage.size() - 1].size() > 0 && in_calibrationPatternData.generatedCalibrationPatternPointsForImage[i].size() > 0 &&
					in_calibrationPatternData.detectedCalibrationPatternPointsForImage[in_calibrationPatternData.detectedCalibrationPatternPointsForImage.size() - 1].size() == in_calibrationPatternData.generatedCalibrationPatternPointsForImage[i].size())
				{
					sourcePoints.insert(sourcePoints.end(), in_calibrationPatternData.detectedCalibrationPatternPointsForImage[in_calibrationPatternData.detectedCalibrationPatternPointsForImage.size() - 1].begin(), in_calibrationPatternData.detectedCalibrationPatternPointsForImage[in_calibrationPatternData.detectedCalibrationPatternPointsForImage.size() - 1].end());
					destinationPoints.insert(destinationPoints.end(), in_calibrationPatternData.generatedCalibrationPatternPointsForImage[i].begin(), in_calibrationPatternData.generatedCalibrationPatternPointsForImage[i].end());
					for (int i = 0; i < sourcePoints.size(); i++)
					{
						std::cout << setw(4) << setprecision(4) << "#" << i << " (x, y) = " << sourcePoints[i].x << ", " << sourcePoints[i].y << " --> " << destinationPoints[i].x << ", " << destinationPoints[i].y << endl;
					}

					homography = findHomography(sourcePoints, destinationPoints);
					//warpPerspective(registeredColorFrame, tempResultPlaceholder, homography, out_chessboardCalibrationData.searchPattern.size());
					cv::invert(homography, inv_homography);
					std::cout << "tempResult placeholder stream resolution              = " << tempResultPlaceholder.cols << " x " << tempResultPlaceholder.rows << endl;
					ApplyInverseHomographyAndWarpImage(inv_homography, in_out_kinectFrameData.registeredColorFrame, tempResultPlaceholder);
					ApplyHomographyAndWarpPoints(homography, in_calibrationPatternData.detectedCalibrationPatternPointsForImage, in_calibrationPatternData.detectedCalibrationPatternPointsWarped);
					int radius = 5, thickness = -1;
					for (int i = 0; i < in_calibrationPatternData.detectedCalibrationPatternPointsWarped[in_calibrationPatternData.detectedCalibrationPatternPointsWarped.size() - 1].size(); i++)
						cv::circle(tempResultPlaceholder, in_calibrationPatternData.detectedCalibrationPatternPointsWarped[in_calibrationPatternData.detectedCalibrationPatternPointsWarped.size() - 1][i], radius, Scalar(255, 0, 255, 255), thickness, LINE_8, 0);

					std::cout << "registered stream resolution              = " << in_out_kinectFrameData.registeredColorFrame.cols << " x " << in_out_kinectFrameData.registeredColorFrame.rows << endl;
					std::cout << "tempResult1 stream resolution              = " << tempResultPlaceholder.cols << " x " << tempResultPlaceholder.rows << endl;
					std::cout << "resolution of image for searching pattern = " << in_calibrationPatternData.collectionOfSearchPatterns[i].cols << " x " << in_calibrationPatternData.collectionOfSearchPatterns[i].rows << endl;
					std::cout << "points generated (destination)            = " << destinationPoints.size() << endl;
					std::cout << "points found (source)                     = " << sourcePoints.size() << endl;
					imshow("PROJECTOR", tempResultPlaceholder);																			// show chessboard image
					if (in_calibrationPatternData.manualMode)
						cv::waitKey(0);
					else
						cv::waitKey(2 * delay);																								// delay for view
				}
			}

			imshow("Kinect REGISTERED COLOR", in_out_kinectFrameData.registeredColorFrame);													// show registered color image
			in_calibrationPatternData.chessboardSquareLengthInMeters = in_calibrationPatternData.measurements.rmsLength;

			delta = clock() - current;
			fpsOfDetection = CLOCKS_PER_SEC / (float)delta;
			countTotalFrames++;

			std::cout << setw(2) << setprecision(4)
				<< "\nFPS = " << fpsOfDetection
				<< "\t patternCaptured = " << patternCaptured << " *** "
				<< "\t # of correspondences = " << in_calibrationPatternData.detectedCalibrationPatternPointsForImage.size()
				<< "\t key = " << key
				<< endl;

			if (patternCaptured)
				i++;
		}
	}
}

int main(int argc, char* argv[]) {

	// Projector setup
	Size ProjectorFrameSize = Size(1024, 768);
	AspectRatio ProjectorAspectRatio = AspectRatio_4x3;
	std::cout << "Projector resolution = " << ProjectorFrameSize.width << " x " << ProjectorFrameSize.height << endl;
	if (ProjectorAspectRatio == AspectRatio_4x3)
	{
		std::cout << "Projector aspect ratio = " << "4:3" << endl;
		chessboardSquareLengthInPixels = ProjectorFrameSize.width / 4;
	}
	else if (ProjectorAspectRatio == AspectRatio_16x9)
	{
		std::cout << "Projector aspect ratio = " << "16:9" << endl;
		chessboardSquareLengthInPixels = ProjectorFrameSize.width / 16;
	}

	int scaleBy = (int)(chessboardSquareLengthInPixels / (float)minChessboardSquareDimension);
	while (scaleBy >= 2)
	{
		chessboardSquareLengthInPixels >>= 1;
		scaleBy = (int)(chessboardSquareLengthInPixels / (float)minChessboardSquareDimension);
	}

	std::cout << "Chessboard size = " << chessboardSize.width << " x " << chessboardSize.height << endl;
	std::cout << "Chessboard square length (pixels) = " << chessboardSquareLengthInPixels << endl;

	// Create chessboard pattern
	int monitorResolution = 1920;
	namedWindow("PROJECTOR", WINDOW_NORMAL);
	moveWindow("PROJECTOR", monitorResolution, 0);
	setWindowProperty("PROJECTOR", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

	namedWindow("PROJECTOR - UNDISTORTED", WINDOW_NORMAL);
	moveWindow("PROJECTOR - UNDISTORTED", monitorResolution, 0);
	setWindowProperty("PROJECTOR - UNDISTORTED", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

	namedWindow("PROJECTOR - RECTIFIED", WINDOW_NORMAL);
	moveWindow("PROJECTOR - RECTIFIED", monitorResolution, 0);
	setWindowProperty("PROJECTOR - RECTIFIED", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

	/*VideoCapture vid(0);
	if (!vid.isOpened())
	return -1;
	vid.read(frame);*/

	// Initialize Kinect
	KinectFrameData kinectFrameData = KinectFrameData(colorWidth, colorHeight, depthWidth, depthHeight, bytesPerPixel);
	if (!initKinect(kinectFrameData)) return 1;
	TestOpenCV(false);

	namedWindow("Kinect COLOR", WINDOW_NORMAL);													// Create windows for display
	resizeWindow("Kinect COLOR", colorWidth / downsampleSize, colorHeight / downsampleSize);
	namedWindow("Kinect DEPTH", WINDOW_AUTOSIZE);
	namedWindow("Kinect REGISTERED COLOR", WINDOW_AUTOSIZE);
	namedWindow("Kinect REGISTERED COLOR - UNDISTORTED", WINDOW_AUTOSIZE);
	namedWindow("Kinect REGISTERED COLOR - RECTIFIED", WINDOW_AUTOSIZE);

	int numberOfFramesWithPatternToDetect = 100;
	int maxOffsetX = ProjectorFrameSize.width / chessboardSquareLengthInPixels - chessboardSize.width - 1;
	int maxOffsetY = ProjectorFrameSize.height / chessboardSquareLengthInPixels - chessboardSize.height - 1;
	Size minChessboardOffset = Size(1, 1);
	Size maxChessboardOffset = Size(maxOffsetX, maxOffsetY);

	/*CollectCalibrationPatternPoints(
	colorDataLength, colorData, colorFrame,
	depthDataLength, rawDepthData, vizDepthData, depthFrame,
	registeredColorDataLength, registeredColorData, registeredColorFrame,
	coordinateMapper, colorSpacePoints, cameraSpacePoints,
	allDetectedPatternPoints, numberOfFramesWithPatternToDetect,
	downsampleFrame, fastCheck, showResults, showLogs);*/

	Mat projectorFrame = Mat(ProjectorFrameSize.height, ProjectorFrameSize.width, CV_8UC4);
	Mat projectorFrameUndistorted = Mat(ProjectorFrameSize.height, ProjectorFrameSize.width, CV_8UC4);
	std::cout << "#############tempResult stream resolution              = " << projectorFrame.cols << " x " << projectorFrame.rows << endl;

	CalibrationPatternData chessboardCalibrationData;
	chessboardCalibrationData.searchPatternSize = ProjectorFrameSize;
	chessboardCalibrationData.chessboardSize = chessboardSize;
	chessboardCalibrationData.minOffset = minChessboardOffset;
	chessboardCalibrationData.maxOffset = maxChessboardOffset;
	chessboardCalibrationData.chessboardSquareLengthInPixels = chessboardSquareLengthInPixels;
	chessboardCalibrationData.downsampleFrame = false;
	chessboardCalibrationData.fastCheck = false;
	chessboardCalibrationData.manualMode = false;
	//chessboardCalibrationData.manualMode = true;
	chessboardCalibrationData.showResults = true;
	chessboardCalibrationData.showLogs = true;

	//CreateChessBoardImage(chessboardCalibrationData);
	CreateChessBoardPatternImages(chessboardCalibrationData);

	Mat homography;

	CollectCalibrationPatternPointsFromProjector(
		homography, projectorFrame,
		kinectFrameData,
		chessboardCalibrationData);

	CreateWorldCoordinatesForChessBoardCornerPositions(chessboardCalibrationData);

	//waitKey(0);
	// return 0;

	Mat inv_homography;
	invert(homography, inv_homography);

	double m11 = homography.at<double>(0, 0);
	double m12 = homography.at<double>(0, 1);
	double m13 = homography.at<double>(0, 2);

	double m21 = homography.at<double>(1, 0);
	double m22 = homography.at<double>(1, 1);
	double m23 = homography.at<double>(1, 2);

	double m31 = homography.at<double>(2, 0);
	double m32 = homography.at<double>(2, 1);
	double m33 = homography.at<double>(2, 2);

	double inv_m11 = inv_homography.at<double>(0, 0);
	double inv_m12 = inv_homography.at<double>(0, 1);
	double inv_m13 = inv_homography.at<double>(0, 2);

	double inv_m21 = inv_homography.at<double>(1, 0);
	double inv_m22 = inv_homography.at<double>(1, 1);
	double inv_m23 = inv_homography.at<double>(1, 2);

	double inv_m31 = inv_homography.at<double>(2, 0);
	double inv_m32 = inv_homography.at<double>(2, 1);
	double inv_m33 = inv_homography.at<double>(2, 2);

	std::cout << "\n\n Homography size: " << homography.cols << " x " << homography.rows << "\n\n";
	std::cout <<
		"[" << m11 << ", " << m12 << ", " << m13 << "]\n" <<
		"[" << m21 << ", " << m22 << ", " << m23 << "]\n" <<
		"[" << m31 << ", " << m32 << ", " << m33 << "]\n";

	std::cout << "\n\n Inverted Homography size: " << inv_homography.cols << " x " << inv_homography.rows << "\n\n";
	std::cout <<
		"[" << inv_m11 << ", " << inv_m12 << ", " << inv_m13 << "]\n" <<
		"[" << inv_m21 << ", " << inv_m22 << ", " << inv_m23 << "]\n" <<
		"[" << inv_m31 << ", " << inv_m32 << ", " << inv_m33 << "]\n";

	int newX, newY;
	while (1)
	{
		GetLatestColorDataFromKinect(kinectFrameData) & GetLatestDepthDataFromKinect(kinectFrameData);
		UpdateDepthAndCameraSpaceMapping(kinectFrameData);
		warpPerspective(kinectFrameData.registeredColorFrame, projectorFrame, homography, projectorFrame.size());
		//warpPerspective(depthFrame, tempResultPlaceholder, homography, tempResultPlaceholder.size());

		imshow("Kinect COLOR", kinectFrameData.colorStream);																		// Show color image
		imshow("Kinect DEPTH", kinectFrameData.depthStream);																		// Show depth image
		imshow("Kinect REGISTERED COLOR", kinectFrameData.registeredColorFrame);													// Show registered color image
		imshow("PROJECTOR", projectorFrame);

		char key = cv::waitKey(30);
		if (key == 27)
		{
			std::cout << "\nExiting streams...\n";
			break;
		}
	}

	std::cout << "\n\n length in meters: " << chessboardCalibrationData.chessboardSquareLengthInMeters << endl;
	StartCalibration(true, chessboardCalibrationData, kinectFrameData.registeredColorFrameSize, cameraMatrix, cameraDistortionCoefficients);

	double cameraApertureWidth = 0, cameraApertureHeight = 0, cameraAspectRatio = 0;
	double cameraFoVX = 0, cameraFoVY = 0, cameraFocalLength = 0;
	Point2d cameraFocalPoint = Point2d();
	try
	{
		calibrationMatrixValues(cameraMatrix, kinectFrameData.registeredColorFrameSize, cameraApertureWidth, cameraApertureHeight, cameraFoVX, cameraFoVY, cameraFocalLength, cameraFocalPoint, cameraAspectRatio);
	}
	catch (Exception e)
	{
		std::cout << "\n" << e.msg << endl;
	}

	std::cout << "\n\n";

	for (int i = 0; i < cameraMatrix.rows; i++)
		for (int j = 0; j < cameraMatrix.cols; j++)
			std::cout << cameraMatrix.at<double>(i, j) << endl;

	std::cout << "\n\n";

	for (int i = 0; i < cameraDistortionCoefficients.rows; i++)
		for (int j = 0; j < cameraDistortionCoefficients.cols; j++)
			std::cout << cameraDistortionCoefficients.at<double>(i, j) << endl;

	std::cout << "\n\n";

	std::cout
		<< "Image width = " << kinectFrameData.registeredColorFrameSize.width << endl
		<< "Image height = " << kinectFrameData.registeredColorFrameSize.height << endl
		<< "Aperture width = " << cameraApertureWidth << endl
		<< "Aperture height = " << cameraApertureHeight << endl
		<< "Aspect ratio = " << cameraAspectRatio << endl
		<< "fovX = " << cameraFoVX << endl
		<< "fovY = " << cameraFoVY << endl
		<< "focal length = " << cameraFocalLength << endl
		<< "focal point X = " << cameraFocalPoint.x << endl
		<< "focal point Y = " << cameraFocalPoint.y << endl;

	std::cout << "\n\n";

	///////////////////////////////////////////////////
	//StartCalibration(allDetectedPatternPointsWarped, chessboardSize, chessboardSquareLengthInMeters, ProjectorFrameSize, projectorMatrix);
	StartCalibration(false, chessboardCalibrationData, ProjectorFrameSize, projectorMatrix, projectorDistortionCoefficients);

	double projectorApertureWidth = 0, projectorApertureHeight = 0, projectorAspectRatio = 0;
	double projectorFoVX = 0, projectorFoVY = 0, projectorFocalLength = 0;
	Point2d projectorFocalPoint = Point2d();
	try
	{
		calibrationMatrixValues(projectorMatrix, ProjectorFrameSize, projectorApertureWidth, projectorApertureHeight, projectorFoVX, projectorFoVY, projectorFocalLength, projectorFocalPoint, projectorAspectRatio);
	}
	catch (Exception e)
	{
		std::cout << "\n" << e.msg << endl;
	}

	std::cout << "\n\n";

	for (int i = 0; i < projectorMatrix.rows; i++)
		for (int j = 0; j < projectorMatrix.cols; j++)
			std::cout << projectorMatrix.at<double>(i, j) << endl;

	std::cout << "\n\n";

	for (int i = 0; i < projectorDistortionCoefficients.rows; i++)
		for (int j = 0; j < projectorDistortionCoefficients.cols; j++)
			std::cout << projectorDistortionCoefficients.at<double>(i, j) << endl;

	std::cout << "\n\n";

	std::cout
		<< "Image width = " << ProjectorFrameSize.width << endl
		<< "Image height = " << ProjectorFrameSize.height << endl
		<< "Aperture width = " << projectorApertureWidth << endl
		<< "Aperture height = " << projectorApertureHeight << endl
		<< "Aspect ratio = " << projectorAspectRatio << endl
		<< "fovX = " << projectorFoVX << endl
		<< "fovY = " << projectorFoVY << endl
		<< "focal length = " << projectorFocalLength << endl
		<< "focal point X = " << projectorFocalPoint.x << endl
		<< "focal point Y = " << projectorFocalPoint.y << endl;

	std::cout << endl;
	std::cout << "Stereo calibration - uncalibrated" << endl;
	Mat H1, H2, inv_cameraMatrix, inv_projectorMatrix;
	Mat R1, R2, P1, P2, Q;
	Mat map1x, map1y, map2x, map2y;
	Mat imgU1, imgU2;

	vector<Point2f> cameraChessboardPoints;
	for (int i = 0; i < chessboardCalibrationData.detectedCalibrationPatternPointsForImage.size(); i++) {
		vector<Point2f> temp = chessboardCalibrationData.detectedCalibrationPatternPointsForImage[i];
		for (int j = 0; j < temp.size(); j++) {
			cameraChessboardPoints.push_back(temp[j]);
		}
	}

	vector<Point2f> projectorChessboardPoints;
	for (int i = 0; i < chessboardCalibrationData.detectedCalibrationPatternPointsWarped.size(); i++) {
		vector<Point2f> temp = chessboardCalibrationData.detectedCalibrationPatternPointsWarped[i];
		for (int j = 0; j < temp.size(); j++) {
			projectorChessboardPoints.push_back(temp[j]);
		}
	}

	Mat FundamentalMatrix = findFundamentalMat(cameraChessboardPoints, projectorChessboardPoints, FM_LMEDS);
	invert(cameraMatrix, inv_cameraMatrix);
	invert(projectorMatrix, inv_projectorMatrix);
	stereoRectifyUncalibrated(cameraChessboardPoints, projectorChessboardPoints, FundamentalMatrix, ProjectorFrameSize, H1, H2, 5.0);
	R1 = inv_cameraMatrix * H1 * cameraMatrix;
	R2 = inv_projectorMatrix * H2 * projectorMatrix;
	cv::initUndistortRectifyMap(cameraMatrix, cameraDistortionCoefficients, R1, P1, kinectFrameData.registeredColorFrameSize, CV_32FC1, map1x, map1y);
	cv::initUndistortRectifyMap(projectorMatrix, projectorDistortionCoefficients, R2, P2, ProjectorFrameSize, CV_32FC1, map2x, map2y);

	Mat registeredColorFrameRectified, projectorFrameRectified;

	while (1)
	{
		bool updatedFrames = GetLatestColorDataFromKinect(kinectFrameData) & GetLatestDepthDataFromKinect(kinectFrameData);
		UpdateDepthAndCameraSpaceMapping(kinectFrameData);
		warpPerspective(kinectFrameData.registeredColorFrame, projectorFrame, homography, projectorFrame.size());

		undistort(kinectFrameData.registeredColorFrame, kinectFrameData.registeredColorFrameUndistorted, cameraMatrix, cameraDistortionCoefficients);
		undistort(projectorFrame, projectorFrameUndistorted, projectorMatrix, projectorDistortionCoefficients);

		//warpAffine(kinectFrameData.registeredColorFrameUndistorted, registeredColorFrameRectified, R1, kinectFrameData.registeredColorFrameSize, INTER_LINEAR, BORDER_CONSTANT, Scalar());
		//warpAffine(projectorFrameUndistorted, projectorFrameRectified, R2, ProjectorFrameSize, INTER_LINEAR, BORDER_CONSTANT, Scalar());

		remap(kinectFrameData.registeredColorFrame, registeredColorFrameRectified, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT);
		warpPerspective(kinectFrameData.registeredColorFrame, projectorFrame, homography, projectorFrame.size());
		remap(projectorFrame, projectorFrameRectified, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT);

		imshow("Kinect COLOR", kinectFrameData.colorStream);																		// Show color image
		imshow("Kinect DEPTH", kinectFrameData.depthStream);																		// Show depth image
		imshow("Kinect REGISTERED COLOR", kinectFrameData.registeredColorFrame);													// Show registered color image
		imshow("Kinect REGISTERED COLOR - UNDISTORTED", kinectFrameData.registeredColorFrameUndistorted);													// Show registered color image
		imshow("PROJECTOR", projectorFrame);
		imshow("PROJECTOR - UNDISTORTED", projectorFrameUndistorted);
		imshow("Kinect REGISTERED COLOR - RECTIFIED", registeredColorFrameRectified);
		imshow("PROJECTOR - RECTIFIED", projectorFrameRectified);

		char key = cv::waitKey(30);
		if (key == 27)
		{
			std::cout << "\nExiting streams...\n";
			break;
		}
	}

	cv::waitKey(0);
	std::cout << "Stereo calibration" << endl;

	Mat R, T, E, F;
	double rms = stereoCalibrate(chessboardCalibrationData.worldCoordinatesForGeneratedCalibrationPatternPoints,
		chessboardCalibrationData.detectedCalibrationPatternPointsForImage,
		chessboardCalibrationData.detectedCalibrationPatternPointsWarped,
		cameraMatrix, cameraDistortionCoefficients,
		projectorMatrix, projectorDistortionCoefficients,
		kinectFrameData.registeredColorFrameSize,
		R, T, E, F,
		cv::CALIB_USE_INTRINSIC_GUESS | cv::CALIB_RATIONAL_MODEL | cv::CALIB_ZERO_TANGENT_DIST |
		cv::CALIB_FIX_K1 | cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3 | cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5 | cv::CALIB_FIX_K6,
		//cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));
		cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 1e-5));

	std::cout << "\nRMS error for stereo calibration = " << rms << endl;

	cv::stereoRectify(cameraMatrix, cameraDistortionCoefficients, projectorMatrix, projectorDistortionCoefficients, kinectFrameData.registeredColorFrameSize, R, T, R1, R2, P1, P2, Q);

	std::cout << "Done Rectification\n";
	std::cout << "Applying Undistort\n";

	cv::initUndistortRectifyMap(cameraMatrix, cameraDistortionCoefficients, R1, P1, kinectFrameData.registeredColorFrameSize, CV_32FC1, map1x, map1y);
	cv::initUndistortRectifyMap(projectorMatrix, projectorDistortionCoefficients, R2, P2, ProjectorFrameSize, CV_32FC1, map2x, map2y);

	std::cout << "Undistort complete\n";

	std::cout << "R1 type: " << R1.type() << endl;
	std::cout << "R2 type: " << R2.type() << endl;
	std::cout << "Map 1 x: " << map1x.cols << " x " << map1x.rows << endl;
	std::cout << "Map 1 y: " << map1y.cols << " x " << map1y.rows << endl;
	std::cout << "Map 2 x: " << map2x.cols << " x " << map2x.rows << endl;
	std::cout << "Map 2 y: " << map2y.cols << " x " << map2y.rows << endl;

	int min = INT_MAX, max = INT_MIN;
	int minX, minY, maxX, maxY;
	std::cout << endl;
	for (int i = 0; i < map1x.rows; i++) {
		for (int j = 0; j < map1x.cols; j++) {
			int value = map1x.at<int>(i, j);
			if (value < min) {
				min = value;
				minX = j;
				minY = i;
			}
			if (value > max) {
				max = value;
				maxX = j;
				maxY = i;
			}
		}
	}
	std::cout << "Map1x: min = " << min << ", @ (" << minX << ", " << minY << ") and max = " << max << ", @ (" << maxX << ", " << maxY << ")" << endl;

	int min_map1x = INT_MAX, max_map1x = INT_MIN;
	int minX_map1x, minY_map1x, maxX_map1x, maxY_map1x;
	for (int i = 0; i < map1x.rows; i++) {
		for (int j = 0; j < map1x.cols; j++) {
			int value = map1x.at<int>(i, j);
			if (value < min_map1x) {
				min_map1x = value;
				minX_map1x = j;
				minY_map1x = i;
			}
			if (value > max_map1x) {
				max_map1x = value;
				maxX_map1x = j;
				maxY_map1x = i;
			}
		}
	}
	std::cout << "Map1x: min = " << min_map1x << ", @ (" << minX_map1x << ", " << minY_map1x << ") and max = " << max_map1x << ", @ (" << maxX_map1x << ", " << maxY_map1x << ")" << endl;
	std::cout << endl;


	min = INT_MAX; max = INT_MIN;
	std::cout << endl;
	for (int i = 0; i < map1y.rows; i++) {
		for (int j = 0; j < map1y.cols; j++) {
			int value = map1y.at<int>(i, j);
			if (value < min) {
				min = value;
				minX = j;
				minY = i;
			}
			if (value > max) {
				max = value;
				maxX = j;
				maxY = i;
			}
		}
	}
	std::cout << "Map1y: min = " << min << ", @ (" << minX << ", " << minY << ") and max = " << max << ", @ (" << maxX << ", " << maxY << ")" << endl;

	int min_map1y = INT_MAX, max_map1y = INT_MIN;
	int minX_map1y, minY_map1y, maxX_map1y, maxY_map1y;
	for (int i = 0; i < map1y.rows; i++) {
		for (int j = 0; j < map1y.cols; j++) {
			int value = map1y.at<int>(i, j);
			if (value < min_map1y) {
				min_map1y = value;
				minX_map1y = j;
				minY_map1y = i;
			}
			if (value > max_map1y) {
				max_map1y = value;
				maxX_map1y = j;
				maxY_map1y = i;
			}
		}
	}
	std::cout << "Map1y: min = " << min_map1y << ", @ (" << minX_map1y << ", " << minY_map1y << ") and max = " << max_map1y << ", @ (" << maxX_map1y << ", " << maxY_map1y << ")" << endl;
	std::cout << endl;

	min = INT_MAX; max = INT_MIN;
	std::cout << endl;
	for (int i = 0; i < map2x.rows; i++) {
		for (int j = 0; j < map2x.cols; j++) {
			int value = map2x.at<int>(i, j);
			if (value < min) {
				min = value;
				minX = j;
				minY = i;
			}
			if (value > max) {
				max = value;
				maxX = j;
				maxY = i;
			}
		}
	}
	std::cout << "Map2x: min = " << min << ", @ (" << minX << ", " << minY << ") and max = " << max << ", @ (" << maxX << ", " << maxY << ")" << endl;

	int min_map2x = INT_MAX, max_map2x = INT_MIN;
	int minX_map2x, minY_map2x, maxX_map2x, maxY_map2x;
	for (int i = 0; i < map2x.rows; i++) {
		for (int j = 0; j < map2x.cols; j++) {
			int value = map2x.at<int>(i, j);
			if (value < min_map2x) {
				min_map2x = value;
				minX_map2x = j;
				minY_map2x = i;
			}
			if (value > max_map2x) {
				max_map2x = value;
				maxX_map2x = j;
				maxY_map2x = i;
			}
		}
	}
	std::cout << "Map2x: min = " << min_map2x << ", @ (" << minX_map2x << ", " << minY_map2x << ") and max = " << max_map2x << ", @ (" << maxX_map2x << ", " << maxY_map2x << ")" << endl;
	std::cout << endl;

	min = INT_MAX; max = INT_MIN;
	std::cout << endl;
	for (int i = 0; i < map2y.rows; i++) {
		for (int j = 0; j < map2y.cols; j++) {
			int value = map2y.at<int>(i, j);
			if (value < min) {
				min = value;
				minX = j;
				minY = i;
			}
			if (value > max) {
				max = value;
				maxX = j;
				maxY = i;
			}
		}
	}
	std::cout << "Map2y: min = " << min << ", @ (" << minX << ", " << minY << ") and max = " << max << ", @ (" << maxX << ", " << maxY << ")" << endl;

	int min_map2y = INT_MAX, max_map2y = INT_MIN;
	int minX_map2y, minY_map2y, maxX_map2y, maxY_map2y;
	for (int i = 0; i < map2y.rows; i++) {
		for (int j = 0; j < map2y.cols; j++) {
			int value = map2y.at<int>(i, j);
			if (value < min_map2y) {
				min_map2y = value;
				minX_map2y = j;
				minY_map2y = i;
			}
			if (value > max_map2y) {
				max_map2y = value;
				maxX_map2y = j;
				maxY_map2y = i;
			}
		}
	}
	std::cout << "Map2y: min = " << min_map2y << ", @ (" << minX_map2y << ", " << minY_map2y << ") and max = " << max_map2y << ", @ (" << maxX_map2y << ", " << maxY_map2y << ")" << endl;
	std::cout << endl;


	min = INT_MAX; max = INT_MIN;
	std::cout << endl;

	std::cout << "\nR1:\n";
	for (int i = 0; i < R1.rows; i++)
		for (int j = 0; j < R1.cols; j++)
			std::cout << R1.at<double>(i, j) << endl;

	std::cout << "\nR2:\n";
	for (int i = 0; i < R2.rows; i++)
		for (int j = 0; j < R2.cols; j++)
			std::cout << R2.at<double>(i, j) << endl;

	std::cout << endl << endl << "at 0" << endl << endl;

	for (int j = 0; j < map1x.cols; j++) {
		std::cout << map1x.at<int>(0, j) << " ";
	}
	std::cout << endl << endl;

	for (int j = 0; j < map1y.cols; j++) {
		std::cout << map1y.at<int>(0, j) << " ";
	}
	std::cout << endl << endl;

	for (int j = 0; j < map2x.cols; j++) {
		std::cout << map2x.at<int>(0, j) << " ";
	}
	std::cout << endl << endl;

	for (int j = 0; j < map2y.cols; j++) {
		std::cout << map2y.at<int>(0, j) << " ";
	}
	std::cout << endl << endl << "at half way" << endl << endl;

	cv::waitKey(0);

	for (int j = 0; j < map1x.cols; j++) {
		std::cout << map1x.at<int>(map1x.rows / 2, j) << " ";
	}
	std::cout << endl << endl;

	for (int j = 0; j < map1y.cols; j++) {
		std::cout << map1y.at<int>(map1y.rows / 2, j) << " ";
	}
	std::cout << endl << endl;

	for (int j = 0; j < map2x.cols; j++) {
		std::cout << map2x.at<int>(map2x.rows / 2, j) << " ";
	}
	std::cout << endl << endl;

	for (int j = 0; j < map2y.cols; j++) {
		std::cout << map2y.at<int>(map2y.rows / 2, j) << " ";
	}
	std::cout << endl << endl;

	cv::waitKey(0);

	/*for (int i = 0; i < map1x.rows; i++) {
	for (int j = 0; j < map1x.cols; j++) {
	map1x.at<int>(i, j) -= min_map1x;
	if (i == map1x.rows / 2)
	cout << map1x.at<int>(i, j) << " ";
	}
	}
	cout << endl;

	for (int i = 0; i < map1y.rows; i++) {
	for (int j = 0; j < map1y.cols; j++) {
	map1y.at<int>(i, j) -= min_map1y;
	if (i == map1y.rows / 2)
	cout << map1y.at<int>(i, j) << " ";
	}
	}
	cout << endl;

	for (int i = 0; i < map2x.rows; i++) {
	for (int j = 0; j < map2x.cols; j++) {
	map2x.at<int>(i, j) -= min_map2x;
	if (i == map2x.rows / 2)
	cout << map2x.at<int>(i, j) << " ";
	}
	}
	cout << endl;

	for (int i = 0; i < map2y.rows; i++) {
	for (int j = 0; j < map2y.cols; j++) {
	map2y.at<int>(i, j) -= min_map2y;
	if (i == map2y.rows / 2)
	cout << map2y.at<int>(i, j) << " ";
	}
	}
	cout << endl;*/

	cv::waitKey(0);

	int countFrame = 0;
	while (1)
	{
		bool updatedFrames = GetLatestColorDataFromKinect(kinectFrameData) & GetLatestDepthDataFromKinect(kinectFrameData);
		UpdateDepthAndCameraSpaceMapping(kinectFrameData);

		remap(kinectFrameData.registeredColorFrame, imgU1, map1x, map1y, INTER_LINEAR, BORDER_CONSTANT);
		warpPerspective(kinectFrameData.registeredColorFrame, projectorFrame, homography, projectorFrame.size());
		remap(projectorFrame, imgU2, map2x, map2y, INTER_LINEAR, BORDER_CONSTANT);

		//warpAffine(kinectFrameData.registeredColorFrame, imgU1, R1, kinectFrameData.registeredColorFrameSize);
		//warpAffine(projectorFrame, imgU2, R2, ProjectorFrameSize);

		//warpPerspective(kinectFrameData.registeredColorFrame, imgU1, R1, kinectFrameData.registeredColorFrameSize);
		//warpPerspective(projectorFrame, imgU2, R2, projectorFrame.size());

		imshow("Kinect COLOR", kinectFrameData.colorStream);																		// Show color image
		imshow("Kinect DEPTH", kinectFrameData.depthStream);																		// Show depth image
																																	//imshow("Kinect REGISTERED COLOR", kinectFrameData.registeredColorFrame);													// Show registered color image
																																	//imshow("PROJECTOR", projectorFrame);
		imshow("Kinect REGISTERED COLOR", imgU1);													// Show registered color image
		imshow("PROJECTOR", imgU2);

		if (updatedFrames)
			std::cout << "Frame number: " << countFrame++ << " updated " << imgU1.cols << " x " << imgU1.rows << " " << imgU2.cols << " x " << imgU2.rows << endl;

		char key = cv::waitKey(30);
		if (key == 27)
		{
			std::cout << "\nExiting streams...\n";
			break;
		}
	}

	kinectFrameData.sensor->Close();
	kinectFrameData.sensor->Release();

	std::cout << "\nPress any key to exit...";
	cv::waitKey(0);
	cv::destroyAllWindows();

	////////////// stereo rectify

	return 0;
}