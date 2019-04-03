#include <vector>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

float calibrationSquareDimension = 0.019f; // meters
Size chessboardDimensions = Size(6, 9);

int fps = 30;

void CreateKnownBoardPositions(Size boardSize, float squareEdgeLength, vector<Point3f>& corners)
{
	for (int i=0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			corners.push_back(Point3f(j * squareEdgeLength, i * squareEdgeLength, 0.0f));
}

void GetChessBoardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults = false)
{
	for (vector<Mat>::iterator it = images.begin(); it != images.end(); it++)
	{
		vector<Point2f> cornerPointsBuffer;
		// change Size(9, 6) to Size(6, 9) if errors
		bool found = findChessboardCorners(*it, Size(9, 6), cornerPointsBuffer, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		if (found)
			allFoundCorners.push_back(cornerPointsBuffer);

		if (showResults)
		{
			drawChessboardCorners(*it, Size(9, 6), cornerPointsBuffer, found);
			imshow("Identified corners", *it);
			waitKey(0);
		}
	}
}

int Process()
{
	Mat frame;
	Mat drawToFrame;

	Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
	Mat distanceCoefficients;
	vector<Mat> savedFrames;

	vector<vector<Point2f>> markerCorners;
	VideoCapture vid(0);

	if (!vid.isOpened())
		return -1;

	namedWindow("Webcam", CV_WINDOW_AUTOSIZE);

	while (1)
	{
		if (!vid.read(frame))
			break;

		vector<Vec2f> foundPoints;
		bool found = findChessboardCorners(frame, chessboardDimensions, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
		frame.copyTo(drawToFrame);

		drawChessboardCorners(drawToFrame, chessboardDimensions, foundPoints, found);

		if (found)
			imshow("Webcam", drawToFrame);
		else
			imshow("Webcam", frame);

		int key = waitKey(1000/fps);
	}

	return 0;
}