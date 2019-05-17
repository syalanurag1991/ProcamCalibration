#include "CalibrationHandler.h"

CalibrationHandler::CalibrationHandler(cv::Size cameraFrameSize, cv::Size primaryScreenResolution, cv::Size projectorFrameSize, cv::Size projectorAspectRatio, cv::Size chessboardSize, UINT minChessboardSquareLengthInPixels, bool downsampleFrame, bool fastCheck, bool manualMode, bool showResults, bool showLogs) {
	this->cameraFrameSize = cameraFrameSize;
	this->primaryScreenResolution = primaryScreenResolution;
	this->projectorFrameSize = projectorFrameSize;
	this->projectorAspectRatio = projectorAspectRatio;
	this->calibrationPatternSize = chessboardSize;
	this->minChessboardSquareLengthInPixels = minChessboardSquareLengthInPixels;
	this->downsampleFrame = downsampleFrame;
	this->fastCheck = fastCheck;
	this->manualMode = manualMode;
	this->showResults = showResults;
	this->showLogs = showLogs;

	this->CalculateAndSetChessboardSquareEdgeLengthInpixels();
	this->minOffset = cv::Size(1, 1);
	this->maxOffset = cv::Size(projectorFrameSize.width / chessboardSquareLengthInPixels - chessboardSize.width - 1, projectorFrameSize.height / chessboardSquareLengthInPixels - chessboardSize.height - 1);

	this->projectorFrame = cv::Mat(projectorFrameSize.height, projectorFrameSize.width, CV_8UC4);
	this->undistortedProjectorFrame = cv::Mat(projectorFrameSize.height, projectorFrameSize.width, CV_8UC4);
	this->undistortedProjectorFrame = cv::Mat(projectorFrameSize.height, projectorFrameSize.width, CV_8UC4);
}

void CalibrationHandler::CalculateAndSetChessboardSquareEdgeLengthInpixels() {
	if (this->projectorAspectRatio == cv::Size(4, 3))
	{
		std::cout << "\nProjector aspect ratio = 4:3" << std::endl;
		this->chessboardSquareLengthInPixels = this->projectorFrameSize.width / 4;
	}
	else if (this->projectorAspectRatio == cv::Size(16, 9))
	{
		std::cout << "\nProjector aspect ratio = 16:9" << std::endl;
		this->chessboardSquareLengthInPixels = this->projectorFrameSize.width / 16;
	}

	int scaleBy = (int)(chessboardSquareLengthInPixels / (float)this->minChessboardSquareLengthInPixels);
	while (scaleBy >= 2)
	{
		this->chessboardSquareLengthInPixels >>= 1;
		scaleBy = (int)(chessboardSquareLengthInPixels / (float)(float)this->minChessboardSquareLengthInPixels);
	}
	return;
}

void CalibrationHandler::CreateWindowsForDisplayingFrames(bool showUndistortedProjectorFrame, bool showrectifiedProjectorFrame) {
	cv::namedWindow("PROJECTOR", cv::WINDOW_NORMAL);
	cv::moveWindow("PROJECTOR", this->primaryScreenResolution.width, 0);
	cv::setWindowProperty("PROJECTOR", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

	this->showUndistortedProjectorFrame = showUndistortedProjectorFrame;
	this->showrectifiedProjectorFrame = showrectifiedProjectorFrame;

	if (showUndistortedProjectorFrame)
	{
		cv::namedWindow("PROJECTOR - UNDISTORTED", cv::WINDOW_NORMAL);
		cv::moveWindow("PROJECTOR - UNDISTORTED", this->primaryScreenResolution.width, 0);
		cv::setWindowProperty("PROJECTOR - UNDISTORTED", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
	}

	if (showrectifiedProjectorFrame)
	{
		cv::namedWindow("PROJECTOR - RECTIFIED", cv::WINDOW_NORMAL);
		cv::moveWindow("PROJECTOR - RECTIFIED", this->primaryScreenResolution.width, 0);
		cv::setWindowProperty("PROJECTOR - RECTIFIED", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
	}
	return;
}

void CalibrationHandler::DisplayFrames() {
	imshow("PROJECTOR", this->projectorFrame);
	if(this->showUndistortedProjectorFrame)
		imshow("PROJECTOR - UNDISTORTED", this->undistortedProjectorFrame);
	if(this->showrectifiedProjectorFrame)
	imshow("PROJECTOR - RECTIFIED", this->rectifiedProjectorFrame);
}

void CalibrationHandler::DisplayFrames(bool displayProjectorFrame, bool displayUndistortedProjectorFrame, bool displayRectifiedProjectorFrame) {
	if (displayProjectorFrame)
		imshow("PROJECTOR", this->projectorFrame);
	if (displayUndistortedProjectorFrame)
		imshow("PROJECTOR - UNDISTORTED", this->undistortedProjectorFrame);
	if (displayRectifiedProjectorFrame)
		imshow("PROJECTOR - RECTIFIED", this->rectifiedProjectorFrame);
}

void CalibrationHandler::CreateChessBoardPatternImages()
{
	// Setup chessboard colors and dimensions
	cv::Scalar colorBlack = cv::Scalar(0, 0, 0, 255);
	cv::Scalar colorWhite = cv::Scalar(255, 255, 255, 255);
	cv::Scalar colorRed = cv::Scalar(255, 0, 0, 255);
	cv::Scalar colorBlue = cv::Scalar(0, 0, 255, 255);
	UINT chessboardWidth = (this->calibrationPatternSize.width + 1) * this->chessboardSquareLengthInPixels;
	UINT chessboardHeight = (this->calibrationPatternSize.height + 1) * this->chessboardSquareLengthInPixels;

	int cornerCount = 1;
	for (int yOffset = this->minOffset.height; yOffset < this->maxOffset.height; yOffset++) {
		for (int xOffset = this->minOffset.width; xOffset < this->maxOffset.width; xOffset++) {
			std::cout << "\n Offset = (" << xOffset << ", " << yOffset << ")" << std::endl << "Corner locations:" << std::endl;

			cv::Mat searchPattern = cv::Mat(this->projectorFrameSize.height, this->projectorFrameSize.width, CV_8UC4, colorWhite);
			cv::Mat testPattern = cv::Mat(chessboardHeight, chessboardWidth, CV_8UC4, colorWhite);

			bool storeInReverseOrder = false;
			if ((xOffset + yOffset) % 2 != 0)
				storeInReverseOrder = true;

			std::vector<cv::Point2f> newCalibrationPatternPoints;
			for (int y = yOffset, y_searchPattern = 0; y <= yOffset + this->calibrationPatternSize.height; y++, y_searchPattern += this->chessboardSquareLengthInPixels) {
				for (int x = xOffset, x_searchPattern = 0; x <= xOffset + this->calibrationPatternSize.width; x++, x_searchPattern += this->chessboardSquareLengthInPixels) {
					int xPosition = x * this->chessboardSquareLengthInPixels;
					int yPosition = y * this->chessboardSquareLengthInPixels;
					std::cout << cornerCount++ << " @ (" << xPosition << ", " << yPosition << ")" << std::endl;

					// don't include points at boundaries
					if (x != xOffset && y != yOffset)
					{
						cv::Point2f chessboardPoint = cv::Point2f((float)xPosition, (float)yPosition);
						if (!storeInReverseOrder)
							newCalibrationPatternPoints.push_back(chessboardPoint);
						else
							newCalibrationPatternPoints.insert(newCalibrationPatternPoints.begin(), chessboardPoint);
					}

					cv::Rect chessboardBlock(xPosition, yPosition, this->chessboardSquareLengthInPixels, this->chessboardSquareLengthInPixels);
					cv::Rect chessboardBlock_searchPattern(x_searchPattern, y_searchPattern, this->chessboardSquareLengthInPixels, this->chessboardSquareLengthInPixels);
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
			this->collectionOfCalibrationPatterns.push_back(searchPattern);
			this->generatedCalibrationPatternPoints.push_back(newCalibrationPatternPoints);

			imshow("PROJECTOR", searchPattern);
			cv::waitKey((this->delay) / 4);														// just reducing the delay, nothing else
		}
	}
	return;
}

void CalibrationHandler::CreateWorldCoordinatesForChessBoardCornerPositions()
{
	std::cout << "\n\nCreating world coordinates: \n\n";
	int pointNumber = 0;
	int collectionNumber = 0;
	for (int yOffset = this->minOffset.height; yOffset < this->maxOffset.height; yOffset++)
	{
		for (int xOffset = this->minOffset.width; xOffset < this->maxOffset.width; xOffset++)
		{
			if (this->worldPointsToBeConsidered[collectionNumber]) {
				bool storeInReverseOrder = false;
				if ((xOffset + yOffset) % 2 != 0)
					storeInReverseOrder = true;

				std::vector<cv::Point3f> newWorldCoordinatesForCalibrationPatternPoints;

				// don't include points at boundaries, we need (calibrationPatternSize.width - 1) x (calibrationPatternSize.height - 1) points
				for (int y = yOffset; y < yOffset + this->calibrationPatternSize.height; y++) {
					for (int x = xOffset; x < xOffset + this->calibrationPatternSize.width; x++) {
						float xPosition = (float)x * this->chessboardSquareLengthInMeters;
						float yPosition = (float)y * this->chessboardSquareLengthInMeters;
						std::cout << "Collection # " << collectionNumber << " -> point # " << pointNumber++ << " @ (" << x << ", " << y << ") = " << xPosition << ", " << yPosition << std::endl;
						cv::Point3f chessboardPoint = cv::Point3f((float)xPosition, (float)yPosition, 0.0f);
						if (!storeInReverseOrder)
							newWorldCoordinatesForCalibrationPatternPoints.push_back(chessboardPoint);
						else
							newWorldCoordinatesForCalibrationPatternPoints.insert(newWorldCoordinatesForCalibrationPatternPoints.begin(), chessboardPoint);
					}
				}
				std::cout << std::endl;
				this->worldCoordinatesForGeneratedCalibrationPatternPoints.push_back(newWorldCoordinatesForCalibrationPatternPoints);
			}
			collectionNumber++;
		}
	}
	return;
}

bool CalibrationHandler::GetCalibrationPatternPointsInCurrentFrame(KinectWrapper& in_out_kinectFrameData, UINT frameNumber)
{
	std::vector<cv::Point2f> foundPoints, foundPoints_upsampled;
	cv::Mat in_out_frame_downsampled;

	if (this->downsampleFrame)
	{
		cv::resize(in_out_kinectFrameData.registeredColorFrame, in_out_kinectFrameData.downsampledRegisteredColorFrame, cv::Size(), 1 / (float)in_out_kinectFrameData.downsampleSize, 1 / (float)in_out_kinectFrameData.downsampleSize, cv::INTER_AREA);
		cv::cvtColor(in_out_kinectFrameData.downsampledRegisteredColorFrame, this->grayRegisteredColorFrame, cv::COLOR_BGR2GRAY);
	}
	else
		cv::cvtColor(in_out_kinectFrameData.registeredColorFrame, this->grayRegisteredColorFrame, cv::COLOR_BGR2GRAY);

	bool found = false;																										// if pattern is detected by opencv function
	try
	{
		if (this->fastCheck)
		{
			found = findChessboardCorners(this->grayRegisteredColorFrame, this->calibrationPatternSize, foundPoints, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE | CV_CALIB_CB_FILTER_QUADS);
			if (foundPoints.size() > 0)
				found = find4QuadCornerSubpix(this->grayRegisteredColorFrame, foundPoints, cv::Size(50, 50));
		}
		else
			found = findChessboardCorners(this->grayRegisteredColorFrame, this->calibrationPatternSize, foundPoints, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | CV_CALIB_CB_FILTER_QUADS);

		if (found)
		{
			int pointIndex = 0;
			float sumX = 0;
			float sumY = 0;
			float squareSumX = 0;
			float squareSumY = 0;

			for (int i = 0; i < this->calibrationPatternSize.height; i++)
			{
				float previousX = 0;
				for (int j = 0; j < this->calibrationPatternSize.width; j++)
				{
					float foundPointX = foundPoints[pointIndex].x;
					float foundPointY = foundPoints[pointIndex].y;
					if (this->downsampleFrame) {
						foundPointX = foundPoints[pointIndex].x * (float)in_out_kinectFrameData.downsampleSize;
						foundPointY = foundPoints[pointIndex].y * (float)in_out_kinectFrameData.downsampleSize;
					}

					int foundPointX_int = (int)foundPoints[pointIndex].x;
					int foundPointY_int = (int)foundPoints[pointIndex].y;
					int depthSpaceIndexForCurrentFoundPoint = foundPointY_int * in_out_kinectFrameData.depthFrameWidth + foundPointX_int;
					float differenceX = 0, differenceY = 0;

					if (j != 0)
					{
						int previousFoundPointX_int = (int)foundPoints[pointIndex - 1].x;
						int depthSpaceIndexForPreviousFoundPointInX = foundPointY_int * in_out_kinectFrameData.depthFrameWidth + previousFoundPointX_int;
						differenceX = std::abs(in_out_kinectFrameData.cameraSpacePoints[depthSpaceIndexForCurrentFoundPoint].X - in_out_kinectFrameData.cameraSpacePoints[depthSpaceIndexForPreviousFoundPointInX].X);
						sumX += differenceX;
						squareSumX += differenceX * differenceX;
					}
					previousX = in_out_kinectFrameData.cameraSpacePoints[depthSpaceIndexForCurrentFoundPoint].X;

					if (i != 0)
					{
						int previousFoundPointY_int = (int)foundPoints[pointIndex - this->calibrationPatternSize.width].y;
						int depthSpaceIndexForPreviousFoundPointInY = previousFoundPointY_int * in_out_kinectFrameData.depthFrameWidth + foundPointX_int;
						differenceY = std::abs(in_out_kinectFrameData.cameraSpacePoints[depthSpaceIndexForCurrentFoundPoint].Y - in_out_kinectFrameData.cameraSpacePoints[depthSpaceIndexForPreviousFoundPointInY].Y);
						sumY += differenceY;
						squareSumY += differenceY * differenceY;
					}

					if (this->showLogs)
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
							<< std::endl;
					}
					pointIndex++;
				}
			}

			this->measurements.meanLengthX = sumX / ((this->calibrationPatternSize.width - 1) * this->calibrationPatternSize.height);
			this->measurements.meanLengthY = sumY / (this->calibrationPatternSize.width * (this->calibrationPatternSize.height - 1));
			float expectationOfSquareOfDifferenceX = squareSumX / ((this->calibrationPatternSize.width - 1) * this->calibrationPatternSize.height);
			float expectationOfSquareOfDifferenceY = squareSumY / (this->calibrationPatternSize.width * (this->calibrationPatternSize.height - 1));
			float squareMeanMeasureX = this->measurements.meanLengthX * this->measurements.meanLengthX;
			float squareMeanMeasureY = this->measurements.meanLengthY * this->measurements.meanLengthY;
			float varianceOfDifferenceX = expectationOfSquareOfDifferenceX - squareMeanMeasureX;
			float varianceOfDifferenceY = expectationOfSquareOfDifferenceY - squareMeanMeasureY;
			this->measurements.stdDevX = sqrt(varianceOfDifferenceX);
			this->measurements.stdDevY = sqrt(varianceOfDifferenceY);
			this->measurements.rmsLength = sqrt(squareMeanMeasureX + squareMeanMeasureY) / sqrt(2);
			this->measurements.stdDev = sqrt(this->measurements.stdDevX * this->measurements.stdDevX + this->measurements.stdDevY * this->measurements.stdDevY) / sqrt(2);


			if (this->showResults)
			{
				drawChessboardCorners(in_out_kinectFrameData.registeredColorFrame, this->calibrationPatternSize, foundPoints, found);
				std::cout << std::endl
					<< "\nsumX = " << sumX << std::endl
					<< "sumY = " << sumY << std::endl
					<< "squareSumX = " << squareSumX << std::endl
					<< "squareSumY = " << squareSumY << std::endl
					<< "# of points = " << foundPoints.size() << std::endl
					<< "meanMeasureX = " << this->measurements.meanLengthX << std::endl
					<< "meanMeasureY = " << this->measurements.meanLengthY << std::endl
					<< "meanMeasure = " << this->measurements.rmsLength << std::endl
					<< "standardDeviationX = " << this->measurements.stdDevX << std::endl
					<< "standardDeviationY = " << this->measurements.stdDevY << std::endl
					<< "standardDeviation = " << this->measurements.stdDev << std::endl;
			}

			if (this->manualMode)
			{
				std::cout << "\nPattern detected ... press 'c' to capture or 'r' to reject, frame # " << frameNumber << "\n";
				char key = cv::waitKey(0);
				if (key == 'r')
					return false;
				else if (key == 'c')
				{
					this->detectedCalibrationPatternPointsInCameraFrame.push_back(foundPoints);
					return true;
				}
				return false;
			}
			else
			{
				std::cout << "\nPattern detected ... standard deviation = " << this->measurements.stdDev << ", frame # " << frameNumber << "\n";
				cv::waitKey(this->delay);
				if (this->measurements.stdDev < 0.01)
				{
					this->detectedCalibrationPatternPointsInCameraFrame.push_back(foundPoints);
					return true;
				}
				return false;
			}

		}
		return false;
	}
	catch (std::exception& e)
	{
		std::cout << "\n" << e.what() << std::endl;
		return false;
	}

	return false;
}

void CalibrationHandler::ApplyInverseHomographyAndWarpImage(cv::Mat& in_source, cv::Mat& out_destination)
{
	double denominator = 1;
	int newX = 0, newY = 0;
	for (int y = 0; y < out_destination.rows; y++)
	{
		for (int x = 0; x < out_destination.cols; x++)
		{
			denominator = this->inverseHomography.at<double>(2, 0) * x + this->inverseHomography.at<double>(2, 1) * y + this->inverseHomography.at<double>(2, 2);
			newX = (int)((this->inverseHomography.at<double>(0, 0) * x + this->inverseHomography.at<double>(0, 1) * y + this->inverseHomography.at<double>(0, 2)) / denominator);
			newY = (int)((this->inverseHomography.at<double>(1, 0) * x + this->inverseHomography.at<double>(1, 1) * y + this->inverseHomography.at<double>(1, 2)) / denominator);

			if (newX >= 0 && newX < in_source.cols && newY >= 0 && newY < in_source.rows)
				out_destination.at<cv::Vec4b>(y, x) = in_source.at<cv::Vec4b>(newY, newX);
		}
	}
	return;
}

void CalibrationHandler::ApplyHomographyAndWarpPoints(std::vector<std::vector<cv::Point2f>>& in_collectionOfSourcePoints, std::vector<std::vector<cv::Point2f>>& out_collectionOfDestinationPoints)
{
	double x, y, denominator = 1.0, newX = 0, newY = 0;
	std::vector<cv::Point2f> latestSourcePoints = in_collectionOfSourcePoints[in_collectionOfSourcePoints.size() - 1];
	std::vector<cv::Point2f> latestDestinationPoints;
	for (int i = 0; i < latestSourcePoints.size(); i++)
	{
		x = latestSourcePoints[i].x;
		y = latestSourcePoints[i].y;
		denominator = (this->homography.at<double>(2, 0) * x) + (this->homography.at<double>(2, 1) * y) + this->homography.at<double>(2, 2);
		newX = ((this->homography.at<double>(0, 0) * x) + (this->homography.at<double>(0, 1) * y) + this->homography.at<double>(0, 2)) / denominator;
		newY = ((this->homography.at<double>(1, 0) * x) + (this->homography.at<double>(1, 1) * y) + this->homography.at<double>(1, 2)) / denominator;
		cv::Point2f newDestinationPoint = cv::Point2f((float)newX, (float)newY);
		latestDestinationPoints.push_back(newDestinationPoint);
	}

	out_collectionOfDestinationPoints.push_back(latestDestinationPoints);
	std::cout << "\n\nsize of latest destination points collection = " << out_collectionOfDestinationPoints.size() << " x " << latestDestinationPoints.size() << "\n\n";
	return;
}

void CalibrationHandler::CollectCalibrationPatternPointsFromProjector(KinectWrapper& in_out_kinectFrameData)
{
	int countTotalFrames = 0;
	int key = -1;
	bool firstAttempt = true;
	bool patternCaptured = false;
	bool gotLatestData = false;

	UINT currentAttempt = 0;
	for (int i = 0; i < this->collectionOfCalibrationPatterns.size();)
	{
		current = clock();

		std::cout << "\n\nFrame count: " << countTotalFrames << std::endl;
		imshow("PROJECTOR", this->collectionOfCalibrationPatterns[i]);														// Show chessboard image
		cv::waitKey(this->delay);

		gotLatestData = in_out_kinectFrameData.GetLatestColorDataFromKinect() & in_out_kinectFrameData.GetLatestDepthDataFromKinect();

		if (gotLatestData)
		{
			in_out_kinectFrameData.UpdateDepthAndCameraSpaceMapping();

			patternCaptured = this->GetCalibrationPatternPointsInCurrentFrame(in_out_kinectFrameData, countTotalFrames);
			drawChessboardCorners(this->collectionOfCalibrationPatterns[i], this->calibrationPatternSize, this->generatedCalibrationPatternPoints[i], true);
			imshow("PROJECTOR", this->collectionOfCalibrationPatterns[i]);													// Show chessboard image

			cv::waitKey(this->delay);
			std::vector<cv::Point2f> sourcePoints, destinationPoints;

			if (patternCaptured && this->detectedCalibrationPatternPointsInCameraFrame.size() > 0)
			{
				std::cout << "\n # of points detected = " << this->detectedCalibrationPatternPointsInCameraFrame.size() << " x " << this->detectedCalibrationPatternPointsInCameraFrame[this->detectedCalibrationPatternPointsInCameraFrame.size() - 1].size() << ", # of points projected = " << this->generatedCalibrationPatternPoints[i].size() << std::endl;
				if (this->detectedCalibrationPatternPointsInCameraFrame[this->detectedCalibrationPatternPointsInCameraFrame.size() - 1].size() > 0 && this->generatedCalibrationPatternPoints[i].size() > 0 &&
					this->detectedCalibrationPatternPointsInCameraFrame[this->detectedCalibrationPatternPointsInCameraFrame.size() - 1].size() == this->generatedCalibrationPatternPoints[i].size())
				{
					sourcePoints.insert(sourcePoints.end(), this->detectedCalibrationPatternPointsInCameraFrame[this->detectedCalibrationPatternPointsInCameraFrame.size() - 1].begin(), this->detectedCalibrationPatternPointsInCameraFrame[this->detectedCalibrationPatternPointsInCameraFrame.size() - 1].end());
					destinationPoints.insert(destinationPoints.end(), this->generatedCalibrationPatternPoints[i].begin(), this->generatedCalibrationPatternPoints[i].end());
					for (int i = 0; i < sourcePoints.size(); i++)
					{
						std::cout << std::setw(4) << std::setprecision(4) << "#" << i << " (x, y) = " << sourcePoints[i].x << ", " << sourcePoints[i].y << " --> " << destinationPoints[i].x << ", " << destinationPoints[i].y << std::endl;
					}

					this->homography = cv::findHomography(sourcePoints, destinationPoints, CV_RANSAC, 0.5);
					cv::invert(this->homography, this->inverseHomography);
					ApplyInverseHomographyAndWarpImage(in_out_kinectFrameData.registeredColorFrame, this->projectorFrame);
					ApplyHomographyAndWarpPoints(this->detectedCalibrationPatternPointsInCameraFrame, this->detectedCalibrationPatternPointsInProjectorFrame);
					int radius = 5, thickness = -1;
					for (int i = 0; i < this->detectedCalibrationPatternPointsInProjectorFrame[this->detectedCalibrationPatternPointsInProjectorFrame.size() - 1].size(); i++)
						cv::circle(this->projectorFrame, this->detectedCalibrationPatternPointsInProjectorFrame[this->detectedCalibrationPatternPointsInProjectorFrame.size() - 1][i], radius, cv::Scalar(255, 0, 255, 255), thickness, cv::LINE_8, 0);

					if (this->showLogs) {
						std::cout << "points generated (destination)            = " << destinationPoints.size() << std::endl;
						std::cout << "points found (source)                     = " << sourcePoints.size() << std::endl;
					}

					imshow("PROJECTOR", this->projectorFrame);																			// show chessboard image
					if (this->manualMode)
						cv::waitKey(0);
					else
						cv::waitKey(2 * this->delay);																								// delay for view
				}
			}

			in_out_kinectFrameData.insetImage1 = in_out_kinectFrameData.allKinectFrames(cv::Rect(106, 32, 448, 252));
			in_out_kinectFrameData.insetImage2 = in_out_kinectFrameData.allKinectFrames(cv::Rect(106, 316, 448, 371));
			in_out_kinectFrameData.insetImage3 = in_out_kinectFrameData.allKinectFrames(cv::Rect(660, 148, in_out_kinectFrameData.registeredFrameWidth, in_out_kinectFrameData.registeredFrameHeight));

			cv::resize(in_out_kinectFrameData.colorFrame, in_out_kinectFrameData.insetImage1, cv::Size(448, 252), 0, 0, cv::INTER_LINEAR);
			cv::resize(in_out_kinectFrameData.depthFrame, in_out_kinectFrameData.insetImage2, cv::Size(448, 371), 0, 0, cv::INTER_LINEAR);
			in_out_kinectFrameData.registeredColorFrame.copyTo(in_out_kinectFrameData.insetImage3);

			cv::imshow("Kinect COLOR DEPTH and REGISTERED frames", in_out_kinectFrameData.allKinectFrames);
			
			this->chessboardSquareLengthInMeters = this->measurements.rmsLength;

			delta = clock() - current;
			fpsOfDetection = CLOCKS_PER_SEC / (float)delta;
			countTotalFrames++;

			if (this->showLogs) {
				std::cout << std::setw(2) << std::setprecision(4)
					<< "\nFPS = " << fpsOfDetection
					<< "\t Pattern # " << i << " captured = " << patternCaptured << " attempt = " << currentAttempt
					<< "\t # of correspondences = " << this->detectedCalibrationPatternPointsInCameraFrame.size()
					<< "\t key = " << key
					<< std::endl;
			}

			if (patternCaptured) {
				this->worldPointsToBeConsidered.push_back(true);
				i++;
				currentAttempt = 0;
			}
			else
				currentAttempt++;

			if (currentAttempt > this->maxCaptureAttempts) {
				this->worldPointsToBeConsidered.push_back(false);
				i++;
				currentAttempt = 0;
			}
		}
	}

	//invert(this->homography, this->inverseHomography);
	return;
}

void CalibrationHandler::StartCalibration(bool isCamera)
{
	if (this->showLogs) {
		std::cout << "Calibration started ..." << std::endl;
		std::cout << "Length of square in meters = " << this->chessboardSquareLengthInMeters << std::endl;
		std::cout << "Found points vector size = " << this->detectedCalibrationPatternPointsInCameraFrame.size() << " x " << this->detectedCalibrationPatternPointsInCameraFrame[0].size() << std::endl;
		std::cout << "World coord. vector size = " << this->worldCoordinatesForGeneratedCalibrationPatternPoints.size() << " x " << this->worldCoordinatesForGeneratedCalibrationPatternPoints[0].size() << "\n\n";
	}

	if (isCamera) {
		std::cout << "\nCalibrating camera ...\n";
		double rms = cv::calibrateCamera(this->worldCoordinatesForGeneratedCalibrationPatternPoints, this->detectedCalibrationPatternPointsInCameraFrame, this->cameraFrameSize, this->cameraCalibrationMatrix, this->cameraDistortionCoefficients, this->cameraRVectors, this->cameraTVectors);
		std::cout << "RMS error for calibration = " << rms << std::endl;

		try
		{
			cv::calibrationMatrixValues(this->cameraCalibrationMatrix, this->cameraFrameSize, this->cameraApertureWidth, this->cameraApertureHeight, this->cameraFoVX, this->cameraFoVY, this->cameraFocalLength, this->cameraFocalPoint, this->cameraFocalLengthAspectRatio);
			std::cout << "\n\n"
				<< "Image size = " << this->cameraFrameSize.width << " x " << this->cameraFrameSize.height << std::endl
				<< "Aperture size = " << this->cameraApertureWidth << " x " << this->cameraApertureHeight << std::endl
				<< "Aspect ratio = " << this->cameraFocalLengthAspectRatio << std::endl
				<< "Focal length = " << this->cameraFocalLength << std::endl
				<< "FoV (x, y) = " << this->cameraFoVX << ", " << this->cameraFoVY << std::endl
				<< "Focal point (x, y) = " << this->cameraFocalPoint.x << ", " << this->cameraFocalPoint.y << std::endl
				<< "\n\n";
		}
		catch (std::exception& e)
		{
			std::cout << "\n" << e.what() << std::endl;
		}
	}
	else
	{
		std::cout << "\nCalibrating projector ...\n";
		double rms = cv::calibrateCamera(this->worldCoordinatesForGeneratedCalibrationPatternPoints, this->detectedCalibrationPatternPointsInProjectorFrame, this->projectorFrameSize, this->projectorCalibrationMatrix, this->projectorDistortionCoefficients, this->projectorRVectors, this->projectorTVectors);
		std::cout << "RMS error for calibration = " << rms << std::endl;

		try
		{
			cv::calibrationMatrixValues(this->projectorCalibrationMatrix, this->projectorFrameSize, this->projectorApertureWidth, this->projectorApertureHeight, this->projectorFoVX, this->projectorFoVY, this->projectorFocalLength, this->projectorFocalPoint, this->projectorFocalLengthAspectRatio);
			std::cout << "\n\n"
				<< "Image size = " << this->projectorFrameSize.width << " x " << this->projectorFrameSize.width << std::endl
				<< "Aperture size = " << this->projectorApertureWidth << " x " << this->projectorApertureHeight << std::endl
				<< "Aspect ratio = " << this->projectorFocalLengthAspectRatio << std::endl
				<< "Focal length = " << this->projectorFocalLength << std::endl
				<< "FoV (x, y) = " << this->projectorFoVX << ", " << this->projectorFoVY << std::endl
				<< "Focal point (x, y) = " << this->projectorFocalPoint.x << ", " << this->projectorFocalPoint.y << std::endl
				<< "\n\n";
		}
		catch (std::exception& e)
		{
			std::cout << "\n" << e.what() << std::endl;
		}
	}
}

void CalibrationHandler::StartProcamCalibration()
{
	std::cout << "Stereo calibration (uncalibrated cameras) started ..." << std::endl;

	std::vector<cv::Point2f> cameraChessboardPoints;
	for (int i = 0; i < this->detectedCalibrationPatternPointsInCameraFrame.size(); i++) {
		std::vector<cv::Point2f> temp = this->detectedCalibrationPatternPointsInCameraFrame[i];
		for (int j = 0; j < temp.size(); j++) {
			cameraChessboardPoints.push_back(temp[j]);
		}
	}

	std::vector<cv::Point2f> projectorChessboardPoints;
	for (int i = 0; i < this->detectedCalibrationPatternPointsInProjectorFrame.size(); i++) {
		std::vector<cv::Point2f> temp = this->detectedCalibrationPatternPointsInProjectorFrame[i];
		for (int j = 0; j < temp.size(); j++) {
			projectorChessboardPoints.push_back(temp[j]);
		}
	}

	cv::Mat FundamentalMatrix = findFundamentalMat(cameraChessboardPoints, projectorChessboardPoints, cv::FM_LMEDS);
	cv::invert(this->cameraCalibrationMatrix, this->inverseCameraCalibrationMatrix);
	cv::invert(this->projectorCalibrationMatrix, this->inverseProjectorCalibrationMatrix);
	stereoRectifyUncalibrated(cameraChessboardPoints, projectorChessboardPoints, FundamentalMatrix, projectorFrameSize, H1, H2, 5.0);
	R1 = this->inverseCameraCalibrationMatrix * H1 * this->cameraCalibrationMatrix;
	R2 = this->inverseProjectorCalibrationMatrix * H2 * this->projectorCalibrationMatrix;
	cv::initUndistortRectifyMap(this->cameraCalibrationMatrix, this->cameraDistortionCoefficients, R1, P1, this->cameraFrameSize, CV_32FC1, map1x, map1y);
	cv::initUndistortRectifyMap(this->projectorCalibrationMatrix, this->projectorDistortionCoefficients, R2, P2, projectorFrameSize, CV_32FC1, map2x, map2y);
}

CalibrationHandler::~CalibrationHandler()
{
}
