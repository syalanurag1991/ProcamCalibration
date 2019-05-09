#include "KinectWrapper.h"

KinectWrapper::KinectWrapper(UINT colorFrameWidth, UINT colorFrameHeight, UINT depthFrameWidth, UINT depthFrameHeight, UINT bpp, UINT downsampleSize)
{
	this->colorFrameWidth = colorFrameWidth;
	this->colorFrameHeight = colorFrameHeight;
	this->depthFrameWidth = depthFrameWidth;
	this->depthFrameHeight = depthFrameHeight;
	this->registeredFrameWidth = depthFrameWidth;
	this->registeredFrameHeight = depthFrameHeight;
	this->downsampleSize = downsampleSize;
	this->BPP = bpp;

	colorFrameDataLength = colorFrameWidth * colorFrameHeight * bpp;
	depthFrameDataLength = depthFrameWidth * depthFrameHeight;
	registeredColorFrameDataLength = depthFrameDataLength * bpp;

	colorData = new BYTE[colorFrameDataLength];
	vizDepthData = new BYTE[registeredColorFrameDataLength];
	rawDepthData = new USHORT[depthFrameDataLength];
	registeredColorData = new BYTE[registeredColorFrameDataLength];

	colorStream = cv::Mat(colorFrameHeight, colorFrameWidth, CV_8UC4, &colorData[0]);
	depthStream = cv::Mat(depthFrameHeight, depthFrameWidth, CV_8UC4, &vizDepthData[0]);
	registeredColorFrame = cv::Mat(depthFrameHeight, depthFrameWidth, CV_8UC4, &registeredColorData[0]);

	colorFrameSize = cv::Size(colorFrameWidth, colorFrameHeight);
	depthFrameSize = cv::Size(depthFrameWidth, depthFrameHeight);
	registeredColorFrameSize = cv::Size(depthFrameWidth, depthFrameHeight);

	colorSpacePoints = new ColorSpacePoint[depthFrameDataLength];
	cameraSpacePoints = new CameraSpacePoint[depthFrameDataLength];
}

bool KinectWrapper::Initialize() {
	if (FAILED(GetDefaultKinectSensor(&(this->sensor)))) {
		return false;
	}
	if (this->sensor) {
		this->sensor->Open();
		IColorFrameSource* colorFrameSource = NULL;
		this->sensor->get_ColorFrameSource(&colorFrameSource);
		colorFrameSource->OpenReader(&(this->colorFrameReader));
		if (colorFrameSource) {
			colorFrameSource->Release();
			colorFrameSource = NULL;
		}

		IDepthFrameSource* framesource = NULL;
		this->sensor->get_DepthFrameSource(&framesource);
		framesource->OpenReader(&(this->depthFrameReader));
		if (framesource) {
			framesource->Release();
			framesource = NULL;
		}

		this->sensor->get_CoordinateMapper(&(this->coordinateMapper));

		return true;
	}
	else
		return false;
}

void KinectWrapper::CreateWindowsForDisplayingFrames() {
	cv::namedWindow("Kinect COLOR", cv::WINDOW_NORMAL);
	cv::resizeWindow("Kinect COLOR", colorFrameWidth / downsampleSize, colorFrameHeight / downsampleSize);
	cv::namedWindow("Kinect DEPTH", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Kinect REGISTERED COLOR", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Kinect REGISTERED COLOR - UNDISTORTED", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Kinect REGISTERED COLOR - RECTIFIED", cv::WINDOW_AUTOSIZE);
	return;
}

void KinectWrapper::DisplayFrames() {
	cv::imshow("Kinect COLOR", this->colorStream);																			// show color frame
	cv::imshow("Kinect DEPTH", this->depthStream);																			// show depth frame
	cv::imshow("Kinect REGISTERED COLOR", this->registeredColorFrame);														// show registered color frame
	cv::imshow("Kinect REGISTERED COLOR - UNDISTORTED", this->undistortedRegisteredColorFrame);								// show undistorted registered color frame
	cv::imshow("Kinect REGISTERED COLOR - RECTIFIED", this->rectifiedRegisteredColorFrame);									// show rectified  registered color frame
}

void KinectWrapper::DisplayFrames(bool displayColorFrame, bool displayDepthFrame, bool displayRegisteredColorFrame, bool displayUndistortedRegisteredColorFrame, bool displayRectifiedRegisteredColorFrame) {
	if(displayColorFrame)
		cv::imshow("Kinect COLOR", this->colorStream);																		// show color frame
	if(displayDepthFrame)
		cv::imshow("Kinect DEPTH", this->depthStream);																		// show depth frame
	if(displayRegisteredColorFrame)
		cv::imshow("Kinect REGISTERED COLOR", this->registeredColorFrame);													// show registered color frame
	if(displayUndistortedRegisteredColorFrame)
		cv::imshow("Kinect REGISTERED COLOR - UNDISTORTED", this->undistortedRegisteredColorFrame);							// show undistorted registered color frame
	if(displayRectifiedRegisteredColorFrame)
		cv::imshow("Kinect REGISTERED COLOR - RECTIFIED", this->rectifiedRegisteredColorFrame);								// show rectified  registered color frame
}

bool KinectWrapper::GetLatestColorDataFromKinect() {
	IColorFrame* colorFrame = NULL;

	if (SUCCEEDED(this->colorFrameReader->AcquireLatestFrame(&colorFrame)))
	{
		if (colorFrame)
		{
			colorFrame->CopyConvertedFrameDataToArray(this->colorFrameDataLength, this->colorData, ColorImageFormat_Bgra);
			colorFrame->Release();
			return true;
		}
		colorFrame->Release();
		return false;
	}
	return false;
}

bool KinectWrapper::GetLatestDepthDataFromKinect() {
	IDepthFrame* depthFrame = NULL;

	if (SUCCEEDED(this->depthFrameReader->AcquireLatestFrame(&depthFrame)))
	{
		if (depthFrame)
		{
			depthFrame->CopyFrameDataToArray(this->depthFrameDataLength, this->rawDepthData);
			depthFrame->Release();
			return true;
		}
		depthFrame->Release();
		return false;
	}
	return false;
}

void KinectWrapper::UpdateDepthAndCameraSpaceMapping()
{
	this->coordinateMapper->MapDepthFrameToColorSpace(this->depthFrameDataLength, this->rawDepthData, this->depthFrameDataLength, this->colorSpacePoints);
	this->coordinateMapper->MapDepthFrameToCameraSpace(this->depthFrameDataLength, this->rawDepthData, this->depthFrameDataLength, this->cameraSpacePoints);
	for (int i = 0; i < this->depthFrameDataLength; i++) {
		this->vizDepthData[4 * i] = (BYTE)(this->rawDepthData[i] >> 5);
		this->vizDepthData[4 * i + 1] = this->vizDepthData[4 * i];
		this->vizDepthData[4 * i + 2] = this->vizDepthData[4 * i];
		this->vizDepthData[4 * i + 3] = 255;

		int colorSpacePointX = (int)this->colorSpacePoints[i].X;
		int colorSpacePointY = (int)this->colorSpacePoints[i].Y;
		UINT colorSpaceIndex = 0;
		if ((colorSpacePointX >= 0) && (colorSpacePointX < this->colorFrameSize.width) && (colorSpacePointY >= 0) && (colorSpacePointY < this->colorFrameSize.height))
		{
			colorSpaceIndex = this->BPP * (colorSpacePointY * this->colorFrameSize.width + colorSpacePointX);
			this->registeredColorData[this->BPP * i] = this->colorData[colorSpaceIndex];
			this->registeredColorData[this->BPP * i + 1] = this->colorData[colorSpaceIndex + 1];
			this->registeredColorData[this->BPP * i + 2] = this->colorData[colorSpaceIndex + 2];
			this->registeredColorData[this->BPP * i + 3] = 255;
		}
	}

	memcpy(this->colorStream.data, this->colorData, this->colorFrameDataLength);
	//memcpy(this->depthStream.data, this->vizDepthData, this->depthDataLength);
	memcpy(this->depthStream.data, this->vizDepthData, this->registeredColorFrameDataLength);
	memcpy(this->registeredColorFrame.data, this->registeredColorData, this->registeredColorFrameDataLength);
	flip(this->registeredColorFrame, this->registeredColorFrame, 1);
}

void KinectWrapper::ShutDown() {
	colorStream.release();
	depthStream.release();
	registeredColorFrame.release();
	undistortedRegisteredColorFrame.release();
	grayRegisteredColorFrame.release();
	downsampledRegisteredColorFrame.release();

	delete[] colorData;
	delete[] vizDepthData;
	delete[] rawDepthData;
	delete[] registeredColorData;

	this->colorFrameReader->Release();
	this->depthFrameReader->Release();
	this->coordinateMapper->Release();
	this->sensor->Close();
	this->sensor->Release();
}


KinectWrapper::~KinectWrapper()
{
	ShutDown();
}
