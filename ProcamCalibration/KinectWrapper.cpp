#include "KinectWrapper.h"

KinectWrapper::KinectWrapper()
{
	this->colorFrameWidth = 1920;
	this->colorFrameHeight = 1080;
	this->depthFrameWidth = 512;
	this->depthFrameHeight = 424;
	this->registeredFrameWidth = 512;
	this->registeredFrameHeight = 424;
	this->downsampleSize = 2;
	this->BPP = 4;

	colorFrameDataLength = this->colorFrameWidth * this->colorFrameHeight * this->BPP;
	depthFrameDataLength = this->depthFrameWidth * this->depthFrameHeight;
	registeredColorFrameDataLength = this->depthFrameDataLength * this->BPP;

	colorData = new BYTE[this->colorFrameDataLength];
	vizDepthData = new BYTE[this->registeredColorFrameDataLength];
	rawDepthData = new USHORT[this->depthFrameDataLength];
	registeredColorData = new BYTE[this->registeredColorFrameDataLength];
	allFramesData = new BYTE[1280 * 720 * this->BPP];

	colorFrame = cv::Mat(this->colorFrameHeight, this->colorFrameWidth, CV_8UC4, &colorData[0]);
	depthFrame = cv::Mat(this->depthFrameHeight, this->depthFrameWidth, CV_8UC4, &vizDepthData[0]);
	registeredColorFrame = cv::Mat(this->depthFrameHeight, this->depthFrameWidth, CV_8UC4, &registeredColorData[0]);
	allKinectFrames = cv::Mat(720, 1280, CV_8UC4, &allFramesData[0]);

	colorFrameSize = cv::Size(this->colorFrameWidth, this->colorFrameHeight);
	depthFrameSize = cv::Size(this->depthFrameWidth, this->depthFrameHeight);
	registeredColorFrameSize = cv::Size(this->depthFrameWidth, this->depthFrameHeight);

	colorSpacePoints = new ColorSpacePoint[this->depthFrameDataLength];
	cameraSpacePoints = new CameraSpacePoint[this->depthFrameDataLength];
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

void KinectWrapper::CreateWindowsForDisplayingFrames(bool showUndistortedRegisteredColorFrame, bool showRectifiedRegisteredColorFrame) {
	cv::namedWindow("Kinect COLOR DEPTH and REGISTERED frames", cv::WINDOW_AUTOSIZE);
	cv::resizeWindow("Kinect COLOR DEPTH and REGISTERED frames", 1280, 720);
	
	this->showUndistortedRegisteredColorFrame = showUndistortedRegisteredColorFrame;
	this->showRectifiedRegisteredColorFrame = showRectifiedRegisteredColorFrame;

	if(showUndistortedRegisteredColorFrame)
		cv::namedWindow("Kinect REGISTERED COLOR - UNDISTORTED", cv::WINDOW_AUTOSIZE);
	if(showRectifiedRegisteredColorFrame)
		cv::namedWindow("Kinect REGISTERED COLOR - RECTIFIED", cv::WINDOW_AUTOSIZE);
	return;
}

void KinectWrapper::DisplayFrames() {
	cv::imshow("Kinect COLOR DEPTH and REGISTERED frames", this->allKinectFrames);
	if(this->showUndistortedRegisteredColorFrame)
		cv::imshow("Kinect REGISTERED COLOR - UNDISTORTED", this->undistortedRegisteredColorFrame);							// show undistorted registered color frame
	if(this->showRectifiedRegisteredColorFrame)
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

	memcpy(this->colorFrame.data, this->colorData, this->colorFrameDataLength);
	memcpy(this->depthFrame.data, this->vizDepthData, this->registeredColorFrameDataLength);
	memcpy(this->registeredColorFrame.data, this->registeredColorData, this->registeredColorFrameDataLength);
	flip(this->colorFrame, this->colorFrame, 1);
	flip(this->depthFrame, this->depthFrame, 1);
	flip(this->registeredColorFrame, this->registeredColorFrame, 1);

	this->insetImage1 = this->allKinectFrames(cv::Rect(106, 32, 448, 252));
	this->insetImage2 = this->allKinectFrames(cv::Rect(106, 316, 448, 371));
	this->insetImage3 = this->allKinectFrames(cv::Rect(660, 148, this->registeredFrameWidth, this->registeredFrameHeight));

	cv::resize(colorFrame, insetImage1, cv::Size(448, 252), 0, 0, cv::INTER_LINEAR);
	cv::resize(depthFrame, insetImage2, cv::Size(448, 371), 0, 0, cv::INTER_LINEAR);
	this->registeredColorFrame.copyTo(insetImage3);
	
	cv::imshow("Kinect COLOR DEPTH and REGISTERED frames", this->allKinectFrames);
}

void KinectWrapper::ShutDown() {
	this->colorFrame.release();
	this->depthFrame.release();
	this->registeredColorFrame.release();
	this->undistortedRegisteredColorFrame.release();
	this->grayRegisteredColorFrame.release();
	this->downsampledRegisteredColorFrame.release();
	this->allKinectFrames.release();

	delete[] this->colorData;
	delete[] this->vizDepthData;
	delete[] this->rawDepthData;
	delete[] this->registeredColorData;

	this->colorFrameReader->Release();
	this->depthFrameReader->Release();
	this->coordinateMapper->Release();
	this->sensor->Close();
	this->sensor->Release();
}


KinectWrapper::~KinectWrapper()
{
	this->ShutDown();
}
