#pragma once
#include <Windows.h>
#include <iostream>
#include <Kinect.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


class KinectWrapper
{
private:
	bool showUndistortedRegisteredColorFrame = false, showRectifiedRegisteredColorFrame = false;

public:
	UINT colorFrameWidth = 1920;
	UINT colorFrameHeight = 1080;
	UINT depthFrameWidth = 512;
	UINT depthFrameHeight = 424;
	UINT registeredFrameWidth = 512;
	UINT registeredFrameHeight = 424;
	UINT BPP = 4;																					// bytes per pixel
	UINT downsampleSize = 2;																		// amount by which a large frame is downsampled (generally for display purposes)

	UINT colorFrameDataLength;																		// total length (in bytes) used for storing color frame data
	UINT depthFrameDataLength;																		// total length (in bytes) used for storing depth frame data
	UINT registeredColorFrameDataLength;															// total length (in bytes) used for storing registered color frame data

	BYTE*   colorData;																				// BGRA array containing color stream data
	BYTE*   vizDepthData;																			// BYTE array containing depth stream data for visualization
	USHORT* rawDepthData;																			// USHORT array containing depth stream data
	BYTE*   registeredColorData;																	// BGRA array containing color stream data
	BYTE*   allFramesData;																			// BGRA array containing all streams data

	cv::Mat colorFrame;																				// buffer for storing color frame
	cv::Mat depthFrame;																				// buffer for storing depth frame
	cv::Mat registeredColorFrame;																	// buffer for storing registsred color frame
	cv::Mat undistortedRegisteredColorFrame;														// buffer for storing undistorted registered color frame
	cv::Mat rectifiedRegisteredColorFrame;															// buffer for storing rectified registered color frame
	cv::Mat grayRegisteredColorFrame;																// buffer for converting color frame to gray
	cv::Mat downsampledRegisteredColorFrame;														// buffer for storing downsampled frame
	
	cv::Mat allKinectFrames;																		// buffer for storing all Kinect frames
	cv::Mat insetImage1, insetImage2, insetImage3;													// buffer for storing reseized frames

	cv::Size colorFrameSize;
	cv::Size depthFrameSize;
	cv::Size registeredColorFrameSize;

	IKinectSensor* sensor;																			// kinect sensor
	IColorFrameReader* colorFrameReader;															// kinect color data source
	IDepthFrameReader* depthFrameReader;															// kinect depth data source
	ICoordinateMapper* coordinateMapper;															// kinect coordinate mapper for registering frames

	ColorSpacePoint* colorSpacePoints;																// coordinate mapping between color and depth spaces
	CameraSpacePoint* cameraSpacePoints;															// coordinate mapping between depth and world spaces

	KinectWrapper();
	bool Initialize();
	void CreateWindowsForDisplayingFrames(bool showUndistortedRegisteredColorFrame, bool showRectifiedRegisteredColorFrame);
	void DisplayFrames();
	bool GetLatestColorDataFromKinect();
	bool GetLatestDepthDataFromKinect();
	void UpdateDepthAndCameraSpaceMapping();
	void ShutDown();
	~KinectWrapper();
};

