#pragma once
#include <Windows.h>
#include <Kinect.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


class KinectWrapper
{
public:
	UINT colorFrameWidth;
	UINT colorFrameHeight;
	UINT depthFrameWidth;
	UINT depthFrameHeight;
	UINT registeredFrameWidth;
	UINT registeredFrameHeight;
	UINT BPP;																							// bytes per pixel
	UINT downsampleSize;																				// amount by which a large frame is downsampled (generally for display purposes)

	UINT colorFrameDataLength;																			// total length (in bytes) used for storing color frame data
	UINT depthFrameDataLength;																			// total length (in bytes) used for storing depth frame data
	UINT registeredColorFrameDataLength;																// total length (in bytes) used for storing registered color frame data

	BYTE*   colorData;																					// BGRA array containing color stream data
	BYTE*   vizDepthData;																				// BYTE array containing depth stream data for visualization
	USHORT* rawDepthData;																				// USHORT array containing depth stream data
	BYTE*   registeredColorData;																		// BGRA array containing color stream data

	cv::Mat colorStream;																				// buffer for storing color frame
	cv::Mat depthStream;																				// buffer for storing depth frame
	cv::Mat registeredColorFrame;																		// buffer for storing registsred color frame
	cv::Mat undistortedRegisteredColorFrame;															// buffer for storing undistorted registered color frame
	cv::Mat rectifiedRegisteredColorFrame;																// buffer for storing rectified registered color frame
	cv::Mat grayRegisteredColorFrame;																	// buffer for converting color frame to gray
	cv::Mat downsampledRegisteredColorFrame;															// buffer for storing downsampled frame

	cv::Size colorFrameSize;
	cv::Size depthFrameSize;
	cv::Size registeredColorFrameSize;

	IKinectSensor* sensor;																			// kinect sensor
	IColorFrameReader* colorFrameReader;															// kinect color data source
	IDepthFrameReader* depthFrameReader;															// kinect depth data source
	ICoordinateMapper* coordinateMapper;															// kinect coordinate mapper for registering frames

	ColorSpacePoint* colorSpacePoints;																// coordinate mapping between color and depth spaces
	CameraSpacePoint* cameraSpacePoints;															// coordinate mapping between depth and world spaces

	KinectWrapper(UINT colorFrameWidth, UINT colorFrameHeight, UINT depthFrameWidth, UINT depthFrameHeight, UINT bpp, UINT downsampleSize);
	bool Initialize();
	void CreateWindowsForDisplayingFrames();
	void DisplayFrames();
	void DisplayFrames(bool displayColorFrame, bool displayDepthFrame, bool displayRegisteredColorFrame, bool displayUndistortedRegisteredColorFrame, bool displayRectifiedRegisteredColorFrame);
	bool GetLatestColorDataFromKinect();
	bool GetLatestDepthDataFromKinect();
	void UpdateDepthAndCameraSpaceMapping();
	void ShutDown();
	~KinectWrapper();
};

