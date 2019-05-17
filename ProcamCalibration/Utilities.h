#pragma once
#include "KinectWrapper.h"
#include "CalibrationHandler.h"

#include <fstream>

class Utilities
{
public:
	//Utilities();
	//~Utilities();
	void TestOpenCV(bool showImage);
	void DisplayHomographicallyCalibratedFramesForProjectorAndKinect(KinectWrapper& kinectFrameData, CalibrationHandler& chessboardCalibrationData);
	void DisplayStereoCalibratedFramesForProjectorAndKinect(KinectWrapper& kinectFrameData, CalibrationHandler& chessboardCalibrationData);
};

