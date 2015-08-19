#define _CRT_SECURE_NO_WARNINGS

#include <cassert>
#include <fstream>
#include "TextDetection.h"
#include <opencv/highgui.h>
#include <exception>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <iostream>
#include <cstdio>
#include <boost/lambda/lambda.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace tesseract;

void convertToFloatImage(IplImage * byteImage, IplImage * floatImage)
{
	cvConvertScale(byteImage, floatImage, 1 / 255., 0);
}

class FeatureError : public std::exception
{
	std::string message;
public:
	FeatureError(const std::string & msg, const std::string & file)
	{
		std::stringstream ss;

		ss << msg << " " << file;
		message = msg.c_str();
	}
	~FeatureError() throw ()
	{
	}
};

IplImage * loadByteImage(const char * name)
{
	IplImage * image = cvLoadImage(name);

	if (!image)
	{
		return 0;
	}
	cvCvtColor(image, image, CV_BGR2RGB);
	return image;
}

IplImage * loadFloatImage(const char * name)
{
	IplImage * image = cvLoadImage(name);

	if (!image)
	{
		return 0;
	}
	cvCvtColor(image, image, CV_BGR2RGB);
	IplImage * floatingImage = cvCreateImage(cvGetSize(image),
		IPL_DEPTH_32F, 3);
	cvConvertScale(image, floatingImage, 1 / 255., 0);
	cvReleaseImage(&image);
	return floatingImage;
}

int mainTextDetection(Mat mat, Point pt1, Point pt2, Mat captureFrame)
{
	//Mat mat = imread("E:/photo.jpg");

	IplImage* img = new IplImage(mat);

	IplImage * byteQueryImage = img;
	if (!byteQueryImage)
	{
		printf("couldn't load query image\n");
		return -1;
	}

	bool darkOnLight = true;

	// Detect text in the image
	IplImage * output = textDetection(byteQueryImage, darkOnLight);



	/////////////////////////////////////////////////////////////////
	//TESSERACT OCR IMPLEMENTATION
	/////////////////////////////////////////////////////////////////
	Mat matCon(output);
	Mat gray, process, can;
	cvtColor(matCon, can, CV_BGR2GRAY);
	equalizeHist(can, gray);
	threshold(gray, can, 50, 255, THRESH_BINARY);

	// Pass it to Tesseract API
	TessBaseAPI tess;
	tess.Init(NULL, "bib", OEM_DEFAULT);
	tess.SetPageSegMode(PSM_SINGLE_BLOCK);
	tess.SetImage((uchar*)can.data, can.cols, can.rows, 1, can.cols);

	// Get the text
	char* out = tess.GetUTF8Text();
	
	//outputs the string of text into the console application
	cout << out;
	////////////////////////////////////////////////////////////////

	cvReleaseImage(&output);
	return 0;
}

int main(int argc, char * * argv)
{
	//create the cascade classifier object used for the face detection
	CascadeClassifier face_cascade;

	//load frontal face xml file 
	face_cascade.load("E:/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml");

	//setup video capture device and link it to webcam, 0 = default and 1 = external
	VideoCapture captureDevice;
	captureDevice.open(1);

	//setup image files used in the capture process
	Mat captureFrame, grayscaleFrame, tess;

	while (true)
	{
		//capture a new image frame
		captureDevice >> captureFrame;

		//convert captured image to gray scale and equalize
		cvtColor(captureFrame, grayscaleFrame, CV_BGR2GRAY);
		equalizeHist(grayscaleFrame, grayscaleFrame);

		//create a vector array to store the face found
		vector<Rect> faces;
		vector<Mat> rgb;

		Mat body;

		//locate faces and store them in the vector array
		face_cascade.detectMultiScale(grayscaleFrame, faces, 1.3, 4, CV_HAAR_SCALE_IMAGE);

		//create boolean in the case that no faces were found
		bool detect = false;

		//draw a rectangle for all found faces in the vector array on the original image
		for (int i = 0; i < faces.size(); i++)
		{
			Point pt1(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			Point pt2(faces[i].x, faces[i].y);
			rectangle(captureFrame, pt1, pt2, cvScalar(0, 0, 255, 0), 1, 8, 0);
		}

		//if face found, draw another rectangle to locate body
		if (faces.size() != 0){
			for (int i = 0; i < faces.size(); i++)
			{
				//points are drawn from the location of the face
				Point pt1((faces[i].x) + (faces[i].width + 50), (faces[i].y) + (faces[i].height + 320));
				Point pt2((faces[i].x - 60), (faces[i].y));

				rectangle(captureFrame, pt1, pt2, cvScalar(0, 255, 255, 0), 1, 8, 0);

				//ensures no out of bounds exception in rectangle
				Rect borders(Point(0, 0), captureFrame.size());

				//copy area that has body to another Mat image
				Rect roi = Rect(pt1, pt2) & borders;
				body = captureFrame(roi);

				mainTextDetection(body, pt1, pt2, captureFrame);
			}

		}
		else{
		}

		//show frames with rectangles on face + body
		imshow("TrackFace", captureFrame);

		if (waitKey(20) == 27)
			break;

	}
	

	///////////////////////////////////////////////////////
	/*Mat mat = imread("E:/mar1.jpg");
	IplImage* img = new IplImage(mat);

	mainTextDetection(img);*/

	return 0;
}
