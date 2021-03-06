// TestOpenCV.cpp : Defines the entry point for the console application.
#include "stdafx.h"
#include<opencv2/opencv.hpp>
#include<iostream>
#include "Classifier.h"

using namespace std;
using namespace cv;

int TAKE_AWAY = 15;
VideoCapture cap("testvideo1.MOV");
//Point2f p1 = Point2f(0, 0);
//Point2f p2 = Point2f(0, 0);
int Blue_h_low = 200;
int Blue_s_low = 45;
int Blue_v_low = 45;
int Blue_h_high = 220;
int Blue_s_high = 100;
int Blue_v_high = 100;

int Red_h_low = 300;
int Red_s_low = 35;
int Red_v_low = 35;
int Red_h_high = 359;
int Red_s_high = 100;
int Red_v_high = 100;

int White_h_low = 210;
int White_s_low = 0;
int White_v_low = 90;
int White_h_high = 220;
int White_s_high = 8;
int White_v_high = 100;

float v = 0.5;

void processingColor(Mat &img) {
	//Convert BGR to HSV
	Mat img1, img2, img3;
	cvtColor(img, img, COLOR_BGR2HSV);
	inRange(img, Scalar(Blue_h_low/2, Blue_s_low*2.55, Blue_v_low*2.55), Scalar(Blue_h_high/2, Blue_s_high*2.55, Blue_v_high*2.55), img1); //Blue
	inRange(img, Scalar(Red_h_low/2, Red_s_low*2.55, Red_v_low*2.55), Scalar(Red_h_high/2, Red_s_high*2.55, Red_v_high*2.55), img2); //Red
	//inRange(img, Scalar(White_h_low/2, White_s_low*2.55, White_v_low*2.55), Scalar(White_h_high/2, White_s_high*2.55, White_v_high*2.55), img3); //White
	addWeighted(img1, 1.0, img2, 1.0, 0.0, img);
	//addWeighted(img, 1.0, img3, 1.0, 0.0, img);
	GaussianBlur(img, img, Size(9, 9), 2, 2);
	threshold(img, img, 128, 255, CV_THRESH_OTSU);

	int maxDistance = 15;
	Mat kernel = Mat(Size(maxDistance, maxDistance), CV_8UC1, Scalar(0));
	line(kernel, Point(kernel.cols / 2, 0), Point(kernel.cols / 2, kernel.rows), Scalar(255, 255, 255), 1);
	cv::morphologyEx(img, img, MORPH_CLOSE, kernel);

}

void contour(Mat &img_origin, Mat &img, Point2f &tl, Point2f &br) {
	std::vector<std::vector<cv::Point> > contours;
	Mat contourOutput = img.clone();
	findContours(contourOutput, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	for (size_t i = 0; i < contours.size(); i++){
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}
	
	for (size_t i = 0; i< contours.size(); i++) {
		Scalar color = cv::Scalar(0, 255, 0);
		if (contourArea(contours[i]) >= TAKE_AWAY*TAKE_AWAY) {
			drawContours(img_origin, contours_poly, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point());
			rectangle(img_origin, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
			tl = boundRect[i].tl();
			br = boundRect[i].br();
		}
	}
}

void updateString(String &s, int ma) {
	switch (ma) {
		case 1: {
			s = "Bien dung";
			break;
		}
		case 2: {
			s = "Re trai";
			break;
		}
		case 3: {
			s = "Re phai";
		}
		case 4: {
			s = "Cam re trai";
			break;
		}
		case 5: {
			s = "Cam re phai";
			break;
		}
		case 6: {
			s = "Bien mot chieu";
			break;
		}
		case 7: {
			s = "Toc do toi da";
			break;
		}
		case 8: {
			s = "Bien khac";
			break;
		}

		default : {
			s = "Noop!";
			break;
		}
	}
}

int main() {
	Ptr<SVM> svm = SVM::create();
	svm = svm->load("svm.xml");
	if (!cap.isOpened()) {
		return -1;
	}
	namedWindow("frame", WINDOW_AUTOSIZE);
	namedWindow("binary", WINDOW_AUTOSIZE);
	namedWindow("catch", WINDOW_AUTOSIZE);
	long long frameNo = 0;
	Point2f  tlp, brp;
	while (1) {
		frameNo++;
		Point2f tl, br;
		Mat frame, binary, drawed, take;
		cap >> frame;
		if (!cap.retrieve(frame)) break;
		frame = poorLight(frame);
		resize(frame, frame, Size(0, 0), 0.5, 0.5);
		//frame = imread("10.jpg");
		cvtColor(frame, binary, 0);
		cvtColor(frame, drawed, 0);
		//line(drawed, p1, p2, Scalar(0, 255, 0), 3);
		processingColor(binary);
		contour(drawed, binary, tl, br);
		try {
			if (tlp.x != 0 && tl == br && brp.x < 629) {
				tlp.x -= v;
				tlp.y -= v;
				brp.x -= v;
				brp.y -= v;
				tl = tlp;
				br = brp;
			}
			if (tlp.x == 0 || brp.x >= 629) {
				Point2f tlp, brp;
			}
			Rect crop = Rect(tl, br);
			if (tl != br) {
				take = frame(crop);
				int maBien = predict(take, svm);
				cout << frameNo << " " << maBien << " " << tl.x << " " << tl.y << " " << br.x << " " << br.y << endl;
				String tenbien = "Hello world";
				updateString(tenbien, maBien);
				Point2f start;
				if (tl.x <= 320) start = Point2f(br.x, br.y);
				else start = Point2f(br.x - tenbien.length()*15 - (br.x - tl.x),br.y);
				putText(drawed, tenbien, start, FONT_HERSHEY_COMPLEX, 0.7, Scalar(0, 255, 0));
				resize(take, take, Size(200, 200));
				imshow("catch", take);
				tlp = tl;
				brp = br;
			}
			
			
		}
		catch (Exception ex) {

		}
		imshow("frame", drawed);
		imshow("binary", binary);
		waitKey(1);
	}
	return 0;
}

