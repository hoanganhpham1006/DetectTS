#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/photo.hpp>

using namespace cv;
using namespace std;
using namespace cv::ml;
using cv::cuda::GpuMat;
vector<string> fileVector;
vector<Point> startVector;
vector<Point> endVector;
vector<int> labels;
vector<Mat> matVector;
int width = 40, height = 40;
int num_files = 0;


Mat poorLight(Mat src) {
	Mat des;
	Mat chanel[3];
	split(src, chanel);
	equalizeHist(chanel[0], chanel[0]);
	equalizeHist(chanel[1], chanel[1]);
	equalizeHist(chanel[2], chanel[2]);

	 //code for process color constancy 
	chanel[0].convertTo(chanel[0], -1, 1.3, 0);
	chanel[1].convertTo(chanel[1], -1, 1.3, 0);
	chanel[2].convertTo(chanel[2], -1, 1.3, 0);

	Mat out;
	merge(chanel, 3, out);
	//imshow("result ", out);
	des = out;
	return des;
}
Mat preprocess(Mat img) {
	Mat temp;
	//Point end;
	//if (endVector.at(i).x > img.cols || startVector.at(i).y > img.rows) {
	//	end.x = endVector.at(i).y;
	//	end.y = endVector.at(i).x;
	//}
	//else {
	//	end = endVector.at(i);
	//}
	//Rect roi(startVector.at(i), end);
	//img = img(roi);

	img = poorLight(img);
	//cvtColor(img, img, CV_BGR2GRAY);
	//equalizeHist(img, img);
	//fastNlMeansDenoising(img, img, 50.0, 7, 21);
	//img = img > 128;

	

	//equalizeHist(img, img);
	//fastNlMeansDenoising(img, img, 50.0, 7, 21);
	//Mat im_bw = img > 128;

	//GpuMat gI;
	//gI.upload(img);
	//cv::cuda::fastNlMeansDenoising(GpuMat(gI), gI, 50);
	//gI.download(img);
	resize(img, temp, Size(width, height));
	return temp;
}
int predict(Mat test_img, Ptr<SVM> svm) {
	try {
		
		test_img = preprocess(test_img);
		Mat test_img_mat(1, width * height, CV_32FC1);
		int x = 0;
		for (int i = 0; i < test_img.rows; i++) {
			for (int j = 0; j < test_img.cols; j++) {
				test_img_mat.at<float>(0, x++) = test_img.at<uchar>(i, j);
			}
		}
		float response = svm->predict(test_img_mat);
		//cout << "Predicted class: " << response << endl;
		if (response == 0 || response == 2 || response == 3 || response == 4 || response == 5) return 7;
		else if (response == 17) return 6;
		else if (response == 29) return 4;
		else if (response == 30) return 5;
		else if (response == 14) return 1;
		else if (response == 34) return 2;
		else if (response == 33) return 3;
		else return 8;
	}
	catch (Exception e) {
		cerr << e.msg << endl;
	}

}

//int main() {
//
//	openFile();
//	augment();
//	createMatrix();
//	Ptr<TrainData> trainData = TrainData::create(new_image, ROW_SAMPLE, label_mat);
//	trainData->setTrainTestSplitRatio(0.8);
//	train(trainData);
//	//predict();
//	cout << "Accuracy: " << calculateAccuracy(trainData) << endl;
//	getchar();
//
//}


