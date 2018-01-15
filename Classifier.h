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
int width = 32, height = 32;
int num_files = 0;

void readCSV(const string &filename) {
	try {
		ifstream file(filename); 
		string fileName, w, h, r1x, r1y, r2x, r2y, classID;
		int cvt1, cvt2, x = 10;
		while (file.good())
		{
			cvt1 = 0;
			cvt2 = 0;
			getline(file, fileName, ';');
			getline(file, w, ';');
			getline(file, h, ';');
			getline(file, r1x, ';');
			getline(file, r1y, ';');
			getline(file, r2x, ';');
			getline(file, r2y, ';');
			getline(file, classID, '\n');
			if (fileName == "" || w == "" || r1x == "" || r1y == "" || r2x == "" || r2y == "" || classID == "")
				continue;
			if (fileName == "Filename")
				continue;
			fileVector.push_back(fileName);
			stringstream cv1(r1x);
			cv1 >> cvt1;
			stringstream cv2(r1y);
			cv2 >> cvt2;
			startVector.push_back(Point(cvt1, cvt2));
			stringstream cv3(r2x);
			cv3 >> cvt1;
			stringstream cv4(r2y);
			cv4 >> cvt2;
			endVector.push_back(Point(cvt1, cvt2));
			stringstream cvt(classID);
			cvt >> cvt1;
			labels.push_back(cvt1);
			num_files++;
		}
		cout << num_files << endl;
	}
	catch (Exception &e) {
		cerr << e.msg << endl;
	}
}
void synthesize(Mat src, int label) {
	Mat dst;
	if (label == 17) {
		flip(src, dst, 0);
		matVector.push_back(dst);
		flip(src, dst, 1);
		matVector.push_back(dst);
		flip(src, dst, -1);
		matVector.push_back(dst);
		labels.push_back(label);
		labels.push_back(label);
		labels.push_back(label);
	}
	else if (label == 32) {
		flip(src, dst, -1);
		matVector.push_back(dst);
		labels.push_back(label);
	} 
	else if (label == 29 || label == 33) {
		flip(src, dst, -1);
		matVector.push_back(dst);
		labels.push_back(label+1);
	}
	else if (label == 30 || label == 34) {
		flip(src, dst, -1);
		matVector.push_back(dst);
		labels.push_back(label - 1);
	}
}
Mat poorLight(Mat src) {
	Mat des;
	Mat chanel[3];
	split(src, chanel);
	//imshow("R", chanel[0]);
	//imshow("G", chanel[1]);
	//imshow("B", chanel[2]);
	//chanel[2] = Mat::zeros(chanel[2].size(), CV_8UC1);
	//chanel[0] = Mat::zeros(chanel[0].size(), CV_8UC1);
	//chanel[1] = Mat::zeros(chanel[1].size(), CV_8UC1);

	//code for histogam equalization
	equalizeHist(chanel[0], chanel[0]);
	equalizeHist(chanel[1], chanel[1]);
	equalizeHist(chanel[2], chanel[2]);

	 //code for process color constancy 
	chanel[0].convertTo(chanel[0], -1, 1.5, 0);
	chanel[1].convertTo(chanel[1], -1, 1.5, 0);
	chanel[2].convertTo(chanel[2], -1, 1.5, 0);

	Mat out;
	merge(chanel, 3, out);
	//imshow("result ", out);
	des = out;
	waitKey(0);
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

	//img = poorLight(img);
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
void imgProcess(const string dir) {
	try {
		for (int i = 0; i < fileVector.size(); i++) {
			Mat img = imread(dir + fileVector.at(i));
			img = preprocess(img);
			matVector.push_back(img);
		}
	}
	catch (Exception &e) {
		cerr << e.msg << endl;
	}
}
void augment() {
	try {
		long x = matVector.size();
		cout << "Number of data before synthesizing: " << x << endl;
		for (int i = 0; i < x; i++) {
			Mat img = matVector.at(i);
			int label = labels.at(i);
			synthesize(img, label);
		}
		cout << "Number of data after synthesizing: " << matVector.size() << endl;
	}
	catch (Exception &e) {
		cerr << e.msg << endl;
	}
}
Mat new_image, label_mat;
void createMatrix() {
	num_files = labels.size();
	Mat data (num_files, height*width, CV_32FC1); //Training sample from input images
	int ii = 0;
	for (int i = 0; i < num_files; i++) {
		Mat temp = matVector.at(i);
		ii = 0;
		for (int j = 0; j < temp.rows; j++) {
			for (int k = 0; k < temp.cols; k++) {
				data.at<float>(i, ii++) = temp.at<uchar>(j, k);
			}
		}
	}
	Mat label(num_files, 1, CV_32S);
	for (int i = 0; i < num_files; i++) {
		label.at<int>(i, 0) = labels.at(i);
	}
	new_image.convertTo(new_image, CV_32FC1);
	new_image = data;
	label_mat = label;
}
float calculateAccuracy(Ptr<TrainData> testData) {
	cout << "Number of train data: " << testData->getTestSamples().rows << endl;
	Ptr<SVM> svm = SVM::create();
	//Ptr<SVMSGD> svm = SVMSGD::create();
	svm = svm->load("svm.xml");
	Mat result;
	return svm->StatModel::calcError(testData, false, result);
}
void train(Ptr<TrainData> trainData) {
	try {
		cout << "Number of train data: " << trainData->getTrainSamples().rows << endl << "Training... " << endl;
		Ptr<SVM> svm = SVM::create();
		svm->setType(SVM::C_SVC);
		svm->setKernel(SVM::LINEAR);
		svm->setC(0.00001);
		svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));
		//svm = svm->load("svm.xml");
		svm->train(trainData);
		svm->save("svm.xml");
		cout << "Done" << endl;
	}
	catch (Exception &e) {
		cerr << e.msg << endl;
	}
}
int predict(Mat test_img) {
	try {
		Ptr<SVM> svm = SVM::create();
		svm = svm->load("svm.xml");
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
void openFile() {
	cout << "Loading dataset..." << endl;
	readCSV("./Training/32/GT-00004.csv");
	imgProcess("./Training/32/");
	fileVector.clear();
	startVector.clear();
	endVector.clear();
	readCSV("./Training/17/GT-00005.csv");
	imgProcess("./Training/17/");
	fileVector.clear();
	startVector.clear();
	endVector.clear();
	readCSV("./Training/29/GT-00006.csv");
	imgProcess("./Training/29/");
	fileVector.clear();
	startVector.clear();
	endVector.clear();
	readCSV("./Training/30/GT-00007.csv");
	imgProcess("./Training/30/");
	fileVector.clear();
	startVector.clear();
	endVector.clear();
	readCSV("./Training/14/GT-00008.csv");
	imgProcess("./Training/14/");
	fileVector.clear();
	startVector.clear();
	endVector.clear();
	readCSV("./Training/34/GT-00009.csv");
	imgProcess("./Training/34/");
	fileVector.clear();
	startVector.clear();
	endVector.clear();
	readCSV("./Training/33/GT-00010.csv");
	imgProcess("./Training/33/");
	fileVector.clear();
	startVector.clear();
	endVector.clear();
	readCSV("./Training/02345/00000/GT-00000.csv");
	imgProcess("./Training/02345/00000/");
	fileVector.clear();
	startVector.clear();
	endVector.clear();
	readCSV("./Training/02345/00001/GT-00001.csv");
	imgProcess("./Training/02345/00001/");
	fileVector.clear();
	startVector.clear();
	endVector.clear();
	readCSV("./Training/02345/00002/GT-00002.csv");
	imgProcess("./Training/02345/00002/");
	fileVector.clear();
	startVector.clear();
	endVector.clear();
	readCSV("./Training/02345/00003/GT-00003.csv");
	imgProcess("./Training/02345/00003/");
	fileVector.clear();
	startVector.clear();
	endVector.clear();
	readCSV("./Training/02345/00004/GT-00004.csv");
	imgProcess("./Training/02345/00004/");
	fileVector.clear();
	startVector.clear();
	endVector.clear();
	cout << "Dataset loaded" << endl;
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


