#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {

	//make empty template 
	Mat img(180, 768, CV_8UC1);
	img = 0;
	Rect2d rc = { 0, 0, 768, 60 };
	rectangle(img, rc, { 100 }, 1);

	//create gray-scale 256b image
	Mat i_1 = img(rc);

	for (int i = 0; i < 256; ++i) {
		for (int j = rc.y; j < rc.y + rc.height; ++j) {
			line(i_1, Point2d(i * 3, j), Point2d(i * 3 + 2, j), Scalar(i), 1);
		}
	}

	//cv::pow gamma-correction method with gamma = 2.3
	Mat g_1 = i_1.clone();
	g_1.convertTo(g_1, CV_32F, 1. / 255);
	//check time for cv::pow method
	chrono::steady_clock::time_point begin_t_pow = chrono::steady_clock::now();

	pow(g_1, 2.3, g_1);

	chrono::steady_clock::time_point end_t_pow = chrono::steady_clock::now();
	cout << chrono::duration_cast<chrono::microseconds>(end_t_pow - begin_t_pow).count() << " micro_s" << endl;

	//copy to template
	g_1.convertTo(g_1, CV_8UC1, 255);
	rc.y += rc.height;
	g_1.copyTo(img(Rect(rc.x, rc.y, g_1.cols, g_1.rows)));
	//draw border to separate images
	rectangle(img, rc, { 150 }, 1);

	//direct gamma-correction method with gamma = 2.3
	Mat g_2 = i_1.clone();
	g_2.convertTo(g_2, CV_32F, 1. / 255);
	//check time for direct method
	chrono::steady_clock::time_point begin_t_direct = chrono::steady_clock::now();

	for (int i = 0; i < g_2.cols; ++i) {
		for (int j = 0; j < g_2.rows; ++j) {
			g_2.at<float>(j, i) = pow(g_2.at<float>(j, i), 2.3);
		}
	}

	chrono::steady_clock::time_point end_t_direct = chrono::steady_clock::now();
	cout << chrono::duration_cast<chrono::microseconds>(end_t_direct - begin_t_direct).count() << " micro_s" << endl;

	//copy to template
	g_2.convertTo(g_2, CV_8UC1, 255);
	rc.y += rc.height;
	g_2.copyTo(img(Rect(rc.x, rc.y, g_2.cols, g_2.rows)));
	//draw border to separate images
	rectangle(img, rc, { 250 }, 1);

	//show result 
	imshow("laba_1", img);
	waitKey(0);

	//save result
	imwrite("lab01.png", img);
}