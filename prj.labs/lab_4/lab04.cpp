#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;



Mat findDifference(Mat etalon, Mat example) {

	Mat stats = Mat::zeros(Size(2, 3), CV_32F);
	//TP, FP,FN,TN
	for (int i = 0; i < etalon.cols; ++i) {
		for (int j = 0; j < etalon.rows; ++j) {
			if (etalon.at<uchar>(j,i) == 0) {

				if (example.at<uchar>(j,i) == 0) {
					stats.at<float>(0, 0) += (float)1;
				}
				else {
					stats.at<float>(0, 1) += (float)1;;
				}
			}
			else {

				if (example.at<uchar>(j, i) == 255) {
					stats.at<float>(1, 1) += (float)1;
				}
				else {
					stats.at<float>(1, 0) += (float)1;
				}
			}
		}
	}

	//balanced accuracy
	float sensitivity = stats.at<float>(0, 0) / (stats.at<float>(0, 0) + stats.at<float>(1, 0));
	float specificity = stats.at<float>(1, 1) / (stats.at<float>(1, 1) + stats.at<float>(0, 1));
	stats.at<float>(2, 0) = (sensitivity + specificity) / 2;

	//f_1 score
	float precision = stats.at<float>(0, 0) / (stats.at<float>(0, 0) + stats.at<float>(0, 1));
	float recall = stats.at<float>(0, 0) / (stats.at<float>(0, 0) + stats.at<float>(1, 0));
	stats.at<float>(2, 1) = 2 * precision * recall / (precision + recall);


	return stats;
}

Mat showDifference(Mat etalon, Mat example) {

	Mat diff_img = example.clone();
	cvtColor(diff_img, diff_img, COLOR_GRAY2BGR);

	for (int i = 0; i < etalon.cols; ++i) {
		for (int j = 0; j < etalon.rows; ++j) {
			if (etalon.at<uchar>(j, i) == 0) {

				if (example.at<uchar>(j, i) != 0) {
					diff_img.at<Vec3b>(j, i) = { 0, 0, 255 };
				}
			}
			else {

				if (example.at<uchar>(j, i) != 255) {
					diff_img.at<Vec3b>(j, i) = {0, 255, 0};
				}
			}
		}
	}
	return diff_img;
}

void getStats(Mat stat) {
	cout << "TP: " << stat.at<float>(0, 0) << endl;
	cout << "FP: " << stat.at<float>(0, 1) << endl;
	cout << "FN: " << stat.at<float>(1, 0) << endl;
	cout << "TN: " << stat.at<float>(1, 1) << endl;
	cout << "Balanced accuracy: " << stat.at<float>(2, 0) << endl;
	cout << "F1 score: " << stat.at<float>(2, 1) << endl;
}


int main() {

	//load source image and etalon
	string image_path_src = samples::findFile("lab04.src.jpg");
	Mat original_img = imread(image_path_src, IMREAD_COLOR);
	string image_path = samples::findFile("lab04.etalon.png");
	Mat etalon = imread(image_path, IMREAD_GRAYSCALE);

	//source to grayscale
	Mat g_1;
	cvtColor(original_img, g_1, COLOR_BGR2GRAY);
	imwrite("lab04.g1.png",g_1);
	
	//define roi
	Rect2d roi = { 203, 100, 2100, 3060 };

	//
	////  ### adaptiveThreshold Opencv ###
	//
	
	//binarization
	Mat b_1;
	adaptiveThreshold(g_1, b_1, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 31, 5);
	imwrite("lab04.b1.png", b_1);
	Mat diff_b1 = findDifference(etalon(roi), b_1(roi));
	//getStats(diff_b1);
	//cout <<"___________"<< endl;

	//filtering
	Mat f_1;
	medianBlur(b_1, f_1, 3);
	Mat diff_f1 = findDifference(etalon(roi), f_1(roi));
	imwrite("lab04.f1.png", f_1);
	//getStats(diff_f1);
	//cout << "___________" << endl;

	//connected components
	bitwise_not(f_1, f_1);
	Mat roi_f1 = f_1(roi);

	Mat labelImage, statsImage, centroidsImage;
	int nLabels = connectedComponentsWithStats(roi_f1, labelImage, statsImage, centroidsImage, 8);
	//cout << nLabels;

	Mat mask = Mat::zeros(roi_f1.size(), CV_8UC1);

	for (int i = 1; i < statsImage.rows; ++i) {
		
		if (statsImage.at<int>(Point(4, i)) > 15) 
		{
				mask += (labelImage == i);
		}		
	}

	//imwrite("lab04.mask1.png", mask);

	Mat v_1 = Mat::zeros(f_1.size(), CV_8UC1);
	Mat v_1_roi = v_1(roi);
	v_1_roi += mask;
	bitwise_not(v_1, v_1);
	imwrite("lab04.v1.png", v_1);
	Mat diff_v1 = findDifference(etalon(roi), v_1(roi));
	//getStats(diff_v1);
	//cout << "___________" << endl;


	//difference from etalon
	Mat e_1 = showDifference(etalon, v_1);
	imwrite("lab04.e1.png", e_1);

	
	//
	////  ### local binarization + Gauss window ###
	//

	//binarization
	Mat gb;
	GaussianBlur(g_1, gb, Size(91, 91), 0, 0);
	Mat m = cv::abs(gb - g_1);
	Mat m_gb;
	GaussianBlur(m, m_gb, Size(91, 91), 0, 0);

	Mat b_2 = Mat::zeros(g_1.size(), CV_32F);
	//b_2 = (gb - g_1) / m_gb+d_0;

	for (int i = 0; i < b_2.cols; ++i) {
		for (int j = 0; j < b_2.rows; ++j) {
			b_2.at<float>(j, i) = (float)
				(gb.at<uchar>(j, i) - g_1.at<uchar>(j, i)) / (m_gb.at<uchar>(j, i) + 2) <= 0.9 ? 1 : 0;
		}
	}

	b_2.convertTo(b_2, CV_8U, 255);
	imwrite("lab04.b2.png", b_2);
	Mat diff_b2 = findDifference(etalon(roi), b_2(roi));
	//getStats(diff_b2);
	//cout << "___________" << endl;

	//filtering
	Mat f_2;
	medianBlur(b_2, f_2, 3);
	imwrite("lab04.f2.png", f_2);
	Mat diff_f2 = findDifference(etalon(roi), f_2(roi));
	//getStats(diff_f2);
	//cout << "___________" << endl;

	//connected components
	bitwise_not(f_2, f_2);
	Mat roi_f2 = f_2(roi);
	nLabels = connectedComponentsWithStats(roi_f2, labelImage, statsImage, centroidsImage, 8);
	//cout << nLabels;


	mask = Mat::zeros(roi_f2.size(), CV_8UC1);

	for (int i = 1; i < statsImage.rows; ++i) {

		if (statsImage.at<int>(Point(4, i)) > 15)
		{
			mask += (labelImage == i);
		}
	}

	//imwrite("lab04.mask2.png", mask);

	Mat v_2 = Mat::zeros(f_2.size(), CV_8UC1);
	Mat v_2_roi = v_2(roi);
	v_2_roi += mask;
	bitwise_not(v_2, v_2);
	imwrite("lab04.v2.png", v_2);
	Mat diff_v2 = findDifference(etalon(roi), v_2(roi));
	//getStats(diff_v2);
	//cout << "___________" << endl;

	//difference from etalon
	Mat e_2 = showDifference(etalon, v_2);
	imwrite("lab04.e2.png", e_2);
	
	waitKey(0); 
	return 0;
}