#include <opencv2/opencv.hpp>
#include <fstream>
#include "json.hpp"

using namespace std;
using namespace cv;

using json = nlohmann::json;

Mat findDifference(Mat etalon, Mat example) {

	Mat stats = Mat::zeros(Size(2, 3), CV_32F);

	//TP, FP, FN, TN
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
                    diff_img.at<Vec3b>(j, i) = { 0, 255, 0 };
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

Mat getEtalonHomography(Mat scan, Mat photo, json scan_borders, json photo_borders) {

    //scan borders
    string file_name_scan = scan_borders["_via_image_id_list"][0];

    vector<Point> scan_points;
    for (int i = 0; i < scan_borders["_via_img_metadata"][file_name_scan]["regions"].size(); i++) {
        for (int j = 0; j < scan_borders["_via_img_metadata"][file_name_scan]["regions"][i]["shape_attributes"]["all_points_x"].size(); j++) {

            int x = scan_borders["_via_img_metadata"][file_name_scan]["regions"][i]["shape_attributes"]["all_points_x"][j];
            int y = scan_borders["_via_img_metadata"][file_name_scan]["regions"][i]["shape_attributes"]["all_points_y"][j];

            scan_points.push_back(Point(x, y));
        }
    }


    //photo borders
    string file_name_photo = photo_borders["_via_image_id_list"][0];

    vector<Point> example_points;
    for (int i = 0; i < photo_borders["_via_img_metadata"][file_name_photo]["regions"].size(); i++) {
        for (int j = 0; j < photo_borders["_via_img_metadata"][file_name_photo]["regions"][i]["shape_attributes"]["all_points_x"].size(); j++) {

            int x = photo_borders["_via_img_metadata"][file_name_photo]["regions"][i]["shape_attributes"]["all_points_x"][j];
            int y = photo_borders["_via_img_metadata"][file_name_photo]["regions"][i]["shape_attributes"]["all_points_y"][j];

            example_points.push_back(Point(x, y));
        }
    }

    //get homography
    Mat scan_inv = 255 - scan;
    Mat homography = findHomography(scan_points, example_points);
    Mat res_H(photo.size(), CV_8U);
    warpPerspective(scan_inv, res_H, homography, photo.size());
    res_H = 255 - res_H;

    return res_H;
}

Mat binarizeImage(Mat src) {

    //binarization
    Mat gb;
    GaussianBlur(src, gb, Size(71, 71), 0, 0);
    Mat m = cv::abs(gb - src);
    Mat m_gb;
    GaussianBlur(m, m_gb, Size(71, 71), 0, 0);

    Mat bin = Mat::zeros(src.size(), CV_32F);


    for (int i = 0; i < bin.cols; ++i) {
        for (int j = 0; j < bin.rows; ++j) {
            bin.at<float>(j, i) = (float)
                (gb.at<uchar>(j, i) - src.at<uchar>(j, i)) / (m_gb.at<uchar>(j, i) + 60) < 0.15 ? 1 : 0;
        }
    }

    bin.convertTo(bin, CV_8U, 255);

    //filtering
    Mat f_2;
    medianBlur(bin, f_2, 3);


    //connected components
    bitwise_not(f_2, f_2);

    Mat labelImage, statsImage, centroidsImage;
    int nLabels = connectedComponentsWithStats(f_2, labelImage, statsImage, centroidsImage, 8);
    cout << nLabels<<endl;


    Mat mask = Mat::zeros(f_2.size(), CV_8UC1);
    for (int i = 1; i < statsImage.rows; ++i) {

        if ((statsImage.at<int>(Point(CC_STAT_AREA, i)) > 20) & (statsImage.at<int>(Point(CC_STAT_AREA, i)) < 750) &
            (statsImage.at<int>(Point(CC_STAT_HEIGHT, i)) > 4) & (statsImage.at<int>(Point(CC_STAT_HEIGHT, i)) < 200) & 
            (statsImage.at<int>(Point(CC_STAT_WIDTH, i)) > 3) & (statsImage.at<int>(Point(CC_STAT_WIDTH, i)) < 80))
           
        {
            mask += (labelImage == i);
        }
    }

   
    return mask;

}

int main() {

    //load scan with borders
    Mat scan_image = imread("scan.png", IMREAD_GRAYSCALE);
    ifstream scan_borders_stream("scan_borders.json");
    json scan_borders;
    scan_borders_stream >> scan_borders;

    //load photos 
    Mat photo_1 = imread("P_20210309_064128_vHDR_Auto.jpg", IMREAD_GRAYSCALE);
    Mat photo_2 = imread("P_20210309_064141_vHDR_Auto.jpg", IMREAD_GRAYSCALE);
    Mat photo_3 = imread("P_20210410_170813_vHDR_On.jpg", IMREAD_GRAYSCALE);
    Mat photo_4 = imread("P_20210410_171055_vHDR_On.jpg", IMREAD_GRAYSCALE);
    Mat photo_5 = imread("P_20210410_171334_vHDR_On.jpg", IMREAD_GRAYSCALE);

    //laod borders
    ifstream photo_1_borders_stream("P_20210309_064128_vHDR_Auto.json");
    ifstream photo_2_borders_stream("P_20210309_064141_vHDR_Auto.json");
    ifstream photo_3_borders_stream("P_20210410_170813_vHDR_On.json");
    ifstream photo_4_borders_stream("P_20210410_171055_vHDR_On.json");
    ifstream photo_5_borders_stream("P_20210410_171334_vHDR_On.json");

    json photo_1_borders;
    json photo_2_borders;
    json photo_3_borders;
    json photo_4_borders;
    json photo_5_borders;

    photo_1_borders_stream >> photo_1_borders;
    photo_2_borders_stream >> photo_2_borders;
    photo_3_borders_stream >> photo_3_borders;
    photo_4_borders_stream >> photo_4_borders;
    photo_5_borders_stream >> photo_5_borders;

    //Get etalon gor photos
    Mat etalon_photo_1 = getEtalonHomography(scan_image, photo_1, scan_borders, photo_1_borders);
    Mat etalon_photo_2 = getEtalonHomography(scan_image, photo_2, scan_borders, photo_2_borders);
    Mat etalon_photo_3 = getEtalonHomography(scan_image, photo_3, scan_borders, photo_3_borders);
    Mat etalon_photo_4 = getEtalonHomography(scan_image, photo_4, scan_borders, photo_4_borders);
    Mat etalon_photo_5 = getEtalonHomography(scan_image, photo_5, scan_borders, photo_5_borders);


    imwrite("etalon_photo_1.png", etalon_photo_1);
    imwrite("etalon_photo_2.png", etalon_photo_2);
    imwrite("etalon_photo_3.png", etalon_photo_3);
    imwrite("etalon_photo_4.png", etalon_photo_4);
    imwrite("etalon_photo_5.png", etalon_photo_5);

    //Define roi
    Rect2d roi_1 = { 250, 150, 2050, 3000 };
    Rect2d roi_2 = { 203, 100, 2100, 3060 };
    Rect2d roi_3 = { 420, 630, 1590, 2260 };
    Rect2d roi_4 = { 900, 200, 2100, 1800 };
    Rect2d roi_5 = { 565, 340, 2480, 1700 };

    Mat roi_img_1 = photo_1(roi_1);
    Mat roi_img_2 = photo_2(roi_2);
    Mat roi_img_3 = photo_3(roi_3);
    Mat roi_img_4 = photo_4(roi_4);
    Mat roi_img_5 = photo_5(roi_5);

    //binarization
    Mat res_bin_1 = binarizeImage(roi_img_1);
    Mat res_bin_2 = binarizeImage(roi_img_2);
    Mat res_bin_3 = binarizeImage(roi_img_3);
    Mat res_bin_4 = binarizeImage(roi_img_4);
    Mat res_bin_5 = binarizeImage(roi_img_5);


    Mat photo_bin_1 = Mat::zeros(photo_1.size(), CV_8U);
    Mat photo_bin_2 = Mat::zeros(photo_2.size(), CV_8U);
    Mat photo_bin_3 = Mat::zeros(photo_3.size(), CV_8U);
    Mat photo_bin_4 = Mat::zeros(photo_4.size(), CV_8U);
    Mat photo_bin_5 = Mat::zeros(photo_5.size(), CV_8U);

    Mat photo_bin_roi_1 = photo_bin_1(roi_1);
    Mat photo_bin_roi_2 = photo_bin_2(roi_2);
    Mat photo_bin_roi_3 = photo_bin_3(roi_3);
    Mat photo_bin_roi_4 = photo_bin_4(roi_4);
    Mat photo_bin_roi_5 = photo_bin_5(roi_5);


    photo_bin_roi_1 += res_bin_1;
    photo_bin_roi_2 += res_bin_2;
    photo_bin_roi_3 += res_bin_3;
    photo_bin_roi_4 += res_bin_4;
    photo_bin_roi_5 += res_bin_5;

    bitwise_not(photo_bin_1, photo_bin_1);
    bitwise_not(photo_bin_2, photo_bin_2);
    bitwise_not(photo_bin_3, photo_bin_3);
    bitwise_not(photo_bin_4, photo_bin_4);
    bitwise_not(photo_bin_5, photo_bin_5);

    //Get statistics
    Mat stats_photo_1 = findDifference(etalon_photo_1(roi_1), photo_bin_1(roi_1));
    Mat stats_photo_2 = findDifference(etalon_photo_2(roi_2), photo_bin_2(roi_2));
    Mat stats_photo_3 = findDifference(etalon_photo_3(roi_3), photo_bin_3(roi_3));
    Mat stats_photo_4 = findDifference(etalon_photo_4(roi_4), photo_bin_4(roi_4));
    Mat stats_photo_5 = findDifference(etalon_photo_5(roi_5), photo_bin_5(roi_5));

    getStats(stats_photo_1);
    getStats(stats_photo_2);
    getStats(stats_photo_3);
    getStats(stats_photo_4);
    getStats(stats_photo_5);

    //Diff images
    Mat diff_1 = showDifference(etalon_photo_1, photo_bin_1);
    Mat diff_2 = showDifference(etalon_photo_2, photo_bin_2);
    Mat diff_3 = showDifference(etalon_photo_3, photo_bin_3);
    Mat diff_4 = showDifference(etalon_photo_4, photo_bin_4);
    Mat diff_5 = showDifference(etalon_photo_5, photo_bin_5);

    //imwrite("res_bin_1.png", photo_bin_1);
    //imwrite("res_bin_2.png", photo_bin_2);
    //imwrite("res_bin_3.png", photo_bin_3);
    //imwrite("res_bin_4.png", photo_bin_4);
    //imwrite("res_bin_5.png", photo_bin_5);

    //imwrite("diff_1.png", diff_1);
    //imwrite("diff_2.png", diff_2);
    //imwrite("diff_3.png", diff_3);
    //imwrite("diff_4.png", diff_4);
    //imwrite("diff_5.png", diff_5);

    imwrite("res_bin_1_v2.png", photo_bin_1);
    imwrite("res_bin_2_v2.png", photo_bin_2);
    imwrite("res_bin_3_v2.png", photo_bin_3);
    imwrite("res_bin_4_v2.png", photo_bin_4);
    imwrite("res_bin_5_v2.png", photo_bin_5);


    imwrite("diff_1_v2.png", diff_1);
    imwrite("diff_2_v2.png", diff_2);
    imwrite("diff_3_v2.png", diff_3);
    imwrite("diff_4_v2.png", diff_4);
    imwrite("diff_5_v2.png", diff_5);


    waitKey(0);
    return 0;
}