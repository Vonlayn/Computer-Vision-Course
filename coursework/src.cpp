#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


/*
Proposed algorithm:
1. Load Grayscale image with poor lighting conditions
2. Opening by reconstruction(parameters: structural element size -- int)
3. Calculate SSIM, PSNR
*/

Scalar getMSSIM(const Mat& i1, const Mat& i2) {
	const double C1 = 6.5025, C2 = 58.5225;
	/***************************** INITS **********************************/
	int d = CV_32F;
	Mat I1, I2;
	i1.convertTo(I1, d);            // cannot calculate on one byte large values
	i2.convertTo(I2, d);
	Mat I2_2 = I2.mul(I2);        // I2^2
	Mat I1_2 = I1.mul(I1);        // I1^2
	Mat I1_I2 = I1.mul(I2);        // I1 * I2
	/*************************** END INITS **********************************/

	Mat mu1, mu2;                   // PRELIMINARY COMPUTING
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);

	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);

	Mat sigma1_2, sigma2_2, sigma12;
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;
	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);   // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
	Mat ssim_map;
	divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;
	Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
	return mssim;
}


double getPSNR(const Mat& I1, const Mat& I2) {
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2
	Scalar s = sum(s1);        // sum elements per channel
	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0 * log10((255 * 255) / mse);
		return psnr;
	}
}


Mat Draw_histogram(Mat src)
{
	//make empty template
	Mat res(src.rows, src.cols * 2, CV_8UC3);

	//define histogram parameters
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true, accumulate = false;

	Mat hist;
	calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	//make image for histogram
	int hist_w = 256, hist_h = 256;
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	//normolize and draw histogram
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());


	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(i, hist_h - cvRound(hist.at<float>(i - 1))),
			Point((i), hist_h),
			Scalar(247, 115, 80), 2, 8, 0);
	}
	line(histImage, Point(0, 0), Point(0, 255), Scalar(0, 0, 255));


	Mat numbers_img(8, histSize, CV_8UC3, Scalar::all(0));
	Rect2d rc = { 0, 0, 256, 8 };

	for (int i = 0; i < 256; ++i) {
		for (int j = rc.y; j < rc.y + rc.height; ++j) {
			line(numbers_img, Point2d(i, j), Point2d(i  + 2, j), Scalar::all(i), 1);
		}
	}

	putText(numbers_img, "0", Point(0, 7), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 180, 249));
	putText(numbers_img, "255", Point(238, 7), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 180, 249), 1);
	line(numbers_img, Point(0, 0), Point(255, 0), Scalar(0, 0, 255));

	vconcat(histImage, numbers_img, histImage);
	
	return histImage;
}


Mat geodesicDilation(Mat src, Mat mask, Mat struct_elem, int power = 1) {

	Mat res = Mat::zeros(src.size(), CV_8U);
	Mat temp = src.clone();

	
	while (!checkRange(temp - res, true, (Point*)0, 0, 1)) {
		res = temp.clone();
		dilate(temp, temp, struct_elem, Point(-1, -1));
		temp = temp & mask;
	}

	return res;

}

Mat openingByReconstructionMethod(Mat src, int struct_elem ) {

	Mat source_img = src.clone();
	int erosion_size = struct_elem;
	int dilation_size = struct_elem;


	//Morphological transformations
	Mat erode_img;
	Mat erode_element = getStructuringElement(0,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(-1, -1));

	erode(source_img, erode_img, erode_element, Point(-1, -1));

	Mat dilation_element = getStructuringElement(0,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(-1, -1));

	Mat mask_source = (source_img > 0);
	Mat b = geodesicDilation(erode_img, mask_source, dilation_element);

	erode_element = getStructuringElement(0,
		Size(3, 3),
		Point(-1, -1));
	erode(b, b, erode_element, Point(-1, -1));

	imwrite("background.png", b);
	//return b;
	Mat k_param = Mat::zeros(source_img.size(), CV_8U);

	for (int cols = 0; cols < k_param.cols; cols++)
	{
		for (int rows = 0; rows < k_param.rows; rows++)
		{
			k_param.at<uchar>(rows, cols) = round((255 - b.at<uchar>(rows, cols)) / log10(256));
		}
	}

	////Apply Weber's Law
	Mat weber_img = Mat::zeros(source_img.size(), CV_8U);

	for (int cols = 0; cols < weber_img.cols; cols++)
	{
		for (int rows = 0; rows < weber_img.rows; rows++)
		{
			weber_img.at<uchar>(rows, cols) = round(k_param.at<uchar>(rows, cols) * log10(source_img.at<uchar>(rows, cols) + 1) + b.at<uchar>(rows, cols));
		}
	}

	return weber_img;

}



int main() {

	String filename_example= "example00659.png";
	String filename_etalon = "normal00659.png";

	//Load Grayscale image 
	string image_path_src = samples::findFile(filename_example);
	Mat example_img = imread(image_path_src, IMREAD_GRAYSCALE);
	Mat hist_example = Draw_histogram(example_img);

	//load etalon image
	image_path_src = samples::findFile(filename_etalon);
	Mat etalon_img = imread(image_path_src, IMREAD_GRAYSCALE);
	Mat hist_etalon = Draw_histogram(etalon_img);

	Scalar mssim_before = getMSSIM(etalon_img,example_img);
	double psnr_before = getPSNR(etalon_img, example_img);
	cout <<"MSSIM before: "<< mssim_before[0] << endl;
	cout << "PNSR before: " << psnr_before << endl;

	//save grayscale soure images + their histograms before method
	//imwrite("GS"+ filename_etalon, etalon_img);
	//imwrite("GS"+ filename_example, example_img);
	//imwrite("HS"+ filename_etalon, hist_etalon);
	//imwrite("HS"+ filename_example, hist_example);

	//opening by reconstruction test
	//Mat obr_img_1 = openingByReconstructionMethod(example_img, 1);
	//Mat obr_img_2 = openingByReconstructionMethod(example_img, 2);
	//Mat obr_img_3 = openingByReconstructionMethod(example_img, 3);
	//Mat obr_img_5 = openingByReconstructionMethod(example_img, 5);
	Mat obr_img_10 = openingByReconstructionMethod(example_img, 10);
	Mat obr_img_20 = openingByReconstructionMethod(example_img, 20);
	//Mat obr_img_25 = openingByReconstructionMethod(example_img, 25);
	//Mat obr_img_30 = openingByReconstructionMethod(example_img, 30);

	Mat obr_img_40 = openingByReconstructionMethod(example_img, 40);
		//Mat obr_img_50 = openingByReconstructionMethod(example_img, 50);
	Mat obr_img_60 = openingByReconstructionMethod(example_img, 60);
	//Mat obr_img_70 = openingByReconstructionMethod(example_img, 70);
	Mat obr_img_80 = openingByReconstructionMethod(example_img, 80);
	//Mat obr_img_90 = openingByReconstructionMethod(example_img, 90);
	Mat obr_img_100 = openingByReconstructionMethod(example_img, 100);
	


	//Mat obr_img_110 = openingByReconstructionMethod(example_img, 110);
	//Mat obr_img_120 = openingByReconstructionMethod(example_img, 120);
	//Mat obr_img_130 = openingByReconstructionMethod(example_img, 130);
	//Mat obr_img_140 = openingByReconstructionMethod(example_img, 140);
	//Mat obr_img_150 = openingByReconstructionMethod(example_img, 150);
	//Mat obr_img_160 = openingByReconstructionMethod(example_img, 160);
	//Mat obr_img_170 = openingByReconstructionMethod(example_img, 170);
	//Mat obr_img_180 = openingByReconstructionMethod(example_img, 180);
	//Mat obr_img_190 = openingByReconstructionMethod(example_img, 190);
	//Mat obr_img_210 = openingByReconstructionMethod(example_img, 210);


	//Mat obr_hist_1 = Draw_histogram(obr_img_1);
	//Mat obr_hist_2 = Draw_histogram(obr_img_2);
	//Mat obr_hist_3 = Draw_histogram(obr_img_3);
	//Mat obr_hist_5 = Draw_histogram(obr_img_5);
	Mat obr_hist_10 = Draw_histogram(obr_img_10);
	Mat obr_hist_20 = Draw_histogram(obr_img_20);
	//Mat obr_hist_25 = Draw_histogram(obr_img_25);
	//Mat obr_hist_30 = Draw_histogram(obr_img_30);


	Mat obr_hist_40 = Draw_histogram(obr_img_40);
		//Mat obr_hist_50 = Draw_histogram(obr_img_50);
	Mat obr_hist_60 = Draw_histogram(obr_img_60);
	//Mat obr_hist_70 = Draw_histogram(obr_img_70);
	Mat obr_hist_80 = Draw_histogram(obr_img_80);
	//Mat obr_hist_90 = Draw_histogram(obr_img_90);
	Mat obr_hist_100 = Draw_histogram(obr_img_100);


	//Mat obr_hist_110 = Draw_histogram(obr_img_110);
	//Mat obr_hist_120 = Draw_histogram(obr_img_120);
	//Mat obr_hist_130 = Draw_histogram(obr_img_130);
	//Mat obr_hist_140 = Draw_histogram(obr_img_140);
	//Mat obr_hist_150 = Draw_histogram(obr_img_150);
	//Mat obr_hist_160 = Draw_histogram(obr_img_160);
	//Mat obr_hist_170 = Draw_histogram(obr_img_170);
	//Mat obr_hist_180 = Draw_histogram(obr_img_180);
	//Mat obr_hist_190 = Draw_histogram(obr_img_190);
	//Mat obr_hist_210 = Draw_histogram(obr_img_210);


	Scalar mssim_10 = getMSSIM(etalon_img, obr_img_10);
	double psnr_10 = getPSNR(etalon_img, obr_img_10);
	cout << "MSSIM_10: " << mssim_10[0] << endl;
	cout << "PNSR_10: " << psnr_10 << endl;

	Scalar mssim_20 = getMSSIM(etalon_img, obr_img_20);
	double psnr_20 = getPSNR(etalon_img, obr_img_20);
	cout << "MSSIM_20: " << mssim_20[0] << endl;
	cout << "PNSR_20: " << psnr_20 << endl;

	Scalar mssim_40 = getMSSIM(etalon_img, obr_img_40);
	double psnr_40 = getPSNR(etalon_img, obr_img_40);
	cout << "MSSIM_40: " << mssim_40[0] << endl;
	cout << "PNSR_40: " << psnr_40 << endl;

	Scalar mssim_60 = getMSSIM(etalon_img, obr_img_60);
	double psnr_60 = getPSNR(etalon_img, obr_img_60);
	cout << "MSSIM_60: " << mssim_60[0] << endl;
	cout << "PNSR_60: " << psnr_60 << endl;

	Scalar mssim_80 = getMSSIM(etalon_img, obr_img_80);
	double psnr_80 = getPSNR(etalon_img, obr_img_80);
	cout << "MSSIM_80: " << mssim_80[0] << endl;
	cout << "PNSR_80: " << psnr_80 << endl;

	Scalar mssim_100 = getMSSIM(etalon_img, obr_img_100);
	double psnr_100 = getPSNR(etalon_img, obr_img_100);
	cout << "MSSIM_100: " << mssim_100[0] << endl;
	cout << "PNSR_100: " << psnr_100 << endl;


	//imwrite(filename_example+"_obr_img_50.png",obr_img_10);
	//imwrite(filename_example+"_obr_img_50_hist.png", obr_hist_10);


 	waitKey(0);
	return 0;
}