#include "stdafx.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <queue>
#include <random>

using namespace std;

float meanIntensityValue(Mat src) {
	int M = src.rows * src.cols;
	float mean_intensity = 0;
	int sum = 0;

	for (int i = 0; i < src.rows - 1; ++i) {
		for (int j = 0; j < src.cols - 1; ++j) {
			sum += src.at<uchar>(i, j);
		}
	}

	mean_intensity = (float)sum / M;

	return mean_intensity;
}
float standard_deviation(Mat src) {
	int M = src.rows * src.cols;
	float mean = meanIntensityValue(src);
	float deviation = 0;

	for (int i = 0; i < src.rows - 1; ++i) {
		for (int j = 0; j < src.cols - 1; ++j) {
			deviation += pow((src.at<uchar>(i, j) - mean), 2);
		}
	}

	deviation /= (float)M;
	return sqrt(deviation);
}
vector<Mat> splitAndReturn(Mat src) {

	vector<Mat> matrices;
	for (int i = 0; i < 3; i++) {
		Mat matrix = Mat::zeros(src.rows, src.cols, CV_8UC1);
		matrices.push_back(matrix);
	}
	
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3b pixel = src.at<Vec3b>(i, j);
			uchar r = pixel[2];
			uchar g = pixel[1];
			uchar b = pixel[0];
			matrices.at(0).at<uchar>(i, j) = b;
			matrices.at(1).at<uchar>(i, j) = g;
			matrices.at(2).at<uchar>(i, j) = r;
		}
	}
	return matrices;
}
float sigmoid(float x) {
	float k = 5;
	float b = 0.5;
	float c = 0.5;
	float firstHalf = (float)1 / (1 + exp(x*(-k) + b) + c);
	float secondHalf = (float)1 / (1 + c);
	float result = firstHalf * secondHalf;
	return result;
}
Mat applySigmoid(Mat gaussian, Mat source)
{
	Mat sigmoid = Mat::zeros(gaussian.rows, gaussian.cols, CV_32FC1);
	for (int i = 0; i < gaussian.rows; i++)
	{
		for (int j = 0 ; j < gaussian.cols; j++)
		{
			uchar sourcePixel = source.at<uchar>(i, j);
			uchar gaussPixel = gaussian.at<uchar>(i, j);
			sigmoid.at<float>(i, j) = ::sigmoid((float)sourcePixel / (float)gaussPixel);
		}
	}
	return sigmoid;
}
Mat normalise(Mat src)
{
	Mat dest = Mat::zeros(src.rows, src.cols, CV_8UC1);
	float max = FLT_MIN, min = FLT_MAX;
	for (int i = 0; i<src.rows; i++)
	{
		for (int j = 0; j<src.cols; j++)
		{
			if (src.at<float>(i, j) > max) {
				if (src.at<float>(i, j) != numeric_limits<float>::infinity())
					max = src.at<float>(i, j);
			}

			if (src.at<float>(i, j) < min) {
				if (src.at<float>(i, j) != -numeric_limits<float>::infinity())	
					min = src.at<float>(i, j);
			}

		}

	}

	for (int i = 0; i<src.rows; i++)
	{
		for (int j = 0; j<src.cols; j++)
		{
			if (max == min)
			{
				max = min + 0.0000000000000001;
			}
				dest.at<uchar>(i, j) = ceil((src.at<float>(i, j) - min) * 255 / (max - min));
		}
	}

	return dest;
}
float findMaxima(vector<Mat>GaussianChannels, int i, int j)
{
	float max = FLT_MIN;
	for (int k = 0 ; k < 3; k++)
	{
		if (GaussianChannels.at(k).at<float>(i, j) > max)
			if (GaussianChannels.at(k).at<float>(i, j) != numeric_limits<float>::infinity())
				max = GaussianChannels.at(k).at<float>(i, j);
	}
	return max;
}
Mat denormalise(Mat src)
{
	Mat dest = Mat::zeros(src.rows, src.cols, CV_32FC1);
	float max = INT_MIN, min = INT_MAX;
	for (int i = 0; i<src.rows; i++)
	{
		for (int j = 0; j<src.cols; j++)
		{
			if (src.at<uchar>(i, j) > max) {
				if (src.at<uchar>(i, j) != numeric_limits<uchar>::infinity())
					max = src.at<uchar>(i, j);
			}

			if (src.at<uchar>(i, j) < min) {
				if (src.at<uchar>(i, j) != -numeric_limits<uchar>::infinity())
					min = src.at<uchar>(i, j);
			}

		}

	}

	for (int i = 0; i<src.rows; i++)
	{
		for (int j = 0; j<src.cols; j++)
		{
			if (max == min)
			{
				max = min + 0.0000000000000001;
			}
			dest.at<float>(i, j) = (src.at<uchar>(i, j) - min) * 1 / (max - min);
		}
	}

	return dest;
}
Mat compose(Mat red, Mat blue, Mat green)
{
	Mat composed = Mat::zeros(red.rows, red.cols, CV_32FC3);
	for (int i = 0 ; i < red.rows; i++)
	{
		for (int j = 0 ; j < red.cols; j++)
		{
			Vec3f pixel;
			pixel[2] = red.at<float>(i, j);
			pixel[1] = green.at<float>(i, j);
			pixel[0] = blue.at<float>(i, j);
			composed.at<Vec3f>(i, j) = pixel;
		}
	}
	return composed;
}
Mat composeAndNormalise(Mat red, Mat blue, Mat green)
{
	Mat composed = Mat::zeros(red.rows, red.cols, CV_8UC3);
	Mat redNormalised = normalise(red);
	Mat greenNormalised = normalise(green);
	Mat blueNormalised = normalise(blue);


	for (int i = 0; i < red.rows; i++)
	{
		for (int j = 0; j < red.cols; j++)
		{
			Vec3b pixel;
			pixel[2] = redNormalised.at<uchar>(i, j);
			pixel[1] = greenNormalised.at<uchar>(i, j);
			pixel[0] = blueNormalised.at<uchar>(i, j);
			composed.at<Vec3b>(i, j) = pixel;
		}
	}
	return composed;



}
Mat applyWeight(Mat src, Mat gaussian, Mat sigmoid)
{
	Mat dest = Mat::zeros(src.rows, src.cols, CV_32FC1);
	for ( int i = 0 ; i < src.rows; i++)
	{
		for (int j = 0 ; j < src.cols ; j++)
		{
			float weight = 1 - pow(1 - gaussian.at<float>(i, j), 20);
			dest.at<float>(i, j) = sigmoid.at<float>(i, j)*weight + src.at<float>(i, j)*(1 - weight);
		}
	}
	return dest;
}
Mat applyFinalWeight(Mat src, Mat gaussian, Mat sigmoid, vector<Mat>GaussianChannels)
{
	Mat dest = Mat::zeros(src.rows, src.cols, CV_32FC1);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			float weight2 = 1 - pow(findMaxima(GaussianChannels, i , j), 0.5);
			float weight = 1 - pow(1 - gaussian.at<float>(i, j), 20);
			float finalWeight = weight2 * weight;
			dest.at<float>(i, j) = sigmoid.at<float>(i, j)*finalWeight + src.at<float>(i, j)*(1 - finalWeight);
		}
	}
	return dest;
}
void lowLightEnhancement(){
	
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		
		Mat src = imread(fname, CV_LOAD_IMAGE_COLOR);
		vector<Mat> srcChannels = splitAndReturn(src);
		Mat gaussian;
		
		GaussianBlur(src, gaussian, Size(standard_deviation(src)/3, standard_deviation(src)/3), standard_deviation(src) ,0);
		
		vector<Mat> gaussianChannels = splitAndReturn(gaussian);

		Mat denormalisedRedGaussian = denormalise(gaussianChannels.at(2));
		Mat denormalisedBlueGaussian = denormalise(gaussianChannels.at(0));
		Mat denormalisedGreenGaussian = denormalise(gaussianChannels.at(1));
		
		Mat denormalisedRedSource = denormalise(srcChannels.at(2));
		Mat denormalisedGreenSource = denormalise(srcChannels.at(1));
		Mat denormalisedBlueSource = denormalise(srcChannels.at(0));

		vector<Mat> denormalisedGaussianChannels;
		denormalisedGaussianChannels.push_back(denormalisedBlueGaussian);
		denormalisedGaussianChannels.push_back(denormalisedGreenGaussian);
		denormalisedGaussianChannels.push_back(denormalisedRedGaussian);

		Mat sigmoidRed = applySigmoid(gaussianChannels.at(2), srcChannels.at(2));
		Mat sigmoidBlue = applySigmoid(gaussianChannels.at(0), srcChannels.at(0));
		Mat sigmoidGreen = applySigmoid(gaussianChannels.at(1), srcChannels.at(1));
		
		Mat color = compose(sigmoidRed, sigmoidBlue, sigmoidGreen);
		Mat normalizedColor = composeAndNormalise(sigmoidRed, sigmoidBlue, sigmoidGreen);
		
		Mat noiseReducedRed = applyWeight(denormalisedRedSource, denormalisedRedGaussian, sigmoidRed);
		Mat noiseReduceBlue = applyWeight(denormalisedBlueSource, denormalisedBlueGaussian, sigmoidBlue);
		Mat noiseReduceGreen = applyWeight(denormalisedGreenSource, denormalisedGreenGaussian, sigmoidGreen);
		
		Mat noiseReducedColor = compose(noiseReducedRed, noiseReduceBlue, noiseReduceGreen);
		Mat noiseReducedNormalised = composeAndNormalise(noiseReducedRed, noiseReduceBlue, noiseReduceGreen);

		Mat preserveRed = applyFinalWeight(denormalisedRedSource, denormalisedRedGaussian, sigmoidRed,denormalisedGaussianChannels);
		Mat preserveBlue = applyFinalWeight(denormalisedBlueSource, denormalisedBlueGaussian, sigmoidBlue, denormalisedGaussianChannels);
		Mat preserveGreen = applyFinalWeight(denormalisedGreenSource, denormalisedGreenGaussian, sigmoidGreen, denormalisedGaussianChannels);

		Mat preservedColor = compose(preserveRed, preserveBlue, preserveGreen);
		Mat preservedColorNormalised = composeAndNormalise(preserveRed, preserveBlue, preserveGreen);
		imshow("Source", src);
		imshow("Result", preservedColorNormalised);
		waitKey(0);
	}
}

int main()
{
	lowLightEnhancement();
	return 0;
}