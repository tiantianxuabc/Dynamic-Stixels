#include <iostream>
#include <opencv.hpp>

#include "stixel_world.h"

#include <stdint.h>
#include <stdio.h>


#include <bitset>
#include <smmintrin.h> // intrinsics
#include <emmintrin.h>
#include<fstream>
#include<stdio.h>
#include "StereoBMHelper.h"
#include "FastFilters.h"
#include "StereoSGM.h"


#include<time.h>

#include <vector>
#include <list>
#include <algorithm>
#include <numeric>
#include <iostream>


#ifdef _DEBUG
#pragma comment(lib, "opencv_world400d.lib")
#else
#pragma comment(lib, "opencv_world400.lib")
#endif

const int dispRange = 128;


typedef struct MyStruct_stereo
{
	float f;
	float cu;
	float cv;
	float base;
	float pitch;
	float height;
}StereoRigParam;

void worldCoorFilter(cv::Mat& dispImg, const StereoRigParam &stereoRigParam, const float tooCloseThreshold = 0.05, const float tooHighThreshold = 2.5)
{
	cv::Point3f pixelWorld;
	uchar* rowPointer;
	float  tmp_Disp_Value;
	for (int v = 0; v < dispImg.rows; v++)
	{
		rowPointer = dispImg.ptr<uchar>(v);

		for (int u = 0; u < dispImg.cols; u++)
		{
			tmp_Disp_Value = rowPointer[u];
			pixelWorld.x = (u - stereoRigParam.cu) * stereoRigParam.base / tmp_Disp_Value;
			pixelWorld.y = stereoRigParam.height - (v - stereoRigParam.cv) * stereoRigParam.base / tmp_Disp_Value;
			pixelWorld.z = stereoRigParam.f * stereoRigParam.base / tmp_Disp_Value;

			if ((pixelWorld.y < tooCloseThreshold) /*|| (pixelWorld.y > tooHighThreshold)*/)
			{
				rowPointer[u] = 63;
			}
			else if (pixelWorld.y > tooHighThreshold)
			{
				rowPointer[u] = 0;
			}
		}
	}
}


template<typename T, typename U, typename V>
inline cv::Scalar cvJetColourMat(T v, U vmin, V vmax) {
	cv::Scalar c = cv::Scalar(1.0, 1.0, 1.0);  // white
	T dv;

	if (v < vmin)
		v = vmin;
	if (v > vmax)
		v = vmax;
	dv = vmax - vmin;

	if (v < (vmin + 0.25 * dv)) {
		c.val[0] = 0;
		c.val[1] = 4 * (v - vmin) / dv;
	}
	else if (v < (vmin + 0.5 * dv)) {
		c.val[0] = 0;
		c.val[2] = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
	}
	else if (v < (vmin + 0.75 * dv)) {
		c.val[0] = 4 * (v - vmin - 0.5 * dv) / dv;
		c.val[2] = 0;
	}
	else {
		c.val[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
		c.val[2] = 0;
	}
	return c;
}

void computeVDisparity(cv::Mat &imgDisp, cv::Mat &vDisp, const int &dispRange_ = 128)
{
	 
	vDisp.create(cv::Size(dispRange, imgDisp.rows),CV_8UC1);
	uint8 tmp = 0;
	int dispMax = 0;
	for (int i = 0; i < imgDisp.rows; i++)
	{
		uint8 a[dispRange] = { 0 };
		for (int j = 0; j < imgDisp.cols; j++)
		{
			if ((tmp = *(imgDisp.data + i*imgDisp.step + j*imgDisp.elemSize())) < dispRange)
			{
				if (a[tmp] < 255)
				{
					a[tmp] = ++a[tmp];
				}
			}
			else
			{
				if (a[dispRange - 1] < 255)
				{
					a[dispRange - 1] = ++a[dispRange - 1];
				}
			}
		}

		for (int k = 0; k < dispRange; k++)
		{
			*(vDisp.data + i*vDisp.step + k*vDisp.elemSize()) = a[k];
		}
	}
	cv::imshow("VDis", vDisp);
	cv::waitKey(1);
	//cv::imwrite("E:\\vDisp.png", vDisp);
}


template<typename T>  /*0000*/
void processCensus5x5SGM(T* leftImg, T* rightImg, float32* output, float32* dispImgRight,
	int width, int height, uint16 paths, const int dispCount)
{
	const int maxDisp = dispCount - 1;

	//std::cout << std::endl << paths << ", " << dispCount << std::endl;

	// get memory and init sgm params
	uint32* leftImgCensus = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);
	uint32* rightImgCensus = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);

	StereoSGMParams_t params;
	params.lrCheck = true;
	params.MedianFilter = true;
	params.Paths = paths;

	params.NoPasses = 2;

	
	uint16* dsi = (uint16*)_mm_malloc(width*height*(maxDisp + 1)*sizeof(uint16), 32);
	StereoSGM<T> m_sgm16(width, height, maxDisp, params);


	census5x5_16bit_SSE(leftImg, leftImgCensus, width, height);
	census5x5_16bit_SSE(rightImg, rightImgCensus, width, height);
	costMeasureCensus5x5_xyd_SSE(leftImgCensus, rightImgCensus, height, width, dispCount, params.InvalidDispCost, dsi);

	m_sgm16.process(dsi, leftImg, output, dispImgRight);
	_mm_free(dsi);
}

void onMouse(int event, int x, int y, int flags, void *param)
{
	cv::Mat *im = reinterpret_cast<cv::Mat *>(param);
	switch (event)
	{
	case cv::EVENT_LBUTTONDBLCLK:
		std::cout << "at (" << std::setw(3) << x << "," << std::setw(3) << y << ") value is: "
			<< static_cast<int>(im->at<uchar>(cv::Point(x, y))) << std::endl;
		break;
	}
}



int formatJPG(cv::Mat& imgL, cv::Mat& imgR, cv::Mat &imgDisp)
{
	


	int cols_ = imgL.cols;
	int rows_ = imgL.rows;



	uint16* leftImg = (uint16*)_mm_malloc(rows_*cols_*sizeof(uint16), 16);
	uint16* rightImg = (uint16*)_mm_malloc(rows_*cols_*sizeof(uint16), 16);
	for (int i = 0; i < rows_; i++)
	{
		for (int j = 0; j < cols_; j++)
		{
			leftImg[i * cols_ + j] = *(imgL.data + i*imgL.step + j * imgL.elemSize());
			rightImg[i * cols_ + j] = *(imgR.data + i*imgR.step + j * imgR.elemSize());
		}
	}

	//左右图像的视差图分配存储空间（width*height*sizeof(float32)）
	float32* dispImg = (float32*)_mm_malloc(rows_*cols_*sizeof(float32), 16);
	float32* dispImgRight = (float32*)_mm_malloc(rows_*cols_*sizeof(float32), 16);



	const int numPaths = 8;


	processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, cols_, rows_, numPaths, dispRange);

	

// 	int test = 0;
// 	for (int i = 0; i < rows_; i++)
// 	{
// 		for (int j = 0; j < cols_; j++)
// 		{
// 			if (dispImg[i * cols_ + j] > 0)
// 			{
// 				*(imgDisp.data + i*imgDisp.step + j * imgDisp.elemSize()) = (uint8)dispImg[i * cols_ + j];
// 			}
// 			else
// 			{
// 				*(imgDisp.data + i*imgDisp.step + j * imgDisp.elemSize()) = 0;
// 			}
// 		}
// 	}

	cv::Mat tmpDisp(cv::Size(cols_, rows_), CV_32FC1, dispImg);
	tmpDisp.copyTo(imgDisp);
	

// 	imgDisp.create(cv::Size(cols_, rows_), CV_32FC1);
// 
// 	for (int i = 0; i < rows_; i++)
// 	{
// 		for (int j = 0; j < cols_; j++)
// 		{
// 			if (dispImg[i * cols_ + j] >= 0)
// 			{
// 				imgDisp.at<float>(i, j) = dispImg[i * cols_ + j];
// 
// 			}
// 			else
// 			{
// 				imgDisp.at<uchar>(i, j) = 0;				
// 			}
// 		}
// 	}

	//std::cout << test << std::endl;

	/*************************v-disparity****************************/
	/*cv::Mat vDisp(cv::Size(dispRange, imgL.rows), CV_8U);
	uint8 tmp = 0;
	int dispMax = 0;
	for (int i = 0; i < imgL.rows; i++)
	{
		uint8 a[dispRange] = { 0 };
		for (int j = 0; j < cols_; j++)
		{
			if ((tmp = *(imgDisp.data + i*imgDisp.step + j*imgDisp.elemSize())) < dispRange)
			{
				if (a[tmp] < 255)
				{
					a[tmp] = ++a[tmp];
				}
			}
			else
			{
				if (a[dispRange - 1] < 255)
				{
					a[dispRange - 1] = ++a[dispRange - 1];
				}
			}
		}

		for (int k = 0; k < dispRange; k++)
		{
			*(vDisp.data + i*vDisp.step + k*vDisp.elemSize()) = a[k];
		}
	}
	cv::imwrite("E:\\vDisp.png", vDisp);*/
	/**************************************************************/

	/*************************u-disparity*************************/
	/*cv::Mat uDisp(cv::Size(cols_, dispRange), CV_8U);

	for (int i = 0; i < cols_; i++)
	{
		int a[dispRange] = { 0 };
		tmp = 0;
		for (int j = 0; j < imgL.rows; j++)
		{
			if ((tmp = *(imgDisp.data + j*imgDisp.step + i*imgDisp.elemSize())) < dispRange)
			{
				if (a[tmp] < 255)
				{
					a[tmp] = ++a[tmp];
				}
			}
			else
			{
				if (a[dispRange - 1] < 255)
				{
					a[dispRange - 1] = ++a[dispRange - 1];
				}
			}
		}
		for (int k = 0; k < dispRange; k++)
		{
			*(uDisp.data + k*uDisp.step + i*uDisp.elemSize()) = a[k];
		}
	}
	cv::imwrite("E:\\uDisp.png", uDisp);*/
	/**************************************************************/


	/*cv::imwrite("E:\\imgDisp.png", imgDisp);
	cv::Mat img_color;
	cv::applyColorMap(imgDisp, img_color, cv::COLORMAP_JET);

	cv::namedWindow("DispImg");
	cv::imshow("DispImg", img_color);
	cv::waitKey(1);*/

	_mm_free(leftImg);
	_mm_free(rightImg);

	return 0;
}

static cv::Scalar computeColor(float val)
{
	const float hscale = 6.f;
	float h = 0.6f * (1.f - val), s = 1.f, v = 1.f;
	float r, g, b;

	static const int sector_data[][3] =
	{ { 1, 3, 0 }, { 1, 0, 2 }, { 3, 0, 1 }, { 0, 2, 1 }, { 0, 1, 3 }, { 2, 1, 0 } };
	float tab[4];
	int sector;
	h *= hscale;
	if (h < 0)
		do h += 6; while (h < 0);
	else if (h >= 6)
		do h -= 6; while (h >= 6);
	sector = cvFloor(h);
	h -= sector;
	if ((unsigned)sector >= 6u)
	{
		sector = 0;
		h = 0.f;
	}

	tab[0] = v;
	tab[1] = v*(1.f - s);
	tab[2] = v*(1.f - s*h);
	tab[3] = v*(1.f - s*(1.f - h));

	b = tab[sector_data[sector][0]];
	g = tab[sector_data[sector][1]];
	r = tab[sector_data[sector][2]];
	//return 255 * cv::Scalar(b, g, r);

	return cv::Scalar(255 * b, 255 * g, 255 * r);
}

static cv::Scalar dispToColor(float disp, float maxdisp)
{
	if (disp < 0)
		return cv::Scalar(128, 128, 128);
	return computeColor(std::min(disp, maxdisp) / maxdisp);
}


static void drawStixel(cv::Mat& img, const Stixel& stixel, cv::Scalar color)
{
	const int radius = std::max(stixel.width / 2, 1);
	const cv::Point tl(stixel.u - radius, stixel.vT);
	const cv::Point br(stixel.u + radius, stixel.vB);
	cv::rectangle(img, cv::Rect(tl, br), color, -1);
}

int main()
{

	// input camera parameters
	const cv::FileStorage cvfs("camera.xml", cv::FileStorage::READ);
	CV_Assert(cvfs.isOpened());
	//const cv::FileNode node(cvfs.fs, NULL);
	const float fx = cvfs["FocalLengthX"];
	const float fy = cvfs["FocalLengthY"];
	const float cu = cvfs["CenterX"];
	const float cv = cvfs["CenterY"];
	const float baseline = cvfs["BaseLine"];
	const float cameraHeight = cvfs["Height"];
	const float cameraPitch = cvfs["Pitch"];




	StixelWrold sw(fx, fy, cu, cv, baseline, cameraHeight, cameraPitch);
	std::string dir = "E:/Image Set/";

	cv::VideoWriter DemoWrite;
	//std::string outputName = dir + "0005.avi";

	for (int frameno = 0;; frameno++)
	{
		
		char base_name[256];
		sprintf(base_name, "%06d.png", frameno);
		std::string bufl = dir + "1/training/0020/" + base_name;
		std::string bufr = dir + "2/training/0020/" + base_name;
// 		std::string bufl = dir + "imageSet/leftImg/" + base_name;
// 		std::string bufr = dir + "imageSet/rightImg/" + base_name;
		std::cout << " " << frameno << std::endl;

		cv::Mat leftBGR = cv::imread(bufl, cv::IMREAD_COLOR);
		cv::Mat right = cv::imread(bufr, cv::IMREAD_GRAYSCALE);

		
		if (leftBGR.empty()  || right.empty())
		{
			std::cout << "Left image no exist!" << std::endl;
			frameno = 0;			
			continue;
			
			//break;
		}
		cv::Mat left;
		if (leftBGR.channels() == 3)
		{
			cv::cvtColor(leftBGR, left, cv::COLOR_BGR2GRAY);
		}
		else
		{
			left = leftBGR.clone();
		}
		
		CV_Assert(left.size() == right.size() && left.type() == right.type());


		cv::Rect roiRect(0, 0, left.cols - left.cols % 16, left.rows);
		cv::Mat leftROI(left, roiRect);
		cv::Mat rightROI(right, roiRect);
		cv::Mat showImage(leftBGR, roiRect);

		

		
		// calculate disparity SGM-Based
		cv::Mat imgDisp;
		formatJPG(leftROI, rightROI, imgDisp);

		
// 		StereoRigParam stereoRigParam;
// 		stereoRigParam.f = 721.5377;
// 		stereoRigParam.cu = 609.5593;
// 		stereoRigParam.cv = 172.854;
// 		stereoRigParam.base = 0.5372;
// 		stereoRigParam.pitch = 0;
// 		stereoRigParam.height = 1.65;

//  		worldCoorFilter(imgDisp, stereoRigParam);
//  
//  
//   		cv::imshow("Filter", imgDisp);
//   		cv::waitKey(10);

		cv::Mat img_color;
		cv::Mat imgDispINT;
		imgDisp.convertTo(imgDispINT, CV_8UC1);
		cv::applyColorMap(imgDispINT, img_color, cv::COLORMAP_HSV);
		
		cv::cvtColor(img_color, img_color, cv::COLOR_RGB2BGRA);

		cv::Mat showResult(cv::Size(leftROI.cols, leftROI.rows * 2), CV_8UC4, 4);

 		
		cv::Mat tmpDisp(showResult, cv::Rect(0,leftROI.rows, leftROI.cols, leftROI.rows));
		img_color.copyTo(tmpDisp);


		//imgDisp.convertTo(imgDisp, CV_32F);
		// calculate stixels
		std::vector<Stixel> stixels;
		sw.compute(imgDisp, stixels, 5);

		// draw stixels
		cv::Mat draw;
		cv::cvtColor(showImage, draw, cv::COLOR_BGR2BGRA);

		cv::Mat stixelImg = cv::Mat::zeros(leftROI.size(), draw.type());
		std::clock_t  endT = clock();

	
		for (const auto& stixel : stixels)
		{
			//drawStixel(stixelImg, stixel, /*cvJetColourMat<int, int, int>(stixel.disp, cvRound(minvalue), cvRound(maxvalue))  */cv::Scalar(52, 242, 4));
			//std::cout << stixel.disp << " ";
			// 
			drawStixel(stixelImg, stixel, dispToColor(stixel.disp, 64));
		}
 		
		stixels.clear();
		draw = draw + 0.5 * stixelImg;
		
		cv::Mat tempDraw(showResult, roiRect);
		draw.copyTo(tempDraw);

		cv::namedWindow("SGM_result");
		cv::imshow("SGM_result", showResult);
		cv::waitKey(1);
	}
}