#ifndef __STIXEL_WORLD_H__
#define __STIXEL_WORLD_H__

#include <opencv2/opencv.hpp>

struct Stixel
{
	int u;
	int vT;
	int vB;
	int width;
	float disp;
};

class StixelWrold
{
public:
	StixelWrold() = delete;

	StixelWrold(float fx, float fy, float cu, float cv,
		        float baseline,
		        float cameraHeight,
		        float cameraPitch);

	void compute(const cv::Mat& disp, std::vector<Stixel>& stixels, int stixelWidth = 5);

	std::vector<int> lowerPath;
	std::vector<int> upperPath;

private:
	float fx_;
	float fy_;
	float cu_;
	float cv_;
	float baseline_;
	float cameraHeight_;
	float cameraPitch_;
};

#endif // !__STIXEL_WORLD_H__