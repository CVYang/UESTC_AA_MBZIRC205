#ifndef _DETECT_H
#define _DETECT_H
#include <opencv2/opencv.hpp>
#include<librealsense2/rs.hpp>
#include<librealsense2/rsutil.h>
extern cv::Mat CM;
extern cv::Mat D;

class detect{

private:
	int minh,maxh,mins,maxs,minv,maxv;

	cv::Mat gray,hsv,dst,midimg;
	cv::Mat kernel;
	cv::Mat Denoising(cv::Mat &img);
	/**
	 * 用于图像滤波
	 * 对图像进行高斯滤波，形态学操作，后提取轮廓
	 * @param:  img：opencv　Mat矩阵视频帧
	 * @return: dst:opencv Mat矩阵
	 * 
	 */
public:
	void deal(char color);
	/**
	 * 用于O,R,G,B四种颜色选择
	 * @param:color:字符类型
	 * @return:空
	 * 
	 */
	cv::Point2d getblock(cv::Mat &img);
	/**
	 * 得到选择颜色阈值的最大区域矩形
	 * @param:  img：opencv　Mat矩阵视频帧
	 * @return:　(x,y)：像素坐标
	 * 
	 */
	cv::Mat align_Depth2Color(cv::Mat depth,cv::Mat color,rs2::pipeline_profile profile);
	/**
	 * 对齐深度图和彩色图
	 * @param:  depth,color,profile:深度图视频帧depth,彩色图视频帧color，数据流
	 * @return:　dst:对齐后的深度图
	 * 
	 */
	float measure_distance(cv::Mat &color,cv::Mat depth,cv::Size range,cv::Point2d &Point,rs2::pipeline_profile profile);
	/**
	 * 从对其的深度图中测量距离（0-10m）
	 * 获得一个矩形框内的像素的距离
	 * @param:  color,depth,range,profile:深度图视频帧depth,彩色图视频帧color,深度像素范围range，数据流
	 * @return:　distance
	 * 
	 */


};

#endif