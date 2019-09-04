#include <opencv2/opencv.hpp>
#include <iostream>
#include <future>
#include <random>
#include <detect.h>
 
#include<librealsense2/rs.hpp>
#include<librealsense2/rsutil.h>
void detect::deal(char color)
{
        switch(color){
    case 'B':
        minh = 90;
        maxh = 124;
        mins = 114;
        maxs = 255;
        minv = 46;
        maxv = 255;
        break;
    case 'G':
        minh = 38;
        maxh = 53;
        mins = 93;
        maxs = 255;
        minv = 46;
        maxv = 255;
        break;
    case 'R':
        minh = 0;
        maxh = 7;
        mins = 80;
        maxs = 255;
        minv = 0;
        maxv = 255;
        break;
    case 'O':
        minh = 8;
        maxh = 30;
        mins = 132;
        maxs = 255;
        minv = 113;
        maxv = 255;
        break;
    default:
        std::cout << "输入错误" << std::endl;
        exit(0);
    }
}

cv::Mat detect::Denoising(cv::Mat &img){
    cv::GaussianBlur(img,img,cv::Size(3,3),3,3);
    
    cv::cvtColor(img,gray,CV_BGR2GRAY);
    cv::cvtColor(img,hsv,CV_BGR2HSV);
    cv::inRange(hsv,cv::Scalar(minh,mins,minv),cv::Scalar(maxh,maxs,maxv), dst);
    //cv::imshow("hsv1",dst);
	kernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	morphologyEx(dst,dst,cv::MORPH_OPEN,kernel);
    morphologyEx(dst,dst,cv::MORPH_CLOSE,kernel);
    cv::dilate(dst,dst,kernel,cv::Point(-1,-1));
    cv::dilate(dst,dst,kernel,cv::Point(-1,-1));
    cv::dilate(dst,dst,kernel,cv::Point(-1,-1));
    cv::erode(dst,dst,kernel,cv::Point(-1,-1));
    //cv::imshow("hsv2",dst);
    medianBlur(dst,dst,3);
    cv::Canny(dst,dst,10,250,3);
    cv::imshow("hsv3",dst);
    return dst;

}
cv::Point2d detect::getblock(cv::Mat &img){
    Denoising(img);
    std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
    cv::findContours(dst,contours,hierarchy, CV_RETR_LIST , CV_CHAIN_APPROX_SIMPLE);
    	cv::Rect rect_;

    //std::cout<<contours.size()<<std::endl;

	drawContours(dst,contours,-1,cv::Scalar(255),3);
    cv::imshow("edge1",dst);
    std::vector<cv::Rect>rect(contours.size()); 

    if(contours.size()>0){
        double maxArea=0;
        for(int i=0;i<contours.size();i++)
        {
        double area = contourArea(contours[static_cast<int>(i)]);
        if (area > maxArea)
				{
					maxArea = area;
				    rect_ = boundingRect(contours[static_cast<int>(i)]);
            }
        }
    }
 
    //std::cout<<rect_.x<<","<<rect_.y<<std::endl;
    cv::Point2d P1(rect_.x+rect_.width/2,rect_.y+rect_.height/2);
       cv::rectangle(img,cv::Point2d(rect_.x,rect_.y),cv::Point2d(rect_.x+rect_.width,rect_.y+rect_.height),cv::Scalar(0,255,0),2);
       cv::circle(img,cv::Point(rect_.x+rect_.width/2,rect_.y+rect_.height/2),6,cv::Scalar(0,255,0));
    double dx=rect_.x+rect_.width/2-320;
    double dy=rect_.y+rect_.height/2-240;
    cv::imshow("edge2",img);
    //std::cout<<"dx,dy"<<dx<<","<<dy<<std::endl;
    return P1;
}
float get_depth_scale(rs2::device dev)
{
    // Go over the device's sensors
    for (rs2::sensor& sensor : dev.query_sensors())
    {
        // Check if the sensor if a depth sensor
        if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
        {
            return dpt.get_depth_scale();
        }
    }
    throw std::runtime_error("Device does not have a depth sensor");
}
cv::Mat detect::align_Depth2Color(cv::Mat depth,cv::Mat color,rs2::pipeline_profile profile){
    //声明数据流
    auto depth_stream=profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto color_stream=profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
 
    //获取内参
    const auto intrinDepth=depth_stream.get_intrinsics();
    const auto intrinColor=color_stream.get_intrinsics();
 
    rs2_extrinsics  extrinDepth2Color;
    rs2_error *error;
    rs2_get_extrinsics(depth_stream,color_stream,&extrinDepth2Color,&error);
 
    float pd_uv[2],pc_uv[2];
    float Pdc3[3],Pcc3[3];
    float depth_scale = get_depth_scale(profile.get_device());
    int y=0,x=0;
    cv::Mat result=cv::Mat(color.rows,color.cols,CV_16U,cv::Scalar(0));
    for(int row=0;row<depth.rows;row++){
        for(int col=0;col<depth.cols;col++){
            pd_uv[0]=col;
            pd_uv[1]=row;
            uint16_t depth_value=depth.at<uint16_t>(row,col);
            float depth_m=depth_value*depth_scale;
            rs2_deproject_pixel_to_point(Pdc3,&intrinDepth,pd_uv,depth_m);
            rs2_transform_point_to_point(Pcc3,&extrinDepth2Color,Pdc3);
            rs2_project_point_to_pixel(pc_uv,&intrinColor,Pcc3);

            x=(int)pc_uv[0];
            y=(int)pc_uv[1];
//            if(x<0||x>color.cols)
//                continue;
//            if(y<0||y>color.rows)
//                continue;
            //最值限定
            x=x<0? 0:x;
            x=x>depth.cols-1 ? depth.cols-1:x;
            y=y<0? 0:y;
            y=y>depth.rows-1 ? depth.rows-1:y;
 
            result.at<uint16_t>(y,x)=depth_value;
        }
    }
    //返回一个与彩色图对齐了的深度信息图像
    return result;
}
float detect::measure_distance(cv::Mat &color,cv::Mat depth,cv::Size range,cv::Point2d &Point,rs2::pipeline_profile profile)
{
    //获取深度像素与现实单位比例（D435默认1毫米）
    float depth_scale = get_depth_scale(profile.get_device());
    //定义图像中心点
    cv::Point2d center=Point;
    //定义计算距离的范围
    cv::Rect RectRange(center.x-range.width/2,center.y-range.height/2,range.width,range.height);
    //遍历该范围
    float distance_sum=0;
    int effective_pixel=0;
    for(int y=RectRange.y;y<RectRange.y+RectRange.height;y++){
        for(int x=RectRange.x;x<RectRange.x+RectRange.width;x++){
            //如果深度图下该点像素不为0，表示有距离信息
            if(depth.at<uint16_t>(y,x)){
                distance_sum+=depth_scale*depth.at<uint16_t>(y,x);
                effective_pixel++;
            }
        }
    }
    //std::cout<<"遍历完成，有效像素点:"<<effective_pixel<<std::endl;
    float effective_distance=0;
    if(effective_pixel == 0)
    {
        effective_distance = 0xffff;
        std::cout<<"目标距离："<<effective_distance<<" m"<<std::endl;
    }
    else
    {
        effective_distance = distance_sum/effective_pixel;
        std::cout<<"目标距离："<<effective_distance<<" m"<<std::endl;
    }
    
    char distance_str[30];
    sprintf(distance_str,"the distance is:%f m",effective_distance);
    cv::rectangle(color,RectRange,cv::Scalar(0,0,255),2,8);
    cv::putText(color,(std::string)distance_str,cv::Point(color.cols*0.02,color.rows*0.05),
                cv::FONT_HERSHEY_PLAIN,2,cv::Scalar(0,255,0),2,8);
    return effective_distance;
}
 