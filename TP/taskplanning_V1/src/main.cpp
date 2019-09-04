#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <stdlib.h> 
#include <string>
#include <string.h>
#include <iostream>

#include <vector>
#include <time.h>


#include <unistd.h>
#include <fcntl.h>

#include <errno.h>
#include <sys/ioctl.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/select.h>


#include <assert.h>
#include <asm/ioctls.h>
#include <asm/termbits.h>

#include "../inc/detect.h"
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

int colorpara[4][3]={
    {30,105,130},   // Blue
    {80,115,65},   // Green
    {7,8,9},   // Red
    {135,60,30} // orange
};

/*SIZE
------------------------
red 0.3*0.2*0.2
green 0.6*0.2*0.2
blue 1.2*0.2*0.2
orange 1.8*0.2*0.2
------------------------
Domain RGB
------------------------
red   (134,59,80) (360,255,231)
green　(22,37,22)  (208,255,238)
blue   
orange
------------------------
Domain HSV
------------------------
red   (0,43,46) (10,255,255)/(156,43,46) (180,255,255)/(155,80,80) (180,255,255)
green  (35,43,46) (77,255,255)
blue   (100,43,46) (124,255,255) 
orange  (11,43,46) (25,255,255)
------------------------
*/
using namespace cv;
using namespace std;


int fd; //serialport
int param_baudrate_;
int serial_initial()
{
	struct termios2 tio;

    ioctl(fd, TCGETS2, &tio);
    bzero(&tio, sizeof(struct termios2));

    tio.c_cflag = BOTHER;
    tio.c_cflag |= (CLOCAL | CREAD | CS8); //8 bit no hardware hanfddshake

    tio.c_cflag &= ~CSTOPB;   //1 stop bit
    tio.c_cflag &= ~CRTSCTS;  //No CTS
    tio.c_cflag &= ~PARENB;   //No Parity

#ifdef CNEW_RTSCTS
    tio.c_cflag &= ~CNEW_RTSCTS; // no hw flow control
#endif

    tio.c_iflag &= ~(IXON | IXOFF | IXANY); // no sw flow control


    tio.c_cc[VMIN] = 0;         //min chars to read
    tio.c_cc[VTIME] = 0;        //time in 1/10th sec wait

    tio.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    // raw output mode   
    tio.c_oflag &= ~OPOST;

    tio.c_ispeed = 115200;
    tio.c_ospeed = 115200;


    ioctl(fd, TCSETS2, &tio);
    return 0;
    
}

int serial_open()
{
	fd = open("/dev/ttyUSB0",O_RDWR| O_NDELAY);
    if(fd == -1)
    {
        perror("Open UART failed!");
        return -1;
    }
    //sleep(1000);
    return fd;
}

void serial_close()
{
	close(fd);
}

void serial_send(unsigned char * buf, int len)
{
	int n =0;
	n = write(fd,buf,len);
	if(n == -1)
	{
		perror("write UART failed!");
	}
}


typedef struct calculateDistanceOfTarget
{
	double f;
	double y;
	double p;
	double w; 
	double ratio;
	double sx;
	double sy;
    double sz;
}calculateDistanceOfTarget;

calculateDistanceOfTarget target;


void FindROI(Mat& img) {
	Mat gray_src;
	cvtColor(img, gray_src, COLOR_BGR2GRAY);
	
	//边缘检测
	Mat canny_output;
	Canny(gray_src, canny_output, 10, 100, 3, false);
 
	//轮廓查找
	vector<vector<Point> > contours;
	vector<Vec4i> hireachy;
	findContours(canny_output, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
 
	//绘制轮廓
	int minw = img.cols*0.75;
	int minh = img.rows*0.75;
	RNG rng(12345);
	Mat drawImage = Mat::zeros(img.size(), CV_8UC3);
	Rect bbox;
	for (size_t t = 0; t < contours.size(); t++) {
		//查找可倾斜的最小外接矩
		RotatedRect minRect = minAreaRect(contours[t]);
		//获得倾斜角度
		float degree = abs(minRect.angle);
		if (minRect.size.width > minw && minRect.size.height > minh && minRect.size.width < (img.cols - 5)) {
			printf("current angle : %f\n", degree);
			Point2f pts[4];
			minRect.points(pts);
			bbox = minRect.boundingRect();
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			for (int i = 0; i < 4; i++) {
				line(drawImage, pts[i], pts[(i + 1) % 4], color, 2, 8, 0);
			}
		}
	}
	imshow("output", drawImage);
 
	//提取ROI区域
	if (bbox.width > 0 && bbox.height > 0) {
		Mat roiImg = img(bbox);
		imshow("roi_win", roiImg);
	}
	return;
}


 int main(int argc, char const *argv[])
 {
    map<int,int>task;
    const char* depth_win="depth_Image";
    cv::namedWindow(depth_win,cv::WINDOW_AUTOSIZE);
    const char* color_win="color_Image";
    cv::namedWindow(color_win,cv::WINDOW_AUTOSIZE);
    const char* canny_win="canny_Image";
    cv::namedWindow(canny_win,cv::WINDOW_AUTOSIZE);
    const char* target_win="target_Image";
    cv::namedWindow(target_win,cv::WINDOW_AUTOSIZE);
    const char* mask_win="mask_Image";
    cv::namedWindow(mask_win,cv::WINDOW_AUTOSIZE);
    const char* mask_canny_win="mask_canny_Image";
    cv::namedWindow(mask_canny_win,cv::WINDOW_AUTOSIZE);
    
 
    //深度图像颜色map
    rs2::colorizer c;                          // Helper to colorize depth images
    rs2::align align_to(RS2_STREAM_COLOR);
    //创建数据管道
    rs2::pipeline pipe;
    rs2::config pipe_config;
    pipe_config.enable_stream(RS2_STREAM_DEPTH,1280,720,RS2_FORMAT_Z16,30);
    pipe_config.enable_stream(RS2_STREAM_COLOR,1280,720,RS2_FORMAT_BGR8,30);
 
    //start()函数返回数据管道的profile
    rs2::pipeline_profile profile = pipe.start(pipe_config);
 
    //定义一个变量去转换深度到距离
    float depth_clipping_distance = 1.f;
 
    //声明数据流
    auto depth_stream=profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto color_stream=profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
 
    //获取内参
    auto intrinDepth=depth_stream.get_intrinsics();
    auto intrinColor=color_stream.get_intrinsics();
 
    //直接获取从深度摄像头坐标系到彩色摄像头坐标系的欧式变换矩阵
    //auto  extrinDepth2Color=depth_stream.get_extrinsics_to(color_stream);
    Mat targetImg, grayImg,hsvImg,cannyImg;
    Mat dst;
    Mat kernel;
    Mat mask,mask_gray,mask_hsv,mask_canny,mask_dst;
    sleep(5);
    vector<std::vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;
    vector<std::vector<cv::Point> > mask_contours;
	vector<cv::Vec4i> mask_hierarchy;
    vector<std::vector<cv::Point> > target_contours;
	vector<cv::Vec4i> target_hierarchy;
    while(cv::waitKey(1)!='q')
    {
        rs2::frameset frameset = pipe.wait_for_frames();
        auto processed = align_to.process(frameset);
        //取深度图和彩色图
        rs2::frame color_frame = processed.get_color_frame();//processed.first(align_to);
        rs2::frame depth_frame = processed.get_depth_frame();
        rs2::frame depth_frame_4_show = processed.get_depth_frame().apply_filter(c);
        //获取宽高
        const int depth_w=depth_frame.as<rs2::video_frame>().get_width();
        const int depth_h=depth_frame.as<rs2::video_frame>().get_height();
        const int color_w=color_frame.as<rs2::video_frame>().get_width();
        const int color_h=color_frame.as<rs2::video_frame>().get_height();
 
        //创建OPENCV类型 并传入数据
        Mat depth_image(Size(depth_w,depth_h),
                                CV_16U,(void*)depth_frame.get_data(),Mat::AUTO_STEP);
        Mat depth_image_4_show(Size(depth_w,depth_h),
                                CV_8UC3,(void*)depth_frame_4_show.get_data(),Mat::AUTO_STEP);
        Mat color_image(Size(color_w,color_h),
                                CV_8UC3,(void*)color_frame.get_data(),Mat::AUTO_STEP);
        
        GaussianBlur(color_image,color_image,cv::Size(3,3),3,3);
        cvtColor(color_image,hsvImg,CV_BGR2HSV);
        inRange(hsvImg,cv::Scalar(0,48,38),cv::Scalar(255,230,255), dst);

	    kernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13));
	    morphologyEx(dst,dst,cv::MORPH_OPEN,kernel);
        dilate(dst,dst,kernel,cv::Point(-1,-1));
        Canny(dst,cannyImg,10,100,3);

        
        cv::findContours(cannyImg,contours,hierarchy, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_SIMPLE);
    	cv::Rect rect_;
        
        if(contours.size()>0)
        {
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
        // cv::Point2d P1(rect_.x+rect_.width/2,rect_.y+rect_.height/2);
        
        if(rect_.x != 0)
        {
            // cv::rectangle(color_image,cv::Point2d(rect_.x,rect_.y),cv::Point2d(rect_.x+rect_.width,rect_.y+rect_.height),cv::Scalar(0,255,0),2);
            mask=color_image(rect_);
            cvtColor(mask,mask_hsv,CV_BGR2HSV);
            inRange(mask_hsv,cv::Scalar(0,48,38),cv::Scalar(255,230,255), mask_dst);

    	    kernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
	        morphologyEx(mask_dst,mask_dst,cv::MORPH_OPEN,kernel);
            dilate(mask_dst,mask_dst,kernel,cv::Point(-1,-1));
            Canny(mask_dst,mask_canny,10,80,3);
            imshow(mask_canny_win,mask_canny);
            
            cv::findContours(mask_canny,mask_contours,mask_hierarchy, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_SIMPLE);
        	float maxw = 0;
	        float maxh = 0;
	        double degree = 0;
            
    	    cv::Rect mask_rect;
            int select = 0;
            if(mask_contours.size()>0)
            {
                double mask_maxArea=0;
                for(int i=0;i<mask_contours.size();i++)
                {
                    double mask_area = contourArea(mask_contours[static_cast<int>(i)]);
                    if (mask_area > mask_maxArea)
                    {
                        mask_maxArea = mask_area;
                        mask_rect = boundingRect(mask_contours[static_cast<int>(i)]);
                        select = i;
                    }
                    RotatedRect rrect = minAreaRect(mask_contours[i]);
                    //cv::rectangle(mask,cv::Point2d(rrect.,rrect.y),cv::Point2d(rrect.x+rrect.width,rrect.y+rrect.height),cv::Scalar(0,255,0),2);
                    if(!rrect.size.empty())
                    {
                        degree = abs(rrect.angle);
                        cout<<"degree: "<<degree<<endl;
                        if (degree > 0) 
                        {
                            maxw = max(maxw, rrect.size.width);
                            maxh = max(maxh, rrect.size.height);
                        }
                    }
                }
                for (size_t t = 0; t < mask_contours.size(); t++) 
                {
                    RotatedRect minRect = minAreaRect(mask_contours[t]);
                    if (maxw == minRect.size.width && maxh == minRect.size.height) 
                    {
                        degree = 90+minRect.angle;
                        Point2f pts[4];
                        minRect.points(pts);
                    }
                }
            }
            

            if(degree != 0)
            {
                Point2f center(mask.cols / 2, mask.rows / 2);
                Mat rotm = getRotationMatrix2D(center, degree, 1.0);  
                
                //旋转图像
                Mat RotateDst;
                warpAffine(mask, RotateDst, rotm, mask.size(), INTER_LINEAR, 0, Scalar(255, 255, 255)); 
                
                // 统计砖块信息 颜色空间
                
                Mat target_gray,target_canny,target_temp;
                Rect cut(5,5,RotateDst.cols-5,RotateDst.rows-12);
                target_temp = RotateDst(cut);
                
                cvtColor(target_temp,target_gray,CV_BGR2GRAY);
                Canny(target_gray,target_canny,10,60,3);
                kernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	            morphologyEx(target_canny,target_canny,cv::MORPH_CLOSE,kernel);
                //dilate(mask_dst,mask_dst,kernel,cv::Point(-1,-1));
                imshow("target_canny",target_canny);
               
                findContours(target_canny,target_contours,target_hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
                cv::Rect tarrect;
                int index = 0;
                if(target_contours.size()>0)
                {
                    cout<<"------------------------"<<endl;
                    for(int i=0;i<target_contours.size();i++)
                    {
                        int area =contourArea(target_contours[static_cast<int>(i)]);   
                        if(area<1600 && area > 100)
                        {
                            tarrect = boundingRect(target_contours[static_cast<int>(i)]);
                            cout<<tarrect.size()<<" area: "<<area<<endl;
                            Point org;
                            org.x = tarrect.x;
                            org.y = tarrect.y+tarrect.height/2;
                            char num[3];
                            sprintf(num,"%d",i);
                            putText(target_temp,(std::string)num,org,cv::FONT_HERSHEY_COMPLEX,0.3,Scalar(0,255,255),1,8,0);
                            //drawContours(RotateDst,target_contours,i,Scalar(0,255,0),CV_FILLED,8,target_hierarchy);
                            cv::rectangle(target_temp,tarrect.tl(),tarrect.br(),cv::Scalar(0,255,0),2);
                        }
                    }
                    cout<<"++++++++++++++++++++++++++++++"<<endl;
                }
                imshow("temp",target_temp);
                imshow("rotatedst",RotateDst);
            }
            
        }
        if(!mask.empty())
        {
            imshow(mask_win,mask);
        }
        
        imshow(target_win,dst);
        imshow(canny_win,cannyImg);
        imshow(depth_win,depth_image_4_show);
        imshow(color_win,color_image);
    }
    // 0 255 48 230 38 255
    return EXIT_SUCCESS;
}
