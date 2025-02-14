#include "inference.h"
#include "tongbu.h"
#include "CameraApi.h" // 相机SDK的API头文件
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <chrono>
#include <future>
#include <atomic>
using namespace cv;

unsigned char *g_pRgbBuffer; // 处理后数据缓存区
#include <iostream>
#include <opencv2/highgui.hpp>
#include <time.h>

class PeriodicPrinter
{
public:
	PeriodicPrinter() : running(true)
	{
		// 启动打印线程
		printer_thread = std::thread(&PeriodicPrinter::print, this);
	}

	~PeriodicPrinter()
	{
		// 停止打印线程
		running = false;
		if (printer_thread.joinable())
		{
			printer_thread.join();
		}
	}

private:
	void print()
	{
		while (running)
		{
			std::cout << "---------------———100ms———------------" << std::endl;
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}

	std::thread printer_thread;
	std::atomic<bool> running;
};

// 新增全局变量控制线程运行
std::atomic<bool> running(true);
std::mutex matDeque_mutex;

void GPU_InferenceThread(yolo::Inference& inference, std::deque<cv::Mat>& matDeque) {
    while (running) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // 检查队列并获取最新帧
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(matDeque_mutex);
            if (!matDeque.empty() && !inference.RUN) {
                frame = matDeque.back().clone(); // 使用最新帧
                matDeque.pop_back(); // 移除已处理的帧避免重复
            }
        }

        // 执行异步推理
        if (!frame.empty()) {
            auto frame_ptr = std::make_shared<cv::Mat>(frame);
            std::async(std::launch::async, [frame_ptr, &inference]() {
                inference.Pose_Run_async_Inference(*frame_ptr);
            });
        }

        // 精确控制100Hz频率
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (elapsed < std::chrono::milliseconds(10)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10) - elapsed);
        }
    }
}

// CPU推理线程函数 (类似GPU线程)
void CPU_InferenceThread(yolo::Inference& Ainference, std::deque<cv::Mat>& matDeque) {
    while (running) {
        auto start = std::chrono::high_resolution_clock::now();
        
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(matDeque_mutex);
            if (!matDeque.empty() && !Ainference.RUN) {
                frame = matDeque.back().clone();
                matDeque.pop_back();
            }
        }

        if (!frame.empty()) {
            auto frame_ptr = std::make_shared<cv::Mat>(frame);
            std::async(std::launch::async, [frame_ptr, &Ainference]() {
                Ainference.Pose_Run_async_Inference(*frame_ptr);
            });
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        if (elapsed < std::chrono::milliseconds(10)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10) - elapsed);
        }
    }
}


int main(int argc, char **argv)
{
	int iCameraCounts = 1;
	int iStatus = -1;
	tSdkCameraDevInfo tCameraEnumList;
	int hCamera;
	tSdkCameraCapbility tCapability; // 设备描述信息
	tSdkFrameHead sFrameInfo;
	BYTE *pbyBuffer;
	int iDisplayFrames = 100000;
	IplImage *iplImage = NULL;
	int channel = 3;

	CameraSdkInit(1);

	// 枚举设备，并建立设备列表
	iStatus = CameraEnumerateDevice(&tCameraEnumList, &iCameraCounts);
	printf("state = %d\n", iStatus);

	printf("count = %d\n", iCameraCounts);
	// 没有连接设备
	if (iCameraCounts == 0)
	{
		return -1;
	}
	// 相机初始化。初始化成功后，才能调用任何其他相机相关的操作接口
	iStatus = CameraInit(&tCameraEnumList, -1, -1, &hCamera);

	// 初始化失败
	printf("state = %d\n", iStatus);
	if (iStatus != CAMERA_STATUS_SUCCESS)
	{
		return -1;
	}

	// 获得相机的特性描述结构体。该结构体中包含了相机可设置的各种参数的范围信息。决定了相关函数的参数
	CameraGetCapability(hCamera, &tCapability);

	//
	g_pRgbBuffer = (unsigned char *)malloc(tCapability.sResolutionRange.iHeightMax * tCapability.sResolutionRange.iWidthMax * 3);
	// g_readBuf = (unsigned char*)malloc(tCapability.sResolutionRange.iHeightMax*tCapability.sResolutionRange.iWidthMax*3);

	/*让SDK进入工作模式，开始接收来自相机发送的图像
	数据。如果当前相机是触发模式，则需要接收到
	触发帧以后才会更新图像。    */
	CameraPlay(hCamera);

	/*其他的相机参数设置
	例如 CameraSetExposureTime   CameraGetExposureTime  设置/读取曝光时间
		 CameraSetImageResolution  CameraGetImageResolution 设置/读取分辨率
		 CameraSetGamma、CameraSetConrast、CameraSetGain等设置图像伽马、对比度、RGB数字增益等等。
		 本例程只是为了演示如何将SDK中获取的图像，转成OpenCV的图像格式,以便调用OpenCV的图像处理函数进行后续开发
	*/

	CameraSetAeState(hCamera, false);
	CameraSetExposureTime(hCamera, 500);

	/// CameraSetGain(hCamera, 255,255,255);  // 设置增益，增加亮度
	// CameraSetConrast(hCamera, 155); // 对比度已设置，你可以根据需要调节

	if (tCapability.sIspCapacity.bMonoSensor)
	{
		channel = 1;
		CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_MONO8);
	}
	else
	{
		channel = 3;
		CameraSetIspOutFormat(hCamera, CAMERA_MEDIA_TYPE_BGR8);
	}
	// const std::string model_path_ = "/home/auto/Desktop/yolov8_pose-/model/best_openvino_model/best.xml";
	const std::string model_path = "/home/auto/Desktop/yolov8_pose-/model/best_openvino_model/best.xml";
	// Define the confidence and NMS thresholds
	const float confidence_threshold = 0.2;
	const float NMS_threshold = 0.5;

	std::string driver = "CPU";
	int num_requests = 1;
	// Initialize the YOLO inference with the specified model and parameters
	yolo::Inference inference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);
	//yolo::Inference Ainference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold, driver, num_requests);

	// yolo::Inference Binference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold,driver);
	//  循环显示1000帧图像
	double simage = 0;
	double time = 0;
	double time_ = 0;
	double result = 0;
	double shanchu = 0;

	std::mutex images_mutex;

	const size_t MAX_BUFFER_SIZE = 8;

	// 创建一个包含5个Mat对象的deque
	std::deque<cv::Mat> matDeque;

	ThreadPool pool(MAX_BUFFER_SIZE); // 创建一个包含 4 个线程的线程池

	// PeriodicPrinter printer;
	// std::this_thread::sleep_for(std::chrono::seconds(1));
	//
	// 新增全局变量控制线程运行
std::atomic<bool> running(true);
std::mutex matDeque_mutex;

// GPU推理线程函数
    // 启动推理线程
   // std::thread gpu_thread(GPU_InferenceThread, std::ref(inference), std::ref(matDeque));
   // std::thread cpu_thread(CPU_InferenceThread, std::ref(Ainference), std::ref(matDeque));
//wu yong
	while (true)
	{

		// PeriodicPrinter printer;
		// std::this_thread::sleep_for(std::chrono::seconds(1));

		auto s = std::chrono::high_resolution_clock::now();

		if (CameraGetImageBuffer(hCamera, &sFrameInfo, &pbyBuffer, 1000) == CAMERA_STATUS_SUCCESS)
		{
			CameraImageProcess(hCamera, pbyBuffer, g_pRgbBuffer, &sFrameInfo);
			cv::Mat image(
				cvSize(sFrameInfo.iWidth, sFrameInfo.iHeight),
				sFrameInfo.uiMediaType == CAMERA_MEDIA_TYPE_MONO8 ? CV_8UC1 : CV_8UC3,
				g_pRgbBuffer);
			if (!image.empty())
			{
				matDeque.push_back(image);
				if (matDeque.size() > MAX_BUFFER_SIZE)
				{
					matDeque.pop_front();
				}
			}
		}

if(matDeque.size()>2)
{

if (!inference.RUN) {
    auto frame_ptr = std::make_shared<cv::Mat>(matDeque[0]);
    std::future<void> result = std::async(std::launch::async, [frame_ptr, &inference]() {
        inference.Pose_Run_async_Inference(*frame_ptr);
    });
	//std::cout << "GPU_inference"  << std::endl;//33ms
} 
/*  		if (Ainference.RUN == false)
		{
    auto frame_ptr = std::make_shared<cv::Mat>(matDeque[1]);
    std::future<void> result = std::async(std::launch::async, [frame_ptr, &Ainference]() {
        Ainference.Pose_Run_async_Inference(*frame_ptr);
	//std::cout << "CPU_inference"  << std::endl;
    });
} 
 */
		} 

/* 		else if (inference.RUN && Ainference.RUN)
		{
			// std::cout<<"all_RUN"<<std::endl;
			continue;
		}
 */


		simage += 1;
		CameraReleaseImageBuffer(hCamera, pbyBuffer);

		auto e = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> diff = e - s;
		time += diff.count();
		// std::cout << "diff.count():" << diff.count() << std::endl;//33ms
		// std::cout << "time" << time << std::endl;//33ms

		if (time > 1000)
		{
			std::cout << "相机帧:" << simage / time * 1000 << std::endl;
			std::cout << "推理帧:" << (/* Ainference.huamianshu + */ inference.huamianshu) / time * 1000 << "\n"
					  << std::endl;
			time = 0;
			//Ainference.huamianshu = 0;
			// Binference.huamianshu=0;
			inference.huamianshu = 0;
			simage = 0;
		}
	}

	CameraUnInit(hCamera);
	// 注意，现反初始化后再free
	free(g_pRgbBuffer);

	return 0;
}