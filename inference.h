#ifndef YOLO_INFERENCE_H_
#define YOLO_INFERENCE_H_

#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

namespace yolo {
	// 定义结构体
	struct Key_PointAndFloat
	{
		// 存储坐标的成员，使用 std::pair 存储两个整数表示坐标
		std::vector<cv::Point2f> key_point;
		// 存储浮点数的成员
		std::vector<float> value;
	};

struct Detection {
	short class_id;
	float confidence;
	cv::Rect box;
	Key_PointAndFloat Key_Point;
};

enum class ProcessState {
	Ainference_request_RUN,  
	Binference_request_RUN,
    
};
class Inference {
 public:
 double Pose_Run_time=0;
double Pose_Run_img=0;
 double huamianshu=0;
//	int flage=1;
bool RUN=false;
int num_requests = 20;//mo ren
//mutable std::mutex flage_mutex;  // 保护 flage 的互斥量
std::string driver = "BATCH:GPU"; //"MULTI:GPU.1,GPU.0" "BATCH:GPU(1)"

	ProcessState run=ProcessState::Ainference_request_RUN;
	Inference() {}
	// Constructor to initialize the model with default input shape
	Inference(const std::string &model_path, const float &model_confidence_threshold, const float &model_NMS_threshold);
	// Constructor to initialize the model with specified input shape
	Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold, const float &model_NMS_threshold);

	Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold, const float &model_NMS_threshold,std::string &driver,int &num_requests_);


	void RunInference(cv::Mat &frame);
	void Pose_RunInference(cv::Mat &frame);
	void Pose_Run_async_Inference(cv::Mat &frame);
 private:
	void InitializeModel(const std::string &model_path);


	void Preprocessing(const cv::Mat &frame, ov::InferRequest &inference_request ,int i);//yu
	//void Pose_PostProcessing(cv::Mat &frame);
	void Pose_PostProcessing(cv::Mat &frame, ov::InferRequest &inference_request ,int i);//hou
	
	cv::Rect GetBoundingBox(const cv::Rect &src) const;
	Key_PointAndFloat GetKeyPointsinBox(Key_PointAndFloat &Key);
	void DrawDetectedObject(cv::Mat &frame, const Detection &detections) const;
	void Pose_DrawDetectedObject(cv::Mat &frame, const Detection &detections) const;
	cv::Point2f scale_factor_;			// Scaling factor for the input frame
	cv::Size2f model_input_shape_;	// Input shape of the model
	cv::Size model_output_shape_;		// Output shape of the model


	ov::InferRequest Ainference_request_;  // OpenVINO inference request
	ov::InferRequest Binference_request_;  // OpenVINO inference request
	ov::InferRequest Cinference_request_;  // OpenVINO inference request
	ov::InferRequest Dinference_request_;  // OpenVINO inference request

std::vector<ov::InferRequest>inference_requests_;
std::vector<bool>flages;

	ov::CompiledModel compiled_model_;    // OpenVINO compiled model

	//ov::InferRequest infer_request;

	float model_confidence_threshold_;  // Confidence threshold for detections
	float model_NMS_threshold_;         // Non-Maximum Suppression threshold

	std::vector<std::string> classes_ {
		"BG", "RG", 
	}; 

private:
    mutable std::mutex flage_mutex;  // 保护 flage 的互斥量
		/*std::vector<std::string> classes_ {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
		"cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
		"scissors", "teddy bear", "hair drier", "toothbrush"
	};*/
};

} // namespace yolo

#endif // YOLO_INFERENCE_H_
