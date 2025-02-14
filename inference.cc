#include "inference.h"
#include <opencv2/highgui.hpp>
#include <memory>
#include <opencv2/dnn.hpp>
#include <random>
#include <thread>
#include <future>

namespace yolo
{

	// Constructor to initialize the model with default input shape
	Inference::Inference(const std::string &model_path, const float &model_confidence_threshold, const float &model_NMS_threshold)
	{
		model_input_shape_ = cv::Size(640, 640); // Set the default size for models with dynamic shapes to prevent errors.
		model_confidence_threshold_ = model_confidence_threshold;
		model_NMS_threshold_ = model_NMS_threshold;
		InitializeModel(model_path);
	}

	// Constructor to initialize the model with specified input shape
	Inference::Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold, const float &model_NMS_threshold)
	{
		

		model_input_shape_ = model_input_shape;
		model_confidence_threshold_ = model_confidence_threshold;
		model_NMS_threshold_ = model_NMS_threshold;
		InitializeModel(model_path);
	}
	
		// Constructor to initialize the model with specified input shape
	Inference::Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold, const float &model_NMS_threshold,std::string &driver_,int &num_requests_ )
	{
		num_requests=num_requests_;
		driver=driver_;
		model_input_shape_ = model_input_shape;
		model_confidence_threshold_ = model_confidence_threshold;
		model_NMS_threshold_ = model_NMS_threshold;

		InitializeModel(model_path);
	}

	void Inference::InitializeModel(const std::string &model_path)
	{
		ov::Core core;													// OpenVINO core object
		std::shared_ptr<ov::Model> model = core.read_model(model_path); // Read the model from file
		
		// If the model has dynamic shapes, reshape it to the specified input shape
		if (model->is_dynamic())
		{
			model->reshape({1, 3, static_cast<long int>(model_input_shape_.height), static_cast<long int>(model_input_shape_.width)});
		}

		// Preprocessing setup for the model
		ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
		ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
		ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255, 255, 255});
		ppp.input().model().set_layout("NCHW");
		ppp.output().tensor().set_element_type(ov::element::f32);
		model = ppp.build(); // Build the preprocessed model

		// Compile the model for inference
		    // 检查 GPU 属性

		// compiled_model_ = core.compile_model(model, "AUTO");
		compiled_model_ = core.compile_model(model, driver,
		 ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)//THROUGHPUT LATENCY
		//,ov::hint::model_priority(ov::hint::Priority::MEDIUM)
		,ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE)
		//, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY)
		 );
//compiled_model_ = core.compile_model(model, "GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
		//Ainference_request_ = compiled_model_.create_infer_request(); 
		//Binference_request_ = compiled_model_.create_infer_request();		 
		    // 初始化多个推理请求

     // 根据硬件调整数量 20-29hz 4-28.68hz 12-29hz
std::cout<<"num_requests=="<<num_requests<<std::endl;
    for (int i = 0; i < num_requests; ++i) {
        inference_requests_.push_back(compiled_model_.create_infer_request());
		flages.push_back(true);
    }

		short width, height;

		// Get input shape from the model
		const std::vector<ov::Output<ov::Node>> inputs = model->inputs();
		const ov::Shape input_shape = inputs[0].get_shape();
		height = input_shape[1];
		width = input_shape[2];
		model_input_shape_ = cv::Size2f(width, height);

		// Get output shape from the model
		const std::vector<ov::Output<ov::Node>> outputs = model->outputs();
		const ov::Shape output_shape = outputs[0].get_shape();
		height = output_shape[1];
		width = output_shape[2];
		model_output_shape_ = cv::Size(width, height);
	}

	void Inference::Pose_Run_async_Inference(cv::Mat &frame)
	{
		
		int request_id = -1;

std::lock_guard<std::mutex> lock(flage_mutex);  // 使用正确的变量名
int count=0;
    for (int i = 0; i < flages.size(); i++) {
       // std::cout << i << " == " << flages[i] << "\t";
        count++;

        if (count == 4) {
          //  std::cout << std::endl;
            count = 0;
        }
    }

		for(int i=0;i<flages.size();i++)
		{
			if(flages[i])
			{
				//std::cout<<i<<"____"<<std::endl;

				request_id=i;
				flages[request_id]=false;

				auto frame_ptr = std::make_shared<cv::Mat>(frame); //捕获一下
			
					// 使用 std::async 启动异步任务
				std::future<void> result = std::async(std::launch::async,
					[this, frame_ptr,request_id](std::reference_wrapper<ov::InferRequest> inference_request_ref) {
					// 使用 frame_ptr 和引用传递的 inference_request_
					Preprocessing(*frame_ptr, inference_request_ref.get(),request_id);
					},
					std::ref(inference_requests_[request_id]));  

			break;
			}
		}



		if (request_id == -1) {
		RUN=true;
        // 无可用请求，跳过或等待
		//std::cout<<"_____________Runing__"<<std::endl;
        return;
   		 }



	}

	// Method to preprocess the input frame
	void Inference::Preprocessing(const cv::Mat &frame, ov::InferRequest &inference_request ,int i)
	{
		cv::Mat resized_frame;
		cv::resize(frame, resized_frame, model_input_shape_, 0, 0, cv::INTER_AREA); // Resize the frame to match the model input shape

		// Calculate scaling factor
		scale_factor_.x = static_cast<float>(frame.cols / model_input_shape_.width);
		scale_factor_.y = static_cast<float>(frame.rows / model_input_shape_.height);

		float *input_data = (float *)resized_frame.data;																						 // Get pointer to resized frame data
		const ov::Tensor input_tensor = ov::Tensor(compiled_model_.input().get_element_type(), compiled_model_.input().get_shape(), input_data); // Create input tensor
		inference_request.set_input_tensor(input_tensor);	

		//std::cout << "画面" << huamianshu << std::endl;

					// 使用 lambda 捕获 frame，并处理推理结果
		//auto frame_ptr = std::make_shared<cv::Mat>(frame); // Use shared_ptr to ensure frame's lifecycle
		auto frame_ptr = std::make_shared<cv::Mat>(frame.clone());


		auto inference_request_ref = std::ref(inference_request);  // 包装成引用
			// 设置回调函数
		inference_request.set_callback([this, frame_ptr,inference_request_ref,i](std::exception_ptr ex_ptr)
											{
												if (ex_ptr)
												{
													try
													{
														std::rethrow_exception(ex_ptr);
													}
													catch (const std::exception &e)
													{
														std::cerr << "Error during inference: " << e.what() << std::endl;
													}
												}
        	Pose_PostProcessing(*frame_ptr,inference_request_ref.get(),i);


			/*
        	// 使用 std::async 启动异步任务
			std::future<void> result = std::async(std::launch::async,
    		[this, frame_ptr,inference_request_ref]() {
        	// 使用 frame_ptr 和引用传递的 inference_request_
        	Pose_PostProcessing(*frame_ptr,inference_request_ref.get());
   			 });   // 使用 std::ref 传递引用 */
												//Pose_PostProcessing(*frame_ptr,inference_request_);
												//std::cout << "Pose_PostProcessing应该没完成" << std::endl;
												});
												
			inference_request.start_async();		  // 启动新的推理任务

			//wo bu zuo yi bu le JOJO!
			//inference_request.wait();
    // 后处理完成后，安全重置标志位


																					 // Set input tensor for inference
	}

	// Method to postprocess the inference results
	void Inference::Pose_PostProcessing(cv::Mat &frame, ov::InferRequest &inference_request, int i)
	{

		//std::cout << "The  Pose_PostProcessing成功  " << std::endl;
		// auto frame_copy = frame.clone();  // 创建一个副本
		const float *detections = inference_request.get_output_tensor().data<const float>();//赶紧用掉inference_request解锁
		//std::cout << "Pose_PostProcessing进入" << std::endl;



		std::vector<int> class_list;
		std::vector<float> confidence_list;
		std::vector<cv::Rect> box_list;
		std::vector<Key_PointAndFloat> key_list;
		// Get the output tensor from the inference request

		const cv::Mat detection_outputs(model_output_shape_, CV_32F, (float *)detections); // Create OpenCV matrix from output tensor

		// std::cout << "The full i-th column matrix at column " << i << ":\n" << classes_scores << std::endl;
		int labels_size = 2;
		int labels__zhanwei = 3 + labels_size;
		for (int i = 0; i < detection_outputs.cols; ++i)
		{
			const cv::Mat classes_scores = detection_outputs.col(i).rowRange(4, labels__zhanwei); // 现在是两类

			cv::Point class_id;
			double score;
			cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id); // Find the class with the highest score

			// Check if the detection meets the confidence threshold
			if (score > model_confidence_threshold_)
			{
				// std::cout << "The  detection_outputs  "  << ":\n" << detection_outputs.col(i) << std::endl;
				// std::cout << "The full i-th column matrix at column " << i << ":\n" << classes_scores << std::endl;
				// std::cout << "The full i-th column matrix at column " << i << ":\n" << detection_outputs.col(i) << std::endl;

				class_list.push_back(class_id.y);
				confidence_list.push_back(score);
				// std::cout << score << "_Pose" << std::endl;

				const float x = detection_outputs.at<float>(0, i);
				const float y = detection_outputs.at<float>(1, i);
				const float w = detection_outputs.at<float>(2, i);
				const float h = detection_outputs.at<float>(3, i);

				Key_PointAndFloat paf;
				for (int j = 0; j < int(detection_outputs.rows - labels__zhanwei) / 3; j++)
				{
					paf.key_point.push_back(cv::Point(detection_outputs.at<float>(labels__zhanwei + 1 + 3 * j, i), detection_outputs.at<float>(labels__zhanwei + 2 + 3 * j, i)));
					paf.value.push_back(detection_outputs.at<float>(labels__zhanwei + 3 + 3 * j, i));
				}
				// std::cout<<paf.key_point[0]<<std::endl;
				key_list.push_back(paf);

				cv::Rect box;
				box.x = static_cast<int>(x);
				box.y = static_cast<int>(y);
				box.width = static_cast<int>(w);
				box.height = static_cast<int>(h);
				box_list.push_back(box);
			}
		}
		// std::cout << "The  detection_outputs  "  << ":\n" << detection_outputs<< std::endl;
		//  Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes
		std::vector<int> NMS_result;
		cv::dnn::NMSBoxes(box_list, confidence_list, model_confidence_threshold_, model_NMS_threshold_, NMS_result);

		
		//std::cout<<"后处理画面数："<<huamianshu<<std::endl;
		// Collect final detections after NMS
		for (int i = 0; i < NMS_result.size(); ++i)
		{
			Detection result;
			const unsigned short id = NMS_result[i];
			// std::cout << "NMS后的box_list ID"<<id << std::endl;
			result.class_id = class_list[id];
			result.confidence = confidence_list[id];
			result.box = GetBoundingBox(box_list[id]);
			result.Key_Point = GetKeyPointsinBox(key_list[id]);

			//Pose_DrawDetectedObject(frame, result);


		} 

	//	std::cout << "Pose_PostProcessing完成" << std::endl;

std::lock_guard<std::mutex> lock(flage_mutex); 
				flages[i]=true;
		RUN=false;
		huamianshu += 1;
	}

	Key_PointAndFloat Inference::GetKeyPointsinBox(Key_PointAndFloat &Key)
	{

		for (int i = 0; i < Key.key_point.size(); i++)
		{
			Key.key_point[i].x = Key.key_point[i].x * scale_factor_.x;
			Key.key_point[i].y = Key.key_point[i].y * scale_factor_.y;
		}
		return Key;
	}

	// Method to get the bounding box in the correct scale
	cv::Rect Inference::GetBoundingBox(const cv::Rect &src) const
	{
		cv::Rect box = src;
		box.x = (box.x - box.width / 2) * scale_factor_.x;
		box.y = (box.y - box.height / 2) * scale_factor_.y;
		box.width *= scale_factor_.x;
		box.height *= scale_factor_.y;
		return box;
	}

	void Inference::Pose_DrawDetectedObject(cv::Mat &frame, const Detection &detection) const
	{
		// auto frame = frame_.clone();  // 创建一个副本
		// std::cout << "识别 " << std::endl;
		const cv::Rect &box = detection.box;
		const float &confidence = detection.confidence;
		const int &class_id = detection.class_id;
		const Key_PointAndFloat &Key_points = detection.Key_Point;

		// Generate a random color for the bounding box
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dis(120, 255);
		const cv::Scalar &color = cv::Scalar(dis(gen), dis(gen), dis(gen));
		const cv::Scalar &Key_color = cv::Scalar(0, 255, 255);

		float soufang = 0.75;
		// Draw the bounding box around the detected object
		cv::rectangle(frame, cv::Point(box.x - (box.width * 0.25), box.y - (box.height * 0.5)), cv::Point(box.x + box.width * 1.25, box.y + box.height * 1.5), color, 3);

		// Prepare the class label and confidence text
		std::string classString = classes_[class_id] + std::to_string(confidence).substr(0, 4);

		// Get the size of the text box
		cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 0.75, 2, 0);
		cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

		// Draw the text box
		cv::rectangle(frame, textBox, color, cv::FILLED);

		// Put the class label and confidence text above the bounding box
		cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 0), 2, 0);

		// 做四个点
		for (int i = 0; i < Key_points.key_point.size(); i++)
		{
			cv::circle(frame, Key_points.key_point[i], 2, Key_color, 2);
			// std::cout << "点" << Key_points.key_point[i] << std::endl;
			cv::Rect Key_textBox(Key_points.key_point[i].x + 10, Key_points.key_point[i].y + 10, textSize.width + 25, textSize.height + 5);
			cv::rectangle(frame, Key_textBox, Key_color, cv::FILLED);
			// std::cout << "点txt"  << std::endl;
			cv::putText(frame, std::to_string(Key_points.value[i]), cv::Point(Key_textBox.x + 1, Key_textBox.y + 17), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 0), 2, 0);
		}
		// 绘制第一对对角点的连线
		cv::line(frame, Key_points.key_point[0], Key_points.key_point[2], cv::Scalar(0, 0, 255), 2);
		// 绘制第二对对角点的连线
		cv::line(frame, Key_points.key_point[1], Key_points.key_point[3], cv::Scalar(0, 255, 0), 2);

		std::cout << "0   " << Key_points.key_point[0].x << "<<x y>>" << Key_points.key_point[0].y << std::endl;
		std::cout << "1	" << Key_points.key_point[1].x << "<<x y>>" << Key_points.key_point[1].y << std::endl;
		std::cout << "2	" << Key_points.key_point[2].x << "<<x y>>" << Key_points.key_point[2].y << std::endl;
		std::cout << "3	" << Key_points.key_point[3].x << "<<x y>>" << Key_points.key_point[3].y << std::endl;

if (!frame.empty()) {
    cv::imshow("show", frame);
    cv::waitKey(1);
}
	}

} // namespace yolo
