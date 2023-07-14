#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <string>

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "cuda_runtime_api.h"
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "logging.h"

using namespace sample;
using namespace std;
using namespace cv;

class TensorRTInference {
public:


	// structure of BoundingBoxes
	struct Bbox {
		float x;
		float y;
		float w;
		float h;
		float score;
		int classes;
	};


	// Convert the input frame to a tensor
	void preprocess(const cv::Mat& img, float data[])
	{
		int w;
		int h;
		int x;
		int y;
		double r_w = INPUT_W / (img.cols*1.0);
		double r_h = INPUT_H / (img.rows*1.0);
		if (r_h > r_w) {
			w = INPUT_W;
			h = r_w * img.rows;
			x = 0;
			y = (INPUT_H - h) / 2;
		}
		else {
			w = r_h * img.cols;
			h = INPUT_H;
			x = (INPUT_W - w) / 2;
			y = 0;
		}
		cv::Mat re(h, w, CV_8UC3);
		cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
		cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
		re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

		int i = 0;
		for (int row = 0; row < INPUT_H; ++row) {
			uchar const* uc_pixel = out.data + row * out.step;
			for (int col = 0; col < INPUT_W; ++col) {
				data[i] = (float)uc_pixel[2] / 255.0;
				data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
				data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
				uc_pixel += 3;
				++i;
			}
		}
	}

	// visualization
	cv::Mat renderBoundingBox(cv::Mat image, const std::vector<Bbox> &bboxes) {
		for (const auto &rect : bboxes)
		{
			cv::Rect rst(rect.x - rect.w / 2, rect.y - rect.h / 2, rect.w, rect.h);
			cv::Scalar color = class_colors[rect.classes];
			cv::rectangle(image, rst, color, 2, cv::LINE_8, 0);

			int baseLine;
			std::string label = class_names[rect.classes] + ": " + std::to_string(rect.score * 100).substr(0, 4) + "%";

			cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_TRIPLEX, 0.5, 1, &baseLine);
			rectangle(image, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2 - round(1.5*labelSize.height)),
					cv::Point(rect.x - rect.w / 2 + round(1.0*labelSize.width), rect.y - rect.h / 2 + baseLine), color, cv::FILLED);
			cv::putText(image, label, cv::Point(rect.x - rect.w / 2, rect.y - rect.h / 2), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
		return image;
	}


	// just map the box back to the original image
	std::vector<Bbox> rescale_box(const std::vector<Bbox> &out, int width, int height)  {
		float gain = static_cast<float>(INPUT_SIZE) / std::max(width, height);
		float pad_x = (static_cast<float>(INPUT_W) - width * gain) / 2;
		float pad_y = (static_cast<float>(INPUT_W)  - height * gain) / 2;

		std::vector<Bbox> boxs;
		Bbox box;
		for (auto const & i : out) {
			box.x = (i.x - pad_x) / gain;
			box.y = (i.y - pad_y) / gain;
			box.w = i.w / gain;
			box.h = i.h / gain;
			box.score = i.score;
			box.classes = i.classes;

			boxs.push_back(box);
		}
		return boxs;
	}

	// default constructor
	TensorRTInference() = default;

	// destructor
	~TensorRTInference() {
		engine_runtime->destroy();
		engine_infer->destroy();
		engine_context->destroy();
	}

	// initialization of model
	void init(const string& engine_file_path) {
		initLibNvInferPlugins(&gLogger, "");
		engine_runtime = nvinfer1::createInferRuntime(gLogger);

		// Deserialize the engine
		std::ifstream file;
		file.open(engine_file_path, std::ios::binary | std::ios::in);
		file.seekg(0, std::ios::end);
		std::streamoff length = file.tellg();
		file.seekg(0, std::ios::beg);

		std::shared_ptr<char> data(new char[length], std::default_delete<char[]>());
		file.read(data.get(), length);
		file.close();

		engine_infer = engine_runtime->deserializeCudaEngine(data.get(), length, nullptr);

		// Create the context
		engine_context = engine_infer->createExecutionContext();

		int input_index = engine_infer->getBindingIndex("images"); //1x3x640x640
		// std::string input_name = engine_infer->getBindingName(0);
		int output_index_1 = engine_infer->getBindingIndex("num_detections");  //1
		int output_index_2 = engine_infer->getBindingIndex("nmsed_boxes");   // 2
		int output_index_3 = engine_infer->getBindingIndex("nmsed_scores");  //3
		int output_index_4 = engine_infer->getBindingIndex("nmsed_classes"); //5

		std::cout << "images: " << input_index << " num_detections-> " << output_index_1 << " nmsed_boxes-> " << output_index_2
				  << " nmsed_scores-> " << output_index_3 << " nmsed_classes-> " << output_index_4 << std::endl;

		if (engine_context == nullptr)
		{
			std::cerr << "Failed to create TensorRT Execution Context." << std::endl;
		}

		std::cout << "loaded trt model , do inference" << std::endl;
	}

	// detection on frames
	void detect(const cv::Mat& frame) {
		std::cout << "Processing frame..." << std::endl;

		cv::Mat image_origin = frame.clone();
		preprocess(frame, input);

		// Allocate memory for the input tensor
		input_size = INPUT_SIZE * INPUT_SIZE * 3 * sizeof(float);  // <- input
		cudaMalloc(&buffers[0], input_size);

		// Allocate memory for the output tensors
		output_size_0 = 1 * sizeof(int);
		output_size_1 = 1 * 100 * 4 * sizeof(float);
		output_size_2 = 1 * 100 * sizeof(float);
		output_size_3 = 1 * 100 * sizeof(float);

		cudaMalloc(&buffers[1], output_size_0);  // <- num_detections
		cudaMalloc(&buffers[2], output_size_1);  // <- nmsed_boxes
		cudaMalloc(&buffers[3], output_size_2);  // <- nmsed_scores
		cudaMalloc(&buffers[4], output_size_3);  // <- nmsed_classes

		// Copy the input tensor to the GPU
		cudaMemcpy(buffers[0], input, input_size, cudaMemcpyDeviceToDevice);

		// -- do execute -- //
		engine_context->executeV2(buffers);

		cudaMemcpy(output_0, buffers[1], output_size_0, cudaMemcpyDeviceToHost);
		cudaMemcpy(output_1, buffers[2], output_size_1, cudaMemcpyDeviceToHost);
		cudaMemcpy(output_2, buffers[3], output_size_2, cudaMemcpyDeviceToHost);
		cudaMemcpy(output_3, buffers[4], output_size_3, cudaMemcpyDeviceToHost);

		std::cout << "THE COUNT OF DETECTION IN THIS FRAME: " << output_0[0] << std::endl;
		std::vector<Bbox> pred_box;

		for (int i = 0; i < output_0[0]; i++)
		{

			Bbox box;
			box.x = (output_1[i * 4 + 2] + output_1[i * 4]) / 2.0;
			box.y = (output_1[i * 4 + 3] + output_1[i * 4 + 1]) / 2.0;
			box.w = output_1[i * 4 + 2] - output_1[i * 4];
			box.h = output_1[i * 4 + 3] - output_1[i * 4 + 1];
			box.score = output_2[i];
			box.classes = static_cast<int>(output_3[i]);  // (int)output_3[i]

			std::cout << "class: " << class_names[box.classes] << ", probability: " << box.score * 100 << "%" << std::endl;

			pred_box.push_back(box);
		}

		std::vector<Bbox> out = rescale_box(pred_box, frame.cols, frame.rows);
		cv::Mat img = renderBoundingBox(frame, out);

		// Display the frame in a window
		cv::namedWindow("Video", 1);
		cv::imshow("Video", img);
		cv::waitKey(1);

		cudaFree(buffers[0]);
		cudaFree(buffers[1]);
		cudaFree(buffers[2]);
		cudaFree(buffers[3]);
		cudaFree(buffers[4]);
	}

private:
	Logger gLogger;
	nvinfer1::IRuntime* engine_runtime;
	nvinfer1::ICudaEngine* engine_infer;
	nvinfer1::IExecutionContext* engine_context;

	void* buffers[5];

	static const int INPUT_W = 640;  // In-class initializer
	static const int INPUT_H = 640;  // In-class initializer
	static const int INPUT_SIZE = 640;  // In-class initializer

	static const int num_classes = 11;  // number of classes (In-class initializer)

	float input[INPUT_SIZE * INPUT_SIZE * 3];
	int output_0[1];
	float output_1[1 * 100 * 4];
	float output_2[1 * 100];
	float output_3[1 * 100];

	int input_size;  // input (images)
	int output_size_0;  // num_detections
	int output_size_1;  // nmsed_boxes
	int output_size_2;  // nmsed_scores
	int output_size_3;  // nmsed_classes

	const std::array<std::string, num_classes> class_names = {
		"biker",
	  	"car",
	  	"pedestrian",
	  	"trafficLight",
	  	"trafficLight-Green",
	  	"trafficLight-GreenLeft",
	  	"trafficLight-Red",
	  	"trafficLight-RedLeft",
	  	"trafficLight-Yellow",
	  	"trafficLight-YellowLeft",
	  	"truck"
	};

	const std::array<cv::Scalar, num_classes> class_colors = {
		cv::Scalar(255, 0, 0), // red
		cv::Scalar(0, 255, 0), // lime
		cv::Scalar(255, 69, 0), // orange
		cv::Scalar(128, 0, 0), // maroon
		cv::Scalar(255, 215, 0), // gold
		cv::Scalar(255, 165, 0), // orange
		cv::Scalar(0, 255, 255), // aqua
		cv::Scalar(255, 255, 0), // yellow
		cv::Scalar(138, 43, 226), // blueviolet
		cv::Scalar(255, 127, 80), // coral
		cv::Scalar(0, 0, 255), // blue
	};
};

int main(int argc, char** argv) {

	// Load the engine file
	TensorRTInference trt_inference;

	// Initialize the inference engine
	trt_inference.init("../model.trt");

	// Open the video file
	cv::VideoCapture cap("../sample_videos/video.mp4");
	if (!cap.isOpened()) {
		std::cerr << "Failed to open video file" << std::endl;
		return -1;
	}

	// Run the inference on each frame
	cv::Mat frame;
	while (cap.read(frame)) {
		trt_inference.detect(frame);
	}

	return 0;
}
