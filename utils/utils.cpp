#include "utils.h"
void utils::saveBinaryFile(float* vec, size_t len, const std::string& file)
{
	std::ofstream  out(file, std::ios::out | std::ios::binary);
	if (!out.is_open())
		return;
	out.write((const char*)vec, sizeof(float) * len);
	out.close();
}

std::vector<uint8_t> utils::readBinaryFile(const std::string& file) 
{

	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open())
		return {};

	in.seekg(0, std::ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0) {
		in.seekg(0, std::ios::beg);
		data.resize(length);

		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}


std::vector<unsigned char> utils::loadModel(const std::string &mode_file, const std::string &engine_file, int &length) {

	std::ifstream slz(engine_file, std::ios::in | std::ios::binary);
    if(!slz.is_open()) {

        std::cout<<"Build serialize engine "<<std::endl;
        auto builder = dl::nne::CreateInferBuilder();
        auto network = builder->CreateNetwork();
        auto parser =  dl::nne::CreateParser();
        parser->Parse(mode_file.c_str(), *network);
        dl::nne::Engine *engine = nullptr;
        engine = builder->BuildEngine(*network);
        parser->Destroy();
        network->Destroy();
        builder->Destroy();
        auto ser_res = engine->Serialize();
        std::ofstream new_slz(engine_file);
        new_slz.write(static_cast<char *>(ser_res->Data()),static_cast<int64_t>(ser_res->Size()));
        new_slz.close();
        ser_res->Destroy();
		slz.open(engine_file, std::ios::in | std::ios::binary);
    }
	
    slz.seekg(0, std::ios::end);
    length = static_cast<uint64_t>(slz.tellg());
    slz.seekg(0, std::ios::beg);
	std::vector<uint8_t> slz_data;
	slz_data.resize(length);
	slz.read((char*)&slz_data[0], static_cast<int64_t>(length));
	slz.close();
	return slz_data;
}

std::string utils::getSystemTimeStr()
{
	return std::to_string(std::rand()); 
}

bool utils::setInputStream(const utils::InputStream& source, const std::string& imagePath, const std::string& videoPath, 
							const int& cameraID, cv::VideoCapture& capture, int& delayTime, utils::InitParameter& param)
{
	int total_frames = 0;
	std::string img_format;
	switch (source)
	{
	case utils::InputStream::IMAGE:
		img_format = imagePath.substr(imagePath.size()-4, 4);
		if (img_format == ".png" || img_format == ".PNG")
		{
			std::cout << "+-----------------------------------------------------------+" << std::endl;
			std::cout << "| If you use PNG format pictures, the file name must be eg: |" << std::endl;
			std::cout << "| demo0.png, demo1.png, demo2.png ......, but not demo.png. |" << std::endl;
			std::cout << "| The above rules are determined by OpenCV.					|" << std::endl;
			std::cout << "+-----------------------------------------------------------+" << std::endl;
		}
		capture.open(imagePath); //cv::CAP_IMAGES : !< OpenCV Image Sequence (e.g. img_%02d.jpg)
		param.batch_size = 1;
		total_frames = 1;
		delayTime = 0;
		std::cout << "total_frames = " << total_frames << std::endl;
		break;
	case utils::InputStream::VIDEO:
		capture.open(videoPath);
		// 获取视频总帧数
		total_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
		std::cout << "total_frames = " << total_frames << std::endl;
		break;
	case utils::InputStream::CAMERA:
		capture.open(cameraID);
		// 配置成 无穷大
		total_frames = INT_MAX;
		break;
	default:
		break;
	}
	param.src_h = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	param.src_w = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	std::cout << "src_h: " << param.src_h << " src_w: " << param.src_w <<  std::endl;
	return capture.isOpened();
}

void utils::setRenderWindow(InitParameter& param)
{
	if (!param.is_show)
		return;
	int max_w = 960;
	int max_h = 540;
	float scale_h = (float)param.src_h / max_h;
	float scale_w = (float)param.src_w / max_w;
	if (scale_h > 1.f && scale_w > 1.f)
	{
		float scale = scale_h < scale_w ? scale_h : scale_w;
		cv::namedWindow(param.winname, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);  // for Linux
		cv::resizeWindow(param.winname, int(param.src_w / scale), int(param.src_h / scale));
		param.char_width = 16;
		param.det_info_render_width = 18;
		param.font_scale = 0.9;
	}
	else
	{
		cv::namedWindow(param.winname);
	}
}

std::string utils::getTimeStamp()
{
	std::chrono::nanoseconds t = std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::system_clock::now().time_since_epoch());
	return std::to_string(t.count());
}

void utils::show(const std::vector<utils::Box>& objectss, const std::vector<std::string>& classNames, const int& cvDelayTime, 
					cv::Mat& img)
{
	std::string windows_title = "image";
	if(!img.empty())
	{
		cv::namedWindow(windows_title, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);  // allow window resize(Linux)

		int max_w = 960;
		int max_h = 540;
		if (img.rows > max_h || img.cols > max_w)
		{
			cv::resizeWindow(windows_title, max_w, img.rows * max_w / img.cols );
		}
	}

	cv::Scalar color = cv::Scalar(0, 255, 0);
	cv::Point bbox_points[1][4];
	const cv::Point* bbox_point0[1] = { bbox_points[0] };
	int num_points[] = { 4 };
	if (!objectss.empty())
	{
		for (auto& box : objectss)
		{
			if (classNames.size() == 91) // coco91
			{
				color = Colors::color91[box.label];
			}
			if (classNames.size() == 80) // coco80
			{
				color = Colors::color80[box.label];
			}
			if (classNames.size() == 20) // voc20
			{
				color = Colors::color20[box.label];
			}
			cv::rectangle(img, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 2, cv::LINE_AA);
			cv::String det_info = classNames[box.label] + " " + cv::format("%.4f", box.confidence);
			bbox_points[0][0] = cv::Point(box.left, box.top);
			bbox_points[0][1] = cv::Point(box.left + det_info.size() * 11, box.top);
			bbox_points[0][2] = cv::Point(box.left + det_info.size() * 11, box.top - 15);
			bbox_points[0][3] = cv::Point(box.left, box.top - 15);
			cv::fillPoly(img, bbox_point0, num_points, 1, color);
			cv::putText(img, det_info, bbox_points[0][0], cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

			if (!box.land_marks.empty()) // for facial landmarks
			{
				for (auto& pt:box.land_marks)
				{
					cv::circle(img, pt, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA, 0);
				}
			}
		}
		
		cv::imshow(windows_title, img);
		cv::waitKey(cvDelayTime);
	}
}

void utils::save(const std::vector<Box>& objectss, const std::vector<std::string>& classNames, const std::string& savePath, 
					cv::Mat& img, const int& batchi)
{
	cv::Scalar color = cv::Scalar(0, 255, 0);
	cv::Point bbox_points[1][4];
	const cv::Point* bbox_point0[1] = { bbox_points[0] };
	int num_points[] = { 4 };
	if (!objectss.empty())
	{
		for (auto& box : objectss)
		{
			if (classNames.size() == 91) // coco91
			{
				color = Colors::color91[box.label];
			}
			if (classNames.size() == 80) // coco80
			{
				color = Colors::color80[box.label];
			}
			if (classNames.size() == 20) // voc20
			{
				color = Colors::color20[box.label];
			}
			cv::rectangle(img, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 1, cv::LINE_AA);
			cv::String det_info = classNames[box.label] + " " + cv::format("%.4f", box.confidence);
			bbox_points[0][0] = cv::Point(box.left, box.top);
			bbox_points[0][1] = cv::Point(box.left + det_info.size() * 11, box.top);
			bbox_points[0][2] = cv::Point(box.left + det_info.size() * 11, box.top - 15);
			bbox_points[0][3] = cv::Point(box.left, box.top - 15);
			cv::fillPoly(img, bbox_point0, num_points, 1, color);
			cv::putText(img, det_info, bbox_points[0][0], cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
			
			if (!box.land_marks.empty())
			{
				for (auto& pt : box.land_marks)
				{
					cv::circle(img, pt, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA, 0);
				}
			}
		}
	}
		
		int imgi = batchi;
		cv::imwrite(savePath + "_" + std::to_string(imgi) + ".jpg", img);
		cv::waitKey(1); // waitting for writting imgs 
}


void utils::DumpGPUMemoryToFile(const void* gpu_ptr, size_t size_bytes, const char* filepath) {
    try {
        void* host_data = malloc(size_bytes);
        if (host_data == nullptr) {
            std::cerr << "Failed to allocate host memory of size " << size_bytes << " bytes" << std::endl;
            exit(1);
        }

        cudaError_t cuda_status = cudaMemcpy(host_data, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(cuda_status) << std::endl;
            free(host_data);
            exit(1);
        }

        std::ofstream outfile(filepath, std::ios::binary);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            free(host_data);
            exit(1);
        }

        outfile.write(static_cast<const char*>(host_data), size_bytes);
        outfile.close();

        free(host_data);
        std::cout << "Successfully dumped " << size_bytes << " bytes to " << filepath << std::endl;

        exit(0);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        exit(1);
    }
}

void utils::DumpCPUMemoryToFile(const void* cpu_ptr, size_t size_bytes, const char* filepath) {
    try {
        std::ofstream outfile(filepath, std::ios::binary);
        if (!outfile.is_open()) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            exit(1);
        }

        outfile.write(static_cast<const char*>(cpu_ptr), size_bytes);
        outfile.close();

        std::cout << "Successfully dumped " << size_bytes << " bytes to " << filepath << std::endl;
        
        exit(0);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        exit(1);
    }
}

utils::HostTimer::HostTimer()
{
    t1 = std::chrono::steady_clock::now();
}

float utils::HostTimer::getUsedTime()
{
    t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    return(1000 * time_used.count()); // ms
}

utils::HostTimer::~HostTimer()
{
}

utils::DeviceTimer::DeviceTimer()
{
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
}

float utils::DeviceTimer::getUsedTime()
{
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float total_time;
	cudaEventElapsedTime(&total_time, start, end);
	return total_time;
}

utils::DeviceTimer::DeviceTimer(cudaStream_t stream)
{
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, stream);
}

float utils::DeviceTimer::getUsedTime(cudaStream_t stream)
{
	cudaEventRecord(end, stream);
	cudaEventSynchronize(end);
	float total_time;
	cudaEventElapsedTime(&total_time, start, end);
	return total_time;
}

utils::DeviceTimer::~DeviceTimer()
{
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}