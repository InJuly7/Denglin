#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv.hpp>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <dlnne/dlnne.h>

#define H 480
#define W 640

void BGR2YCrCb(const cv::Mat& bgr_image) {

    bgr_image.convertTo(bgr_image, CV_32F);
    std::vector<cv::Mat> channels;
    cv::split(bgr_image, channels);
    cv::Mat& B = channels[0];
    cv::Mat& G = channels[1];
    cv::Mat& R = channels[2];

    cv::Mat Y = 0.299 * R + 0.587 * G + 0.114 * B;
    cv::Mat Cr = (R - Y) * 0.713 + 0.5;
    cv::Mat Cb = (B - Y) * 0.564 + 0.5;
    
    std::vector<cv::Mat> ycrcb_channels = {Y, Cr, Cb};
    cv::merge(ycrcb_channels, bgr_image);

}

void ycrcb_to_bgr(cv::Mat& img) {

    cv::Mat bgr;
    img.convertTo(img, CV_32F);
    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    cv::Mat& Y = channels[0];
    cv::Mat& Cr = channels[1];
    cv::Mat& Cb = channels[2];

    cv::Mat B = (Cb - 0.5) * (1.0 / 0.564) + Y;
    cv::Mat R = (Cr - 0.5) * (1.0 / 0.713) + Y;
    cv::Mat G = (Y - 0.299 * R - 0.114 * B) * (1.0 / 0.587);
    
    std::vector<cv::Mat> bgr_channels = {B, G, R};
    cv::merge(bgr_channels, img);
    // return bgr;

}

void HWC2CHW(cv::Mat &img) {

    std::vector<cv::Mat> channels;
    cv::split(img, channels);
    // 遍历每个通道图像
    for (auto &channel : channels) channel = channel.reshape(1,1);
    cv::hconcat(channels, img);

}

void CHW2HWC(cv::Mat &img) {
    
    std::vector<cv::Mat> channels;
    float* data = img.ptr<float>();
    for(int c = 0; c < 3; c++) {
        cv::Mat channel(480, 640, CV_32FC1, data + c*480*640);
        channels.push_back(channel);
    }
    cv::merge(channels, img);
    
}

void SeAFusion_preprocess(cv::Mat &vis_img, cv::Mat &ir_img, std::vector<cv::Mat> &h_buffers) {
    
    cv::resize(vis_img, vis_img, cv::Size(W, H));
    cv::resize(ir_img, ir_img, cv::Size(W, H));
    vis_img.convertTo(vis_img, CV_32F, 1.0/255);
    ir_img.convertTo(ir_img, CV_32F, 1.0/255);
    BGR2YCrCb(vis_img);
    HWC2CHW(vis_img);
    HWC2CHW(ir_img);
    h_buffers[0] = vis_img;
    h_buffers[1] = ir_img;
    // std::cout << "SeAFusion Resize BRG2YCrCb HWC2CHW " << duration_1.count() << " ms" << std::endl;
    // std::cout << "SeAFusion_preprocess finished !" << std::endl;
}

void SeAFusion_postprecess(std::vector<cv::Mat> &h_buffers, void* d_buffers[]) {
    
    CHW2HWC(h_buffers[0]);
    ycrcb_to_bgr(h_buffers[0]);
    cv::normalize(h_buffers[0], h_buffers[0], 0, 255, cv::NORM_MINMAX, CV_8U);
    // std::cout << "SeAFusion_postprecess finished !" << std::endl;
}

bool save_image(const cv::Mat& image, const std::string& name = "output") {
    std::string filename = name + ".png";
    return cv::imwrite(filename, image);
}

void printModel(dl::nne::Engine *engine) {

    int nb_bindings = engine->GetNbBindings();
    for (int i = 0; i < nb_bindings; ++i)
    {
        auto shape = engine->GetBindingDimensions(i);
        auto name = engine->GetBindingName(i);
        auto data_type = engine->GetBindingDataType(i);
        std::cout << name << "  " << data_type << std::endl;
        for (int j = 0; j < shape.nbDims; ++j)
        {
            std::cout << shape.d[j] << "  ";
        }
        std::cout << std::endl;
    }
}

void printTensor(float* tensor, int channel)
{
    for (int c = 0; c < channel; ++c)
    {
        std::cout << "Channel " << c << ":" << std::endl;
        for (int h = 0; h < H; ++h)
        {
            if((h < 3) || (h >= H - 3))
            {   
                if(h == 0) printf("[");
                for (int w = 0; w < W; ++w)
                {  
                    if((w<3)||(w >= W - 3))
                    {
                        if(w == 0) printf("[");
                        std::cout<< tensor[c * H * W + h * W + w] << " ";
                        if (w == 2) printf("......");
                        else if((w == W - 1) && (h != H - 1)) printf("]\n");
                    }
                }
                if(h == (H - 1)) printf("]]\n");
                
            }
        }
    }
}


int main(int argc ,char *argv[]) {

    // 以原始图像格式 读取图像
    std::string ir_path = "/home/firefly/Tingshuo/SeAFusion/0N_ir.png";
    std::string vis_path = "/home/firefly/Tingshuo/SeAFusion/0N_vis.png";
    cv::Mat ir_image = cv::imread(ir_path, cv::IMREAD_GRAYSCALE);
    cv::Mat vis_image = cv::imread(vis_path, cv::IMREAD_UNCHANGED);
    if(ir_image.empty()||vis_image.empty()) {
        std::cout<<"图片读取失败"<<std::endl;
        return -1;
    }
    // 获取红外和可见光图像的尺寸
    std::cout << "Infrared Image Size: " << ir_image.size() << " channels: " << ir_image.channels() << std::endl;
    std::cout << "Visible Image Size: " << vis_image.size() << " channels: " << vis_image.channels() << std::endl;

    std::string model_path = "/home/firefly/Tingshuo/SeAFusion/seafusion.quantized.rlym";
    std::string engine_path = "/home/firefly/Tingshuo/SeAFusion/seafusion.quantized.slz";
 
    std::ifstream slz(engine_path);
    if(!slz.is_open()) {
        std::cout<<"Build seafusion serialize engine "<<std::endl;
        auto builder = dl::nne::CreateInferBuilder();
        auto network = builder->CreateNetwork();
        auto parser =  dl::nne::CreateParser();
        parser->Parse(model_path.c_str(), *network);
        dl::nne::Engine *engine = nullptr;
        engine = builder->BuildEngine(*network);
        parser->Destroy();
        network->Destroy();
        builder->Destroy();
        auto ser_res = engine->Serialize();
        std::ofstream new_slz(engine_path);
        new_slz.write(static_cast<char *>(ser_res->Data()),static_cast<int64_t>(ser_res->Size()));
        new_slz.close();
        ser_res->Destroy();
        return 0;
    }

    // 创建 构建器 网络 解析器 推理引擎
    slz.seekg(0, std::ios::end);
    uint64_t length = static_cast<uint64_t>(slz.tellg());
    slz.seekg(0, std::ios::beg);
    char *slz_data = new char[length];
    slz.read(slz_data, static_cast<int64_t>(length));
    
    dl::nne::Engine *engine = nullptr;
    engine = dl::nne::Deserialize(slz_data, length);
    auto context = engine->CreateExecutionContext();
    delete[] slz_data;
    

    cv::VideoCapture vis_cap("/home/firefly/Tingshuo/SeAFusion/589_15_vis.mp4");
    cv::VideoCapture ir_cap("/home/firefly/Tingshuo/SeAFusion/589_15_ir.mp4");
    // 检查视频是否成功打开
    if (!vis_cap.isOpened() || !ir_cap.isOpened()) {
        std::cout << "Error: Could not open video files." << std::endl;
        return -1;
    }
    double fps = vis_cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('M','P','4','V'), fps, cv::Size(640, 480));
    cv::Mat vis_frame, ir_frame;
    // 获取输入节点和输出节点的信息
    // NCHW vis [1,3,480,640] ir [1,1,480,640] output [1,1,480,640]
    printModel(engine);
    std::vector<cv::Mat> h_buffers(3);
    void* d_buffers[3];
    
    int count = 0;
    // 创建缓冲区
    float* d_input0;
    cudaMalloc((void**)&d_input0, 1*3*480*640*sizeof(float));
    float* d_input1;
    cudaMalloc((void**)&d_input1, 1*1*480*640*sizeof(float));
    float* d_output0;
    cudaMalloc((void**)&d_output0, 1*1*480*640*sizeof(float));

    //  读取视频流
    auto start_0 = std::chrono::high_resolution_clock::now();
    while(vis_cap.read(vis_frame) && ir_cap.read(ir_frame)) {
        count ++;
        if (vis_frame.empty() || ir_frame.empty()) break;

        auto start_1 = std::chrono::high_resolution_clock::now();
        SeAFusion_preprocess(vis_frame, ir_frame, h_buffers);
        auto end_1 = std::chrono::high_resolution_clock::now();
        auto duration_1 = std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1) / 1000.0;
        
        cudaMemcpy(d_input0, vis_frame.data, 1*3*480*640*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_input1, ir_frame.data, 1*1*480*640*sizeof(float), cudaMemcpyHostToDevice);
        d_buffers[0] = d_input0;
        d_buffers[1] = d_input1;
        d_buffers[2] = d_output0;


        // 推理
        auto start_2 = std::chrono::high_resolution_clock::now();
        context->Execute(1, d_buffers);
        auto end_2 = std::chrono::high_resolution_clock::now();
        auto duration_2 = std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2) / 1000.0;
        
        float* h_input0 = h_buffers[0].ptr<float>();
        cudaMemcpy(h_input0, d_output0, 1*640*480*sizeof(float), cudaMemcpyDeviceToHost);

        auto start_3 = std::chrono::high_resolution_clock::now();
        SeAFusion_postprecess(h_buffers, d_buffers);
        auto end_3 = std::chrono::high_resolution_clock::now();
        auto duration_3 = std::chrono::duration_cast<std::chrono::microseconds>(end_3 - start_3) / 1000.0;

        std::cout << "SeAFusion 前处理时间: " << duration_1.count() 
                    << " ms 推理时间: " << duration_2.count() 
                    << " ms 后处理时间: " << duration_3.count() << " ms" << std::endl;

        if(count % 60  == 0) {
            auto end_60 = std::chrono::high_resolution_clock::now();
            auto duration_60 = std::chrono::duration_cast<std::chrono::microseconds>(end_60 - start_0) / 1000.0;
            float fps = 1000/(duration_60.count()/count);
            std::cout << "每60帧 平均推理帧率: " << fps << " fps" << std::endl;
        }

        cv::Mat output_frame = h_buffers[0];
        cv::imshow("Fusion Result", output_frame);
        writer.write(output_frame);
        if (cv::waitKey(1) == 'q') break;
    }

    cudaFree(d_input0);
    cudaFree(d_input1);
    cudaFree(d_output0);
    vis_cap.release();
    ir_cap.release();
    writer.release();
    cv::destroyAllWindows();
   
    return 0;
    
}
