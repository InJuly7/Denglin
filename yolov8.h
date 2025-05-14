#pragma once
#include"utils/common_include.h"
#include"utils/utils.h"
#include"utils/kernel_function.h"

class YOLOV8 {
    public:
        YOLOV8(const utils::InitParameter& param);
        ~YOLOV8();
    public:
    virtual bool init(std::vector<unsigned char>& slz, const int &length);
    virtual void check();
    virtual void copy(const cv::Mat& img);
    virtual void preprocess();
    virtual bool infer();
    virtual void postprocess();
    virtual void reset();

    public:
        std::vector<utils::Box> getObjectss() const;

    protected:
        dl::nne::Engine* m_engine;
        dl::nne::ExecutionContext* m_context;

    protected:
        
        utils::InitParameter m_param;

        const float3 paddingValue = make_float3(114.0f, 114.0f, 114.0f);
        const float3 alpha = make_float3(1.0 / 255.0f, 1.0 / 255.0f, 1.0 / 255.0f);
        const float3 beta = make_float3(0.0f, 0.0f, 0.0f);
        
        dl::nne::Dims m_output_dims;   
        int m_output_area;
        int m_box_tensor;
        int m_total_objects;
        std::vector<utils::Box> m_objectss;
        std::vector<utils::Box> m_detections;
        utils::AffineMat m_dst2src;     
        
        // input
        unsigned char* m_input_src_device;
        float* m_input_dst_device;
        // output
        float* m_output_src_device;
        float* m_output_objects_device;
        float* m_output_objects_host;
        int m_output_objects_width;     
        int* m_output_idx_device;      
        float* m_output_conf_device;
};


