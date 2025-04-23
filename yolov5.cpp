#include "yolov5.h"

YOLOV5::YOLOV5(const utils::InitParameter& param) {

    // input
    m_param = param;
    m_input_src_device = nullptr;
    m_input_resize_device = nullptr;
    m_input_rgb_device = nullptr;
    m_input_norm_device = nullptr;
    m_input_hwc_device = nullptr;
    CHECK(cudaMalloc(&m_input_src_device,    3 * param.src_h * param.src_w * sizeof(unsigned char)));
    CHECK(cudaMalloc(&m_input_resize_device, 3 * param.dst_h * param.dst_w * sizeof(float)));
    CHECK(cudaMalloc(&m_input_rgb_device,    3 * param.dst_h * param.dst_w * sizeof(float)));
    CHECK(cudaMalloc(&m_input_norm_device,   3 * param.dst_h * param.dst_w * sizeof(float)));
    CHECK(cudaMalloc(&m_input_hwc_device,    3 * param.dst_h * param.dst_w * sizeof(float)));

    // output
    m_output_src_device = nullptr; // init() malloc
    m_objectss.resize(m_param.topK);
    m_detections.resize(m_param.topK);
}
    

YOLOV5::~YOLOV5() {

    // input
    CHECK(cudaFree(m_input_src_device));
    CHECK(cudaFree(m_input_resize_device));
    CHECK(cudaFree(m_input_rgb_device));
    CHECK(cudaFree(m_input_norm_device));
    CHECK(cudaFree(m_input_hwc_device));
    // output
    CHECK(cudaFree(m_output_src_device));
    delete[] m_output_objects_host;

}

bool YOLOV5::init(std::vector<unsigned char>& slz, const int &length)
{
    if (slz.empty()) {
        return false;
    }

    this->m_engine = dl::nne::Deserialize((char*)&slz[0], length);
    if (this->m_engine == nullptr) {
        return false;
    }
    if(m_param.is_debug)
        std::cout << "Model Deserialize finished" << std::endl;

    this->m_context = this->m_engine->CreateExecutionContext();
    if (this->m_context == nullptr) {
        return false;
    }
    if(m_param.is_debug)
        std::cout << "Model CreateExecutionContext finished" << std::endl;

    m_output_dims = this->m_context->GetBindingDimensions(1);
    m_total_objects = m_output_dims.d[1];
    m_box_tensor = m_output_dims.d[2];
    assert(m_param.batch_size <= m_output_dims.d[0]);
    m_output_area = 1;
    for (int i = 1; i < m_output_dims.nbDims; i++)
    {
        if (m_output_dims.d[i] != 0)
        {
            m_output_area *= m_output_dims.d[i];
        }
    }
    CHECK(cudaMalloc(&m_output_src_device, m_output_area * sizeof(float)));
    m_output_objects_host = new float[m_output_area];
    

    float a = float(m_param.dst_h) / m_param.src_h;
    float b = float(m_param.dst_w) / m_param.src_w;
    float scale = a < b ? a : b;
    cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * m_param.src_w + m_param.dst_w + scale - 1) * 0.5,
                                                0.f, scale, (-scale * m_param.src_h + m_param.dst_h + scale - 1) * 0.5);
    cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
    cv::invertAffineTransform(src2dst, dst2src);
    m_dst2src.v0 = dst2src.ptr<float>(0)[0];
    m_dst2src.v1 = dst2src.ptr<float>(0)[1];
    m_dst2src.v2 = dst2src.ptr<float>(0)[2];
    m_dst2src.v3 = dst2src.ptr<float>(1)[0];
    m_dst2src.v4 = dst2src.ptr<float>(1)[1];
    m_dst2src.v5 = dst2src.ptr<float>(1)[2];
    std::cout << "v0: " << m_dst2src.v0 << " v1: " << m_dst2src.v1 << " v2: " << m_dst2src.v2 << std::endl;
    std::cout << "v3: " << m_dst2src.v3 << " v4: " << m_dst2src.v4 << " v5: " << m_dst2src.v5 << std::endl;
    return true;

}

void YOLOV5::check()
{
    std::cout << "the engine's info:" << std::endl;
    int nb_bindings = m_engine->GetNbBindings();
    for (int i = 0; i < nb_bindings; ++i)
    {
        auto shape = m_engine->GetBindingDimensions(i);
        auto name = m_engine->GetBindingName(i);
        auto data_type = m_engine->GetBindingDataType(i);
        std::cout << name << "  " << data_type << std::endl;
        for (int j = 0; j < shape.nbDims; ++j)
        {
            std::cout << shape.d[j] << "  ";
        }
        std::cout << std::endl;
    }
}

void YOLOV5::copy(const cv::Mat& img)
{
// #if 0 
//     cv::Mat img_fp32 = cv::Mat::zeros(imgsBatch[0].size(), CV_32FC3); // todo 
//     cudaHostRegister(img_fp32.data, img_fp32.elemSize() * img_fp32.total(), cudaHostRegisterPortable);
//     float* pi = m_input_src_device;
//     for (size_t i = 0; i < imgsBatch.size(); i++)
//     {
//         imgsBatch[i].convertTo(img_fp32, CV_32FC3);
//         CHECK(cudaMemcpy(pi, img_fp32.data, sizeof(float) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice));
//         pi += 3 * m_param.src_h * m_param.src_w;
//     }
//     cudaHostUnregister(img_fp32.data);
// #endif


    unsigned char* pi = m_input_src_device;
    CHECK(cudaMemcpy(pi, img.data, sizeof(unsigned char) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice));
    pi += 3 * m_param.src_h * m_param.src_w;
    // utils::DumpGPUMemoryToFile(m_input_src_device, sizeof(unsigned char) * 3 * m_param.src_h * m_param.src_w, "640640_GPU.bin");


// #if 0 // cuda stream
//     cudaStream_t streams[32];
//     for (int i = 0; i < imgsBatch.size(); i++) 
//     {
//         CHECK(cudaStreamCreate(&streams[i]));
//     }
//     unsigned char* pi = m_input_src_device;
//     for (size_t i = 0; i < imgsBatch.size(); i++)
//     {
//         CHECK(cudaMemcpyAsync(pi, imgsBatch[i].data, sizeof(unsigned char) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice, streams[i]));
//         pi += 3 * m_param.src_h * m_param.src_w;
//     }
//     CHECK(cudaDeviceSynchronize());
// #endif
}

void YOLOV5::preprocess()
{
    resizeDevice(m_input_src_device, m_param.src_w, m_param.src_h, m_input_resize_device, m_param.dst_w, m_param.dst_h, 114, m_dst2src);
    // utils::DumpGPUMemoryToFile(m_input_resize_device, 3 * m_param.dst_h * m_param.dst_w * sizeof(float), "input_resize_device.bin");
    // cudaDeviceSynchronize();

    bgr2rgbDevice(m_input_resize_device, m_param.dst_w, m_param.dst_h, m_input_rgb_device, m_param.dst_w, m_param.dst_h);
    // utils::DumpGPUMemoryToFile(m_input_rgb_device, 3 * m_param.dst_h * m_param.dst_w * sizeof(float), "input_rgb_device.bin");     
    // cudaDeviceSynchronize();

    normDevice(m_input_rgb_device, m_param.dst_w, m_param.dst_h, m_input_norm_device, m_param.dst_w, m_param.dst_h, m_param);
    // utils::DumpGPUMemoryToFile(m_input_norm_device, 3 * m_param.dst_h * m_param.dst_w * sizeof(float), "input_norm_device.bin");
    // cudaDeviceSynchronize();
    
    hwc2chwDevice(m_input_norm_device, m_param.dst_w, m_param.dst_h, m_input_hwc_device, m_param.dst_w, m_param.dst_h);
    // utils::DumpGPUMemoryToFile(m_input_hwc_device, 3 * m_param.dst_h * m_param.dst_w * sizeof(float), "input_hwc_device.bin");
    // cudaDeviceSynchronize();
}

bool YOLOV5::infer()
{
    float* bindings[] = { m_input_hwc_device, m_output_src_device };
    bool context = m_context->Execute(1, (void**)bindings);
    // utils::DumpGPUMemoryToFile(m_output_src_device, m_output_area * sizeof(float), "inference_output.bin");
    return context;
}

float calculate_iou(const utils::Box &box1, const utils::Box &box2) {
    float x1 = std::max(box1.left, box2.left);
    float y1 = std::max(box1.top, box2.top);
    float x2 = std::min(box1.right, box2.right);
    float y2 = std::min(box1.bottom, box2.bottom);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area1 = std::max(0.0f, box1.right - box1.left) * std::max(0.0f, box1.bottom - box1.top);
    float area2 = std::max(0.0f, box2.right - box2.left) * std::max(0.0f, box2.bottom - box2.top);
    float union_area = area1 + area2 - intersection;

    return intersection / union_area;
}


void YOLOV5::postprocess()
{
	cudaMemcpy(m_output_objects_host, m_output_src_device, m_output_area*sizeof(float), cudaMemcpyDeviceToHost);
    int current_boxes = 0;
    // decodeDevice
    for (int i = 0; i < m_total_objects; ++i) {
        const float* box_tensor = &m_output_objects_host[i * m_box_tensor];
        float confidence = box_tensor[4];
        if (confidence < m_param.conf_thresh) continue;

        float max_score = 0;
        int label = -1;
        for (int j = 5; j < m_box_tensor; ++j) {
            if (box_tensor[j] > max_score) {
                max_score = box_tensor[j];
                label = j - 5;
            }
        }
        confidence *= max_score;
        if (confidence < m_param.conf_thresh) continue;
        
        float x_lt = box_tensor[0] - box_tensor[2] * 0.5f;
        float y_lt = box_tensor[1] - box_tensor[3] * 0.5f;
        float x_rb = box_tensor[0] + box_tensor[2] * 0.5f;
        float y_rb = box_tensor[1] + box_tensor[3] * 0.5f;
        
        if(current_boxes < m_param.topK)
            m_detections[current_boxes++] = utils::Box(x_lt, y_lt, x_rb, y_rb, confidence, label);
    }
    if(m_param.is_debug) 
            std::cout << "Detections Box Num: " << current_boxes << std::endl;

    std::sort(m_detections.begin(), m_detections.end(),
        [](const utils::Box& a, const utils::Box& b) { return a.confidence > b.confidence; });

    // NMS
    std::vector<bool> removed(current_boxes, false);

    int object_boxes = 0;
    for(int i = 0; i < current_boxes; i++) {

        if(removed[i]) continue;
        m_objectss[object_boxes++] = m_detections[i];
        
        for (size_t j = i + 1; j < current_boxes; ++j) {
            if (removed[j]) continue; 
            if (m_detections[i].label != m_detections[j].label) continue;

            float iou = calculate_iou(m_detections[i], m_detections[j]);
            if (iou > m_param.iou_thresh) {
                removed[j] = true;  // IoU大于阈值的框被标记为移除
            }
        }
    }

    for(int k = 0; k < object_boxes; k++) {
        m_objectss[k].left = m_dst2src.v0 * m_objectss[k].left + m_dst2src.v1 * m_objectss[k].top + m_dst2src.v2;
        m_objectss[k].top = m_dst2src.v3 * m_objectss[k].left + m_dst2src.v4 * m_objectss[k].top + m_dst2src.v5;
        m_objectss[k].right = m_dst2src.v0 * m_objectss[k].right + m_dst2src.v1 * m_objectss[k].bottom + m_dst2src.v2;
        m_objectss[k].bottom = m_dst2src.v3 * m_objectss[k].right + m_dst2src.v4 * m_objectss[k].bottom + m_dst2src.v5;
        if(m_param.is_debug) {
            std::cout << m_objectss[k].left << " " << m_objectss[k].top << " " << m_objectss[k].right << " " << m_objectss[k].bottom 
                        << " " << m_objectss[k].confidence << " " << m_objectss[k].label << std::endl;
        }
    }

    if(m_param.is_debug)
        std::cout << "Bounding Box Num: " << object_boxes << std::endl;
}

std::vector<utils::Box> YOLOV5::getObjectss() const
{
    return this->m_objectss;
}

void YOLOV5::reset()
{
    std::fill(m_objectss.begin(), m_objectss.end(), utils::Box());
    std::fill(m_detections.begin(), m_detections.end(), utils::Box());
}



