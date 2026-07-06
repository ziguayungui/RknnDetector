#include "native-lib.h"

// ======================== 1. 模型初始化（接收 Java 层 ByteArray）=======================
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_rknndetector_RknnDriver_initModel(JNIEnv *env, jobject thiz,
                                                   jbyteArray model_buf, jint model_size) {
    if (is_model_loaded) {
        rknn_destroy(g_ctx);
        g_ctx = 0;
        is_model_loaded = false;
        if (g_input_attrs) { free(g_input_attrs); g_input_attrs = nullptr; }
        if (g_output_attrs) { free(g_output_attrs); g_output_attrs = nullptr; }
    }

    jbyte *buf = env->GetByteArrayElements(model_buf, nullptr);
    if (buf == nullptr) {
        LOGE("无法获取模型缓冲区");
        return JNI_FALSE;
    }

    // rknn_init 会内部拷贝模型数据到 NPU 驱动内存，
    // 因此此处的 buf 可以在函数返回前安全释放（不依赖 rknn_run）
    int ret = rknn_init(&g_ctx, buf, model_size, 0, NULL);

    // 模型数据已拷贝完毕，立即释放 Java 内存
    env->ReleaseByteArrayElements(model_buf, buf, JNI_ABORT);

    if (ret < 0) {
        LOGE("rknn_init 失败，错误码: %d", ret);
        return JNI_FALSE;
    }

    // 动态查询输入输出属性（彻底解决硬编码尺寸问题）
    ret = rknn_query(g_ctx, RKNN_QUERY_IN_OUT_NUM, &g_io_num, sizeof(g_io_num));
    if (ret < 0) {
        LOGE("rknn_query IN_OUT_NUM 失败: %d", ret);
        rknn_destroy(g_ctx);
        return JNI_FALSE;
    }

    g_input_attrs = (rknn_tensor_attr *)malloc(g_io_num.n_input * sizeof(rknn_tensor_attr));
    g_output_attrs = (rknn_tensor_attr *)malloc(g_io_num.n_output * sizeof(rknn_tensor_attr));

    for (uint32_t i = 0; i < g_io_num.n_input; i++) {
        g_input_attrs[i].index = i;
        rknn_query(g_ctx, RKNN_QUERY_INPUT_ATTR, &(g_input_attrs[i]), sizeof(rknn_tensor_attr));
        LOGD("Input[%d] dims: N=%d, C=%d, H=%d, W=%d", i,
             g_input_attrs[i].dims[0], g_input_attrs[i].dims[1],
             g_input_attrs[i].dims[2], g_input_attrs[i].dims[3]);
    }
    for (uint32_t i = 0; i < g_io_num.n_output; i++) {
        g_output_attrs[i].index = i;
        rknn_query(g_ctx, RKNN_QUERY_OUTPUT_ATTR, &(g_output_attrs[i]), sizeof(rknn_tensor_attr));
    }

    is_model_loaded = true;
    LOGD("模型加载成功！输入尺寸: %dx%d", g_input_attrs[0].dims[3], g_input_attrs[0].dims[2]);
    return JNI_TRUE;
}

// ======================== 2. 获取输入形状（Java 层动态绑定）=======================
extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_rknndetector_RknnDriver_getInputShape(JNIEnv *env, jobject thiz) {
    if (!is_model_loaded || g_input_attrs == nullptr) return nullptr;

    jintArray result = env->NewIntArray(2);
    if (result != nullptr) {
        // RKNN 模型 dims 根据 fmt 不同：
        // NCHW: {N, C, H, W}  -> 返回 {dims[3], dims[2]}
        // NHWC: {N, H, W, C}  -> 返回 {dims[2], dims[1]}
        // 这里根据模型实际情况，取 {W, H} = {dims[2], dims[1]}
        jint tmp[2] = {(jint) g_input_attrs[0].dims[2], (jint) g_input_attrs[0].dims[1]};
        LOGD("getInputShape: dims[N=%d, dim1=%d, dim2=%d, dim3=%d], returning [W=%d, H=%d]",
             g_input_attrs[0].dims[0], g_input_attrs[0].dims[1],
             g_input_attrs[0].dims[2], g_input_attrs[0].dims[3],
             tmp[0], tmp[1]);
        env->SetIntArrayRegion(result, 0, 2, tmp);
    }
    return result;
}

// ======================== 3. 执行推理（含 Use-After-Free 修复）=======================
extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_rknndetector_RknnDriver_runInference(JNIEnv *env, jobject thiz,
                                                      jbyteArray rgb_pixels,
                                                      jint width, jint height) {
    if (!is_model_loaded) {
        LOGE("模型未加载");
        return nullptr;
    }

    jbyte *pixel_data = env->GetByteArrayElements(rgb_pixels, nullptr);
    if (pixel_data == nullptr) {
        LOGE("获取图像数据失败");
        return nullptr;
    }

    // 设置输入 Tensor
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].buf = pixel_data;

    LOGD("rknn_inputs_set: width=%d, height=%d, size=%d, expected=%d", 
         width, height, inputs[0].size, 
         g_input_attrs[0].dims[3] * g_input_attrs[0].dims[2] * g_input_attrs[0].dims[1]);

    int ret = rknn_inputs_set(g_ctx, 1, inputs);
    if (ret < 0) {
        LOGE("rknn_inputs_set 失败: %d", ret);
        env->ReleaseByteArrayElements(rgb_pixels, pixel_data, JNI_ABORT);
        return nullptr;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // 执行 NPU 推理。此时硬件通过 DMA 读取 pixel_data 指向的内存。
    // 只有在 rknn_run 执行完毕后，才能安全释放 Java 内存。
    // 过早释放会导致 NPU 访问已回收内存，引发随机 Segmentation Fault。
    ret = rknn_run(g_ctx, NULL);

    env->ReleaseByteArrayElements(rgb_pixels, pixel_data, JNI_ABORT);

    if (ret < 0) {
        LOGE("rknn_run 失败: %d", ret);
        return nullptr;
    }

    // 获取多输出（使用 std::vector 代替 VLA，兼容 NDK）
    std::vector<rknn_output> outputs(g_io_num.n_output);
    memset(outputs.data(), 0, outputs.size() * sizeof(rknn_output));
    for (size_t i = 0; i < outputs.size(); i++) {
        outputs[i].want_float = 0;
    }

    ret = rknn_outputs_get(g_ctx, g_io_num.n_output, outputs.data(), NULL);
    if (ret < 0) {
        LOGE("rknn_outputs_get 失败: %d", ret);
        return nullptr;
    }

    // 执行 C++ 层后处理（NMS + 置信度过滤）
    std::vector<DetectionResult> detect_results;
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    post_process_yolo(outputs.data(), g_output_attrs, g_io_num.n_output,
                      conf_threshold, nms_threshold, detect_results);

    // 释放 NPU 输出缓冲区
    rknn_outputs_release(g_ctx, g_io_num.n_output, outputs.data());

    auto end_time = std::chrono::high_resolution_clock::now();
    long long cost_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    LOGD("推理+后处理耗时: %lld ms, 检出目标数: %zu", cost_ms, detect_results.size());

    // 打包为 FloatArray 返回 Java（每 6 个一组：x1,y1,x2,y2,score,class_id）
    int num_boxes = (int) detect_results.size();
    int elem_per_box = 6;
    jfloatArray j_result = env->NewFloatArray(num_boxes * elem_per_box);
    if (j_result != nullptr && num_boxes > 0) {
        std::vector<float> temp_buf(num_boxes * elem_per_box);
        for (int i = 0; i < num_boxes; i++) {
            temp_buf[i * 6 + 0] = detect_results[i].x1;
            temp_buf[i * 6 + 1] = detect_results[i].y1;
            temp_buf[i * 6 + 2] = detect_results[i].x2;
            temp_buf[i * 6 + 3] = detect_results[i].y2;
            temp_buf[i * 6 + 4] = detect_results[i].score;
            temp_buf[i * 6 + 5] = (float) detect_results[i].class_id;
        }
        env->SetFloatArrayRegion(j_result, 0, num_boxes * elem_per_box, temp_buf.data());
    }
    return j_result;
}

// ======================== 4. 资源释放（生命周期闭环）=======================
extern "C" JNIEXPORT void JNICALL
Java_com_example_rknndetector_RknnDriver_releaseModel(JNIEnv *env, jobject thiz) {
    if (g_ctx) {
        rknn_destroy(g_ctx);
        g_ctx = 0;
        is_model_loaded = false;
        if (g_input_attrs) { free(g_input_attrs); g_input_attrs = nullptr; }
        if (g_output_attrs) { free(g_output_attrs); g_output_attrs = nullptr; }
        LOGD("RKNN 资源已安全销毁");
    }
}
