// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.
#include "core/common/path_utils.h"
#include "core/graph/onnx_protobuf.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/providers/nv_tensorrt_rtx/test_nv_trt_rtx_ep_util.h"

#include <cmath>
#include <fstream>
#include <filesystem>

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {

namespace test {

// Helper: Run session with zero-filled FP16 inputs and return output as float vector.
// Uses session.Run() with CPU tensors to ensure outputs are on CPU (not IoBinding which may return GPU tensors).
std::vector<float> RunSessionAndGetOutput(Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<const char*> input_names_c;
  std::vector<Ort::Value> input_tensors;
  std::vector<std::vector<uint16_t>> input_buffers;

  for (size_t i = 0; i < session.GetInputCount(); ++i) {
    auto name = session.GetInputNameAllocated(i, allocator);
    auto info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
    auto shape = info.GetShape();
    auto elem_type = info.GetElementType();
    for (auto& d : shape) { if (d == -1) d = 1; }
    size_t num_elements = 1;
    for (auto d : shape) num_elements *= static_cast<size_t>(d);

    input_buffers.emplace_back(num_elements, uint16_t{0});
    input_tensors.push_back(Ort::Value::CreateTensor(
        mem_info, input_buffers.back().data(), num_elements * sizeof(uint16_t),
        shape.data(), shape.size(), elem_type));
    input_names_c.push_back(_strdup(name.get()));
  }

  std::vector<const char*> output_names_c;
  for (size_t i = 0; i < session.GetOutputCount(); ++i) {
    auto name = session.GetOutputNameAllocated(i, allocator);
    output_names_c.push_back(_strdup(name.get()));
  }

  Ort::RunOptions run_options;
  auto output_tensors = session.Run(run_options,
                                     input_names_c.data(), input_tensors.data(), input_tensors.size(),
                                     output_names_c.data(), output_names_c.size());

  auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
  size_t count = type_info.GetElementCount();
  std::vector<float> result(count);

  if (type_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const uint16_t* data = output_tensors[0].GetTensorData<uint16_t>();
    for (size_t i = 0; i < count; ++i) result[i] = MLFloat16::FromBits(data[i]).ToFloat();
  } else {
    const float* data = output_tensors[0].GetTensorData<float>();
    std::copy(data, data + count, result.begin());
  }

  for (auto p : input_names_c) free(const_cast<char*>(p));
  for (auto p : output_names_c) free(const_cast<char*>(p));
  return result;
}

RegisteredEpDeviceUniquePtr AppendTrtEtxEP(Ort::SessionOptions& session_options, std::unordered_map<std::string, std::string>& option_map) {
  RegisteredEpDeviceUniquePtr nv_tensorrt_rtx_ep;
  /// Since this test runs after other tests that use registration interface this test has to use it as well
  /// windows as otherwise the kernel registry inside the EP will not be populated. The legacy APis ony call the initialize once.
  Utils::RegisterAndGetNvTensorRtRtxEp(*ort_env, nv_tensorrt_rtx_ep);
  auto ep_devices = ort_env->GetEpDevices();
  Ort::ConstEpDevice selected_device;
  for (auto& device : ep_devices) {
    if (!std::strcmp(device.EpName(), kNvTensorRTRTXExecutionProvider)) {
      selected_device = device;
    }
  }
  session_options.AppendExecutionProvider_V2(*ort_env, {selected_device}, option_map);
  return nv_tensorrt_rtx_ep;
}

std::vector<char> readBinaryFile(const PathString& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + PathToUTF8String(filename));
  }

  file.seekg(0, std::ios::end);
  std::streamsize filesize = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(filesize);
  if (!file.read(reinterpret_cast<char*>(buffer.data()), filesize)) {
    throw std::runtime_error("Could not read file: " + PathToUTF8String(filename));
  }

  return buffer;
}

struct CompileParam {
  bool embed_mode;
  bool bytestream_io;
  bool external_initialzier_for_parser = false;
  const std::string to_string() const {
    return "embed_mode_" + std::to_string(embed_mode) + "_bytestream_io_" + std::to_string(bytestream_io) + "_ext_init_" + std::to_string(external_initialzier_for_parser);
    ;
  }
};
class CompileApiTest
    : public testing::TestWithParam<CompileParam> {
 public:
  const CompileParam& GetCompileParam() const {
    return GetParam();
  }
};

void SmallModelTest(CompileParam test_param, bool fully_supported_model) {
  std::string test_name = test_param.to_string();
  if (!fully_supported_model)
    test_name += "_fast_gelu";
  PathString model_name = path_utils::MakePathString("nv_execution_provider_compile_" + test_name + ".onnx");
  PathString model_name_ctx = path_utils::MakePathString("nv_execution_provider_compile_" + test_name + "_ctx.onnx");
  clearFileIfExists(model_name_ctx);
  std::string graph_name = "test";
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model_name, graph_name, dims, !fully_supported_model);

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> option_map{
      {onnxruntime::nv::provider_option_names::kUseExternalDataInitializer, std::to_string(test_param.external_initialzier_for_parser)}};
  auto ep = AppendTrtEtxEP(session_options, option_map);

  Ort::ModelCompilationOptions model_compile_options(*ort_env, session_options);
  model_compile_options.SetEpContextEmbedMode(test_param.embed_mode);

  void* output_context = nullptr;
  size_t output_context_size = 0;
  std::vector<char> input_onnx;
  if (test_param.bytestream_io) {
    input_onnx = readBinaryFile(model_name);
    model_compile_options.SetInputModelFromBuffer(input_onnx.data(), input_onnx.size());
    model_compile_options.SetOutputModelBuffer(Ort::AllocatorWithDefaultOptions(), &output_context, &output_context_size);
  } else {
    model_compile_options.SetInputModelPath(model_name.c_str());
    model_compile_options.SetOutputModelPath(model_name_ctx.c_str());
  }
  // AOT time
  ASSERT_TRUE(Ort::CompileModel(*ort_env, model_compile_options).IsOK());

  // JIT time
  Ort::Session session_object{nullptr};
  if (test_param.bytestream_io) {
    session_object = Ort::Session(*ort_env, output_context, output_context_size, session_options);
  } else {
    session_object = Ort::Session(*ort_env, model_name_ctx.c_str(), session_options);
  }
  auto io_binding = generate_io_binding(session_object);
  Ort::RunOptions run_options;
  session_object.Run(run_options, io_binding);
}

TEST_P(CompileApiTest, SmallModel) {
  const auto& test_param = GetCompileParam();
  SmallModelTest(test_param, true);
}

TEST_P(CompileApiTest, SmallSplitModel) {
  const auto& test_param = GetCompileParam();
  SmallModelTest(test_param, false);
}

TEST_P(CompileApiTest, LargeModel) {
  const auto& test_param = GetCompileParam();
  // with embed mode == 1 the resulting file will be over the 2GB proto limit
  if (test_param.embed_mode == 1) {
    GTEST_SKIP();
  }
  std::string test_name = test_param.to_string();
  PathString model_name = path_utils::MakePathString("nv_execution_provider_compile_large_" + test_name + ".onnx");
  PathString external_data_name = path_utils::MakePathString("nv_execution_provider_compile_large_" + test_name + ".onnx_data");
  PathString model_name_ctx = path_utils::MakePathString("nv_execution_provider_compile_large_" + test_name + "_ctx.onnx");
  PathString model_name_ctx_data = path_utils::MakePathString("nv_execution_provider_compile_large_" + test_name + "_ctx.onnx_data");
  clearFileIfExists(model_name_ctx);
  clearFileIfExists(model_name_ctx_data);
  // This accelerates test iterations if the large model was already generated
  if (!std::filesystem::exists(model_name) || !std::filesystem::exists(external_data_name)) {
    CreateSimpleMlpModel(model_name, external_data_name, 32, 4096);
  }

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> option_map{
      {onnxruntime::nv::provider_option_names::kUseExternalDataInitializer,
       std::to_string(test_param.bytestream_io || test_param.external_initialzier_for_parser)}};
  auto ep = AppendTrtEtxEP(session_options, option_map);

  Ort::ModelCompilationOptions model_compile_options(*ort_env, session_options);
  model_compile_options.SetEpContextEmbedMode(test_param.embed_mode);

  void* output_context = nullptr;
  size_t output_context_size = 0;
  std::vector<char> input_onnx, input_data;
  std::vector<PathString> file_names;
  std::vector<char*> file_buffers;
  std::vector<size_t> lengths;
  if (test_param.bytestream_io) {
    input_onnx = readBinaryFile(model_name);
    input_data = readBinaryFile(external_data_name);
    file_names = {external_data_name};
    file_buffers = {input_data.data()};
    lengths = {input_data.size()};
    session_options.AddExternalInitializersFromFilesInMemory(file_names, file_buffers, lengths);

    model_compile_options.SetInputModelFromBuffer(input_onnx.data(), input_onnx.size());
    model_compile_options.SetOutputModelBuffer(Ort::AllocatorWithDefaultOptions(), &output_context, &output_context_size);
  } else {
    model_compile_options.SetInputModelPath(model_name.c_str());
    model_compile_options.SetOutputModelPath(model_name_ctx.c_str());
    model_compile_options.SetOutputModelExternalInitializersFile(model_name_ctx_data.c_str(), 1024);
  }

  // AOT time
  ASSERT_TRUE(Ort::CompileModel(*ort_env, model_compile_options).IsOK());

  // JIT time
  std::unique_ptr<Ort::Session> session;
  if (test_param.bytestream_io) {
    session = std::make_unique<Ort::Session>(*ort_env, output_context, output_context_size, session_options);
  } else {
    session = std::make_unique<Ort::Session>(*ort_env, model_name_ctx.c_str(), session_options);
  }

  auto io_binding = generate_io_binding(*session);
  Ort::RunOptions run_options;
  session->Run(run_options, io_binding);
}

INSTANTIATE_TEST_SUITE_P(
    NvExecutionProviderTest, CompileApiTest,
    ::testing::Values(
        CompileParam{true, false},
        CompileParam{false, false},
        CompileParam{true, true},
        CompileParam{false, true},
        // test with external initializers for parser
        CompileParam{true, true, true},
        CompileParam{true, false, true}),
    [](const testing::TestParamInfo<CompileApiTest::ParamType>& info) {
      return info.param.to_string();
    });

/*
 * Test: Weight-stripped engine produces same output as normal engine.
 *
 * Uses CreateLargeLLMModel which generates a multi-layer MLP with FP16 MatMul weight
 * initializers stored as external data. This ensures the model has real weights that
 * TensorRT can strip from the engine plan (kSTRIP_PLAN) and refit at load time.
 *
 * Compiles the same model twice (normal and weight-stripped), runs inference on both
 * with identical zero-initialized inputs, and verifies that all output values match.
 */
TEST(NvExecutionProviderTest, WeightStrippedEngine_OutputMatchesNormal) {
  PathString model_name = path_utils::MakePathString("nv_ep_weight_stripped_test.onnx");
  PathString external_data_name = path_utils::MakePathString("nv_ep_weight_stripped_test.onnx_data");
  PathString ctx_normal = path_utils::MakePathString("nv_ep_weight_stripped_test_normal_ctx.onnx");
  PathString ctx_stripped = path_utils::MakePathString("nv_ep_weight_stripped_test_stripped_ctx.onnx");
  clearFileIfExists(ctx_normal);
  clearFileIfExists(ctx_stripped);

  // Reuse model if already generated from a previous test run
  if (!std::filesystem::exists(model_name) || !std::filesystem::exists(external_data_name)) {
    CreateSimpleMlpModel(model_name, external_data_name, 32, 4096);
  }

  // Helper to compile and run, returning output values
  auto compile_and_run = [&](bool weight_stripped, const PathString& ctx_path) -> std::vector<float> {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> option_map{
        {onnxruntime::nv::provider_option_names::kUseExternalDataInitializer, "1"}};
    if (weight_stripped) {
      option_map[onnxruntime::nv::provider_option_names::kWeightStrippedEngineEnable] = "1";
    }
    auto ep = AppendTrtEtxEP(session_options, option_map);

    // Compile
    Ort::ModelCompilationOptions compile_opts(*ort_env, session_options);
    compile_opts.SetEpContextEmbedMode(false);
    compile_opts.SetInputModelPath(model_name.c_str());
    compile_opts.SetOutputModelPath(ctx_path.c_str());
    EXPECT_TRUE(Ort::CompileModel(*ort_env, compile_opts).IsOK());

    // Load and run using session.Run() with CPU tensors (not IoBinding)
    Ort::Session session(*ort_env, ctx_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    // Build deterministic input (zero-filled FP16)
    std::vector<const char*> input_names_c;
    std::vector<Ort::Value> input_tensors;
    std::vector<std::vector<uint16_t>> input_buffers;
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (size_t i = 0; i < session.GetInputCount(); ++i) {
      auto name = session.GetInputNameAllocated(i, allocator);
      auto info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
      auto shape = info.GetShape();
      auto elem_type = info.GetElementType();
      for (auto& d : shape) { if (d == -1) d = 1; }
      size_t num_elements = 1;
      for (auto d : shape) num_elements *= static_cast<size_t>(d);

      input_buffers.emplace_back(num_elements, uint16_t{0});  // zero-filled FP16
      input_tensors.push_back(Ort::Value::CreateTensor(
          mem_info, input_buffers.back().data(), num_elements * sizeof(uint16_t),
          shape.data(), shape.size(), elem_type));
      input_names_c.push_back(_strdup(name.get()));
    }

    std::vector<const char*> output_names_c;
    for (size_t i = 0; i < session.GetOutputCount(); ++i) {
      auto name = session.GetOutputNameAllocated(i, allocator);
      output_names_c.push_back(_strdup(name.get()));
    }

    Ort::RunOptions run_options;
    auto output_tensors = session.Run(run_options,
                                       input_names_c.data(), input_tensors.data(), input_tensors.size(),
                                       output_names_c.data(), output_names_c.size());

    EXPECT_FALSE(output_tensors.empty());
    EXPECT_TRUE(output_tensors[0].IsTensor());

    auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t count = type_info.GetElementCount();

    // Copy to float vector for comparison
    std::vector<float> result(count);
    auto elem_type = type_info.GetElementType();
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      const uint16_t* data = output_tensors[0].GetTensorData<uint16_t>();
      for (size_t i = 0; i < count; ++i) {
        result[i] = MLFloat16::FromBits(data[i]).ToFloat();
      }
    } else {
      const float* data = output_tensors[0].GetTensorData<float>();
      std::copy(data, data + count, result.begin());
    }

    for (auto p : input_names_c) free(const_cast<char*>(p));
    for (auto p : output_names_c) free(const_cast<char*>(p));
    return result;
  };

  // Run both modes
  auto output_normal = compile_and_run(false, ctx_normal);
  auto output_stripped = compile_and_run(true, ctx_stripped);

  // Verify outputs match
  ASSERT_EQ(output_normal.size(), output_stripped.size());
  ASSERT_FALSE(output_normal.empty());

  float max_diff = 0.0f;
  for (size_t i = 0; i < output_normal.size(); ++i) {
    float diff = std::abs(output_normal[i] - output_stripped[i]);
    max_diff = std::max(max_diff, diff);
  }

  EXPECT_LT(max_diff, 0.001f)
      << "Max absolute difference between normal and weight-stripped outputs: " << max_diff;

  // Cleanup context files (keep the large model for reuse across test runs)
  clearFileIfExists(ctx_normal);
  clearFileIfExists(ctx_stripped);
  for (auto& entry : std::filesystem::directory_iterator(".")) {
    if (entry.path().extension() == ".engine" &&
        entry.path().string().find("NvTensorRTRTX") != std::string::npos) {
      std::filesystem::remove(entry.path());
    }
  }
}

/*
 * Test: Weight-stripped engine file is significantly smaller than normal engine.
 *
 * Compiles the same LLM model in both modes and compares .engine file sizes.
 * Weight-stripped engines should be orders of magnitude smaller since they
 * contain only the network structure, not the weight data.
 */
TEST(NvExecutionProviderTest, WeightStrippedEngine_EngineSizeReduction) {
  PathString model_name = path_utils::MakePathString("nv_ep_weight_stripped_size_test.onnx");
  PathString external_data_name = path_utils::MakePathString("nv_ep_weight_stripped_size_test.onnx_data");
  PathString ctx_normal = path_utils::MakePathString("nv_ep_weight_stripped_size_normal_ctx.onnx");
  PathString ctx_stripped = path_utils::MakePathString("nv_ep_weight_stripped_size_stripped_ctx.onnx");
  clearFileIfExists(ctx_normal);
  clearFileIfExists(ctx_stripped);

  // Use hidden_dim=2048 so weights (~64MB) dominate engine size and stripping shows a clear reduction
  if (!std::filesystem::exists(model_name) || !std::filesystem::exists(external_data_name)) {
    CreateSimpleMlpModel(model_name, external_data_name, 32, 4096);
  }

  auto compile_model = [&](bool weight_stripped, const PathString& ctx_path) {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> option_map{
        {onnxruntime::nv::provider_option_names::kUseExternalDataInitializer, "1"}};
    if (weight_stripped) {
      option_map[onnxruntime::nv::provider_option_names::kWeightStrippedEngineEnable] = "1";
    }
    auto ep = AppendTrtEtxEP(session_options, option_map);

    Ort::ModelCompilationOptions compile_opts(*ort_env, session_options);
    compile_opts.SetEpContextEmbedMode(false);
    compile_opts.SetInputModelPath(model_name.c_str());
    compile_opts.SetOutputModelPath(ctx_path.c_str());
    ASSERT_TRUE(Ort::CompileModel(*ort_env, compile_opts).IsOK());
  };

  // Find engine files by scanning for .engine files with NvTensorRTRTX in the name
  auto find_engine_size = []() -> uintmax_t {
    uintmax_t total = 0;
    for (auto& entry : std::filesystem::directory_iterator(".")) {
      if (entry.path().extension() == ".engine" &&
          entry.path().string().find("NvTensorRTRTX") != std::string::npos) {
        total += entry.file_size();
      }
    }
    return total;
  };

  auto cleanup_engines = []() {
    for (auto& entry : std::filesystem::directory_iterator(".")) {
      if (entry.path().extension() == ".engine" &&
          entry.path().string().find("NvTensorRTRTX") != std::string::npos) {
        std::filesystem::remove(entry.path());
      }
    }
  };

  // Compile normal
  cleanup_engines();
  compile_model(false, ctx_normal);
  uintmax_t normal_size = find_engine_size();
  ASSERT_GT(normal_size, 0u) << "Normal engine file should exist";

  // Compile weight-stripped
  cleanup_engines();
  compile_model(true, ctx_stripped);
  uintmax_t stripped_size = find_engine_size();
  ASSERT_GT(stripped_size, 0u) << "Weight-stripped engine file should exist";

  // Weight-stripped should be at least 10x smaller
  EXPECT_LT(stripped_size, normal_size / 10)
      << "Weight-stripped engine (" << stripped_size << " bytes) should be much smaller than "
      << "normal engine (" << normal_size << " bytes)";

  // Cleanup
  cleanup_engines();
  clearFileIfExists(ctx_normal);
  clearFileIfExists(ctx_stripped);
}

/*
 * Test: Load pre-compiled weight-stripped context model and refit from ONNX bytestream.
 *
 * This mirrors the TensorRT EP's "Refit weightless context model with ONNX in memory" test.
 * Step 1: Compile a weight-stripped context model
 * Step 2: Load the context model in a new session with the ONNX model provided as bytestream
 * Step 3: Run inference and verify output matches the normal compilation
 */
TEST(NvExecutionProviderTest, WeightStrippedEngine_RefitFromBytestream) {
  PathString model_name = path_utils::MakePathString("nv_ep_weight_stripped_refit_bs.onnx");
  PathString external_data_name = path_utils::MakePathString("nv_ep_weight_stripped_refit_bs.onnx_data");
  PathString ctx_normal = path_utils::MakePathString("nv_ep_weight_stripped_refit_bs_normal_ctx.onnx");
  PathString ctx_stripped = path_utils::MakePathString("nv_ep_weight_stripped_refit_bs_stripped_ctx.onnx");
  clearFileIfExists(ctx_normal);
  clearFileIfExists(ctx_stripped);

  if (!std::filesystem::exists(model_name) || !std::filesystem::exists(external_data_name)) {
    CreateSimpleMlpModel(model_name, external_data_name, 32, 4096);
  }

  // Step 1: Compile normal context model for reference output
  std::vector<float> output_normal;
  {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> option_map{
        {onnxruntime::nv::provider_option_names::kUseExternalDataInitializer, "1"}};
    auto ep = AppendTrtEtxEP(session_options, option_map);

    Ort::ModelCompilationOptions compile_opts(*ort_env, session_options);
    compile_opts.SetEpContextEmbedMode(false);
    compile_opts.SetInputModelPath(model_name.c_str());
    compile_opts.SetOutputModelPath(ctx_normal.c_str());
    ASSERT_TRUE(Ort::CompileModel(*ort_env, compile_opts).IsOK());

    Ort::Session session(*ort_env, ctx_normal.c_str(), session_options);
    output_normal = RunSessionAndGetOutput(session);
  }

  // Step 2: Compile weight-stripped context model
  {
    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> option_map{
        {onnxruntime::nv::provider_option_names::kUseExternalDataInitializer, "1"},
        {onnxruntime::nv::provider_option_names::kWeightStrippedEngineEnable, "1"}};
    auto ep = AppendTrtEtxEP(session_options, option_map);

    Ort::ModelCompilationOptions compile_opts(*ort_env, session_options);
    compile_opts.SetEpContextEmbedMode(false);
    compile_opts.SetInputModelPath(model_name.c_str());
    compile_opts.SetOutputModelPath(ctx_stripped.c_str());
    ASSERT_TRUE(Ort::CompileModel(*ort_env, compile_opts).IsOK());
  }

  // Step 3: Load the weight-stripped context model with ONNX bytestream for refit
  std::vector<float> output_refit;
  {
    auto onnx_bytes = readBinaryFile(model_name);
    auto ext_data_bytes = readBinaryFile(external_data_name);

    Ort::SessionOptions session_options;
    std::unordered_map<std::string, std::string> option_map{
        {onnxruntime::nv::provider_option_names::kUseExternalDataInitializer, "1"},
        {onnxruntime::nv::provider_option_names::kWeightStrippedEngineEnable, "1"},
        {onnxruntime::nv::provider_option_names::kONNXBytestream, std::to_string(reinterpret_cast<size_t>(onnx_bytes.data()))},
        {onnxruntime::nv::provider_option_names::kONNXBytestreamSize, std::to_string(onnx_bytes.size())},
        {onnxruntime::nv::provider_option_names::kExternalDataBytestream, std::to_string(reinterpret_cast<size_t>(ext_data_bytes.data()))},
        {onnxruntime::nv::provider_option_names::kExternalDataBytestreamSize, std::to_string(ext_data_bytes.size())}};
    auto ep = AppendTrtEtxEP(session_options, option_map);

    Ort::Session session(*ort_env, ctx_stripped.c_str(), session_options);
    output_refit = RunSessionAndGetOutput(session);
  }

  // Verify outputs match
  ASSERT_EQ(output_normal.size(), output_refit.size());
  float max_diff = 0.0f;
  for (size_t i = 0; i < output_normal.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(output_normal[i] - output_refit[i]));
  }
  EXPECT_LT(max_diff, 0.001f)
      << "Bytestream-refitted output should match normal output. Max diff: " << max_diff;

  // Cleanup
  clearFileIfExists(ctx_normal);
  clearFileIfExists(ctx_stripped);
  for (auto& entry : std::filesystem::directory_iterator(".")) {
    if (entry.path().extension() == ".engine" &&
        entry.path().string().find("NvTensorRTRTX") != std::string::npos) {
      std::filesystem::remove(entry.path());
    }
  }
}

/*
 * Helper to create a synthetic EPContext ONNX model with a specific "source" attribute.
 * Uses raw ONNX protobuf to bypass schema validation (EPContext is a contrib op).
 */
void CreateSyntheticEPContextModel(const PathString& model_path,
                                   const std::string& source_attr,
                                   bool include_source_attr = true) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(11);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain("com.microsoft");
  ms_opset->set_version(1);

  auto* graph = model.mutable_graph();
  graph->set_name("EPContextSourceTest");

  // Input
  auto* input = graph->add_input();
  input->set_name("input");
  auto* input_type = input->mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  input_type->mutable_shape()->add_dim()->set_dim_value(1);
  input_type->mutable_shape()->add_dim()->set_dim_value(3);

  // Output
  auto* output = graph->add_output();
  output->set_name("output");
  auto* output_type = output->mutable_type()->mutable_tensor_type();
  output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  output_type->mutable_shape()->add_dim()->set_dim_value(1);
  output_type->mutable_shape()->add_dim()->set_dim_value(3);

  // EPContext node
  auto* node = graph->add_node();
  node->set_op_type("EPContext");
  node->set_domain("com.microsoft");
  node->set_name("ep_context_node");
  node->add_input("input");
  node->add_output("output");

  // embed_mode attribute
  auto* attr_embed = node->add_attribute();
  attr_embed->set_name("embed_mode");
  attr_embed->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attr_embed->set_i(1);

  // ep_cache_context attribute (dummy data)
  auto* attr_cache = node->add_attribute();
  attr_cache->set_name("ep_cache_context");
  attr_cache->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  attr_cache->set_s("dummy_context_data");

  // source attribute (conditionally added)
  if (include_source_attr) {
    auto* attr_source = node->add_attribute();
    attr_source->set_name("source");
    attr_source->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
    attr_source->set_s(source_attr);
  }

  // Save to file
  std::ofstream ofs(model_path, std::ios::binary);
  ASSERT_TRUE(ofs.is_open());
  ASSERT_TRUE(model.SerializeToOstream(&ofs));
}

/*
 * Test: NvTensorRTRTX EP should NOT claim an EPContext node whose "source"
 * attribute belongs to a different EP (e.g., OpenVINO).
 *
 * Expected: Session initialization fails because no EP claims the node.
 */
TEST(NvExecutionProviderTest, EPContextNode_ForeignSourceSkipped) {
  PathString model_path = path_utils::MakePathString("ep_context_foreign_source_nv.onnx");
  CreateSyntheticEPContextModel(model_path, "OpenVINOExecutionProvider");

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> option_map;
  auto ep = AppendTrtEtxEP(session_options, option_map);

  // Loading a model with a foreign-source EPContext node should fail during
  // session creation because the NvTensorRTRTX EP correctly skips the node
  // and no other EP can handle it.
  try {
    Ort::Session session(*ort_env, model_path.c_str(), session_options);
    FAIL() << "Expected session creation to fail for EPContext node with foreign source";
  } catch (const Ort::Exception& e) {
    std::string error_msg = e.what();
    EXPECT_TRUE(error_msg.find("EPContext") != std::string::npos)
        << "Error should mention EPContext. Actual: " << error_msg;
  }

  // Clean up
  std::filesystem::remove(model_path);
}

/*
 * Test: NvTensorRTRTX EP should NOT claim an EPContext node whose "source"
 * attribute is set to the classic TensorRT EP name.
 */
TEST(NvExecutionProviderTest, EPContextNode_ClassicTrtSourceSkipped) {
  PathString model_path = path_utils::MakePathString("ep_context_classic_trt_source_nv.onnx");
  CreateSyntheticEPContextModel(model_path, "TensorrtExecutionProvider");

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> option_map;
  auto ep = AppendTrtEtxEP(session_options, option_map);

  try {
    Ort::Session session(*ort_env, model_path.c_str(), session_options);
    FAIL() << "Expected session creation to fail for EPContext node with classic TRT source";
  } catch (const Ort::Exception& e) {
    std::string error_msg = e.what();
    EXPECT_TRUE(error_msg.find("EPContext") != std::string::npos)
        << "Error should mention EPContext. Actual: " << error_msg;
  }

  // Clean up
  std::filesystem::remove(model_path);
}

/*
 * Test: NvTensorRTRTX EP should still claim an EPContext node that has NO
 * "source" attribute (backward compatibility with legacy context models).
 *
 * Expected: The EP claims the node. It may fail later during engine
 * deserialization (since context data is synthetic), but the error must NOT
 * be "is not compatible with any execution provider", which would indicate
 * the node was not claimed at all.
 */
TEST(NvExecutionProviderTest, EPContextNode_NoSourceAttribute_BackwardCompat) {
  PathString model_path = path_utils::MakePathString("ep_context_no_source_nv.onnx");
  CreateSyntheticEPContextModel(model_path, "", /*include_source_attr=*/false);

  Ort::SessionOptions session_options;
  std::unordered_map<std::string, std::string> option_map;
  auto ep = AppendTrtEtxEP(session_options, option_map);

  try {
    Ort::Session session(*ort_env, model_path.c_str(), session_options);
    // If session creation succeeds, backward compatibility is working.
  } catch (const Ort::Exception& e) {
    std::string error_msg = e.what();
    // The node should have been claimed by the EP. Any failure should be
    // EP-internal (e.g., bad engine data), NOT the "not compatible" error
    // that indicates no EP claimed the node.
    EXPECT_TRUE(error_msg.find("is not compatible with any execution provider") == std::string::npos)
        << "Legacy EPContext node without source should still be claimed by EP. Error: " << error_msg;
  }

  // Clean up
  std::filesystem::remove(model_path);
}

}  // namespace test
}  // namespace onnxruntime
