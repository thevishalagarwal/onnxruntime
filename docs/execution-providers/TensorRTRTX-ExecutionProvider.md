---
title: NVIDIA - TensorRT RTX
description: Instructions to execute ONNX Runtime on NVIDIA RTX GPUs with the NVIDIA TensorRT RTX execution provider
parent: Execution Providers
nav_order: 2
redirect_from: /docs/reference/execution-providers/TensorRTRTX-ExecutionProvider
---

# NVIDIA TensorRT RTX Execution Provider
{: .no_toc }

NVIDIA TensorRT RTX Execution Provider (EP) is the **recommended** choice for GPU acceleration on NVIDIA consumer hardware (RTX PCs). It offers a more lightweight experience than the datacenter-focused TensorRT (TRT) EP and delivers superior performance compared to the other EPs.

Here's why it's a better fit for RTX PCs than the legacy TensorRT EP:
*   **Smaller package footprint:** Optimizes resource usage.
*   **Faster model compile and load times:** Get up and running quicker.
*   **Enhanced usability:** Seamlessly use cached models across multiple RTX GPUs.

The TensorRT RTX EP leverages NVIDIA's new deep learning inference engine, [TensorRT RTX](https://developer.nvidia.com/tensorrt-rtx), to accelerate ONNX models on RTX GPUs. Microsoft and NVIDIA collaborated closely to integrate the TensorRT RTX execution provider with ONNX Runtime.

Currently, TensorRT RTX supports RTX GPUs based on Ampere and later architectures. Support for Turing GPUs is coming soon.

For compatibility and support matrix, please refer to [this](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/getting-started/support-matrix.html) page.

## Contents
{: .no_toc }

* TOC placeholder
{:toc toc_levels=1..4}

## Install

Currently, TensorRT RTX EP can be only built from source code. Support for installation from package managers, such as PyPi and NuGet, is coming soon.

## Build from source

Information on how to build from source for TensorRT RTX EP can be found [here](../build/eps.md#nvidia-tensorrt-rtx).

## Usage

### C/C++
```c++
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SampleApp");
Ort::SessionOptions session_options;
session_options.AppendExecutionProvider(onnxruntime::kNvTensorRTRTXExecutionProvider, {});
Ort::Session session(env, model_path, session_options);
```

### Python
With Python APIs, you must explicitly register the TensorRT RTX EP when instantiating the `InferenceSession`.

```python
import onnxruntime as ort
sess = ort.InferenceSession(model_path, providers=['NvTensorRtRtxExecutionProvider'])
```

## Features

### CUDA Graph

CUDA Graph Text

### EP context model

In ONNXRuntime, Execution Providers are responsible for converting ONNX models into the graph format required by its specific backend SDK and subsequently compiling them into a format compatible with the target hardware. In large models like LLMs and Diffusion models, this conversion and compilation process can be resource-intensive and time-consuming, often extending to tens of minutes. This overhead significantly impacts the user experience during session creation.

To mitigate the repetitive nature of model conversion and compilation, the ONNX models can be pre-compiled model as a binary file and persisted in an "EP Context" Model. This pre-compiled model can then be loaded directly by the EP, bypassing the initial compilation steps and enabling immediate execution on the target device. This optimization substantially reduces session creation time and enhances overall operational efficiency.

TensorRT RTX simplifies this approach by separating compilation into two distinct phases:
* Ahead-of-Time (AOT) Compilation: The ONNX model is compiled into an optimized binary blob, which is then stored as an EP context model. This generated model is designed for compatibility across multiple generations of GPUs.
* Just-in-Time (JIT) Compilation: During inference, the compiled EP context model is loaded. TensorRT RTX then performs a JIT compilation of the binary blob (engine) to precisely adapt it to the specific GPU in use.

The primary benefit of this multi-phase compilation workflow is a significant reduction in model load times.

#### Generating EP Context Models with ORT 1.22

ONNX Runtime 1.22 introduced dedicated [Compile APIs](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/session/compile_api.h) to simplify the generation of EP context models:

```cpp
Ort::ModelCompilationOptions compile_options(env, session_options);
compile_options.SetInputModelPath(input_model_path);
compile_options.SetOutputModelPath(compile_model_path);

Ort::Status status = Ort::CompileModel(env, compile_options);
```

After successful generation, the EP context model can be directly loaded for inference:

```cpp
Ort::Session session(env, compile_model_path, session_options);
```

This approach leads to a considerable reduction in session creation time, thereby improving the overall user experience.

For a practical example of usage, please refer to:
* EP context samples
* EP context [unit tests](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/providers/nv_tensorrt_rtx/nv_ep_context_test.cc)


There are two other ways to quick generate an EP context model

**ONNX Runtime Perf Test**

```sh
onnxruntime_perf_test.exe -e nvtensorrtrtx -I -r 1 "/path/to/model.onnx" --compile_ep_context --compile_model_path "/path/to/model_ctx.onnx"
```

**Python Script**

```sh
python tools/python/compile_ep_context_model.py -i "path/to/model.onnx" -o "/path/to/model_ctx.onnx"
```

#### NVIDIA recommended settings

* disable ORT graph optimization
```cpp
session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
```

* For models > 2GB, set embed_mode = 0 in model compilation options. If binary blob is embedded within the EP context, it fails for > 2GB models due to protobuf limitations
```cpp
Ort::ModelCompilationOptions compile_options(env, session_options);
compile_options.SetEpContextEmbedMode(0);
```


### Runtime cache



## Execution Provider Options
TensorRT RTX EP provides the following user configurable options with the [Execution Provider Options](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/providers/nv_tensorrt_rtx/nv_provider_options.h)


| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| device_id | `int` | GPU device identifier | 0 |
| user_compute_stream | `str` | Specify compute stream to run GPU workload | "" |
| nv_max_workspace_size | `int` | Maximum TensorRT engine workspace (bytes) | 0 (auto) |
| nv_max_shared_mem_size | `int` | Maximum TensorRT engine workspace (bytes) | 0 (auto) |
| nv_dump_subgraphs | `bool` | Enable subgraph dumping for debugging | false |
| nv_detailed_build_log | `bool` | Enable detailed build logging | false |
| enable_cuda_graph | `bool` | Enable [CUDA graph](https://developer.nvidia.com/blog/cuda-graphs/) to reduce inference overhead. Helpful for smaller models | false |
| profile_min_shapes | `str` | Comma-separated list of input tensor shapes for the minimum optimization profile. Format: `"input1:dim1xdim2x...,input2:dim1xdim2x..."` | "" (auto) |
| profile_max_shapes | `str` | Comma-separated list of input tensor shapes for the maximum optimization profile. Format: `"input1:dim1xdim2x...,input2:dim1xdim2x..."` | "" (auto) |
| profile_opt_shapes | `str` | Comma-separated list of input tensor shapes for the optimal optimization profile. Format: `"input1:dim1xdim2x...,input2:dim1xdim2x..."` | "" (auto) |
| nv_multi_profile_enable | `bool` | Enable support for multiple optimization profiles in TensorRT engine. Allows dynamic input shapes for different inference requests | false |
| nv_use_external_data_initializer | `bool` | Use external data initializer for model weights. Useful for EP context large models with external data files | false |



#### Click below for Python API example:


<details>

```python
import onnxruntime as ort

model_path = '<path to model>'

# note: for bool type options in python API, set them as False/True
provider_options = {
  'device_id': 0,
  'nv_dump_subgraphs': False,
  'nv_detailed_build_log': True,
  'user_compute_stream': stream_handle
}

sesion_options = ort.SessionOptions()
sess = ort.InferenceSession(model_path, sess_options=sesion_options, providers=[('NvTensorRTRTXExecutionProvider', provider_options)])
```
</details>


#### Click below for C++ API example:


<details>

```c++
Ort::SessionOptions session_options;

// define a cuda stream
cudaStream_t cuda_stream;
cudaStreamCreate(&cuda_stream);

char stream_handle[32];
sprintf_s(stream_handle, "%lld", (uint64_t)cuda_stream);

std::unordered_map<std::string, std::string> provider_options;
provider_options[onnxruntime::nv::provider_option_names::kDeviceId] = "1";
provider_options[onnxruntime::nv::provider_option_names::kUserComputeStream] = stream_handle;

session_options.AppendExecutionProvider(onnxruntime::kNvTensorRTRTXExecutionProvider, provider_options);
```

</details>



> Note: for bool type options, assign them with **True**/**False** in python, or **1**/**0** in C++.


#### Profile shape options

* Description: build with explicit dynamic shapes using a profile with the min/max/opt shapes provided.
  * By default TensorRT RTX engines will support dynamic shapes, for perofmance improvements it is possible to specify one or multiple explicit ranges of shapes.
  * The format of the profile shapes is `input_tensor_1:dim_1xdim_2x...,input_tensor_2:dim_3xdim_4x...,...`
    * These three flags should all be provided in order to enable explicit profile shapes feature.
  * Note that multiple TensorRT RTX profiles can be enabled by passing multiple shapes for the same input tensor.
  * Check TensorRT doc [optimization profiles](https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/work-with-dynamic-shapes.html) for more details.


## Performance test

When using [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/perftest#onnxruntime-performance-test), use the flag `-e nvtensorrttrx`


### Plugins Support
TensorRT RTX doesn't support plugins