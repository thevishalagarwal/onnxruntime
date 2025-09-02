---
title: NVIDIA - TensorRT RTX
description: Instructions to execute ONNX Runtime on NVIDIA RTX GPUs with the NVIDIA TensorRT RTX execution provider
parent: Execution Providers
nav_order: 17
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
{:toc}

## Install

Currently, TensorRT RTX EP can be only built from source code. Support for installation from package managers, such as PyPi and NuGet, is coming soon.

## Build from source

Information on how to build from source for TensorRT RTX EP can be found [here](../build/eps.md#nvidia-tensorrt-rtx).

## Usage

### C/C++
```c++
const auto& api = Ort::GetApi();
Ort::SessionOptions session_options;
api.SessionOptionsAppendExecutionProvider(session_options, "NvTensorRtRtx", nullptr, nullptr, 0);
Ort::Session session(env, model_path, session_options);
```

### Python
With Python APIs, you must explicitly register the TensorRT RTX EP when instantiating the `InferenceSession`.

```python
import onnxruntime as ort
sess = ort.InferenceSession('model.onnx', providers=['NvTensorRtRtxExecutionProvider'])
```

## Execution Provider Options
TensorRT RTX EP provides the following user configurable options with the [Execution Provider Options](./TensorRTRTX-ExecutionProvider.md#execution-provider-options)


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

* 



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

sess_opt = ort.SessionOptions()
sess = ort.InferenceSession(model_path, sess_options=sess_opt, providers=[('NvTensorRTRTXExecutionProvider', provider_options)])
```

</details>

#### Click below for C++ API example:

<details>

```c++
Ort::SessionOptions session_options;

cudaStream_t cuda_stream;
cudaStreamCreate(&cuda_stream);

// Need to put the CUDA stream handle in a string
char streamHandle[32];
sprintf_s(streamHandle, "%lld", (uint64_t)cuda_stream);

const auto& api = Ort::GetApi();
std::vector<const char*> option_keys = {
    "device_id",
    "user_compute_stream",  // this implicitly sets "has_user_compute_stream"
};
std::vector<const char*> option_values = {
    "1",
    streamHandle
};

Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider(session_options, "NvTensorRtRtx", option_keys.data(), option_values.data(), option_keys.size()));

```

</details>





> Note: for bool type options, assign them with **True**/**False** in python, or **1**/**0** in C++.


#### Profile shape options

* Description: build with explicit dynamic shapes using a profile with the min/max/opt shapes provided.
  * By default TensorRT RTX engines will support dynamic shapes, for perofmance improvements it is possible to specify one or multiple explicit ranges of shapes.
  * The format of the profile shapes is `input_tensor_1:dim_1xdim_2x...,input_tensor_2:dim_3xdim_4x...,...`
    * These three flags should all be provided in order to enable explicit profile shapes feature.
  * Note that multiple TensorRT RTX profiles can be enabled by passing multiple shapes for the same input tensor.
  * Check TensorRT doc [optimization profiles](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#opt_profiles) for more details.


## Cache

TensorRT RTX EP supports two types of model cache. These helps in reducing model load time significantly.

1. EP context model
2. Runtime cache


### EP context model

TensorRT RTX separates compilation into two phases - ahead of time (AOT) and just in time (JIT) compilation. In AOT phase, the ONNX model is compiled to an optimized binary blob and stored as an EP context model. This model will be compatible across multiple GPU generations.

During inference, we only use the compiled EP context model. When loaded, TensorRT RTX will JIT compile the binary blob (engine) to fit to the used GPU. This JIT process is accelerated by TensorRT RTX's internal cache.

For an example usage see:
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/providers/nv_tensorrt_rtx/nv_basic_test.cc


## Performance test

When using [onnxruntime_perf_test](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/test/perftest#onnxruntime-performance-test), use the flag `-e nvtensorrttrx`.


### TensorRT RTX Plugins Support
TensorRT RTX doesn't support plugins.