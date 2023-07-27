# Performance measure

## Metodology
Steps to reproduce:
1. Run nvidia triton server from repo root dir
```console
sudo docker run --gpus=1 --rm --net=host -p8000:8000 -p8001:8001 -p8002:8002 \ 
-v ${PWD}/sifigan/nv_triton/server/model-repository:/models nvcr.io/nvidia/tritonserver:23.06-py3 tritonserver \ 
--model-repository=/models
```

2. Open new terminal to run performance analyzing request
```bash
$ cd sifigan/nv_triton/client
$ perf_analyzer -m <model-name> --input-data <path-to-file-with-real-data>
```
As file with real data for FP32 inference we will use ```measurement_data/real_data_fp32.json```. <br>
As file with real data for FP16 inference we will use ```measurement_data/real_data_fp16.json```.

## Initial results
Run this command.
```bash
$ cd sifigan/nv_triton/client
$ perf_analyzer -m sifigan-pt-fp32 --input-data measurement_data/real_data_fp32.json
```
You will get output that looks like this
```console
Successfully read data for 1 stream/streams with 1 step/steps.
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client:
    Request count: 358
    Throughput: 19.8874 infer/sec
    Avg latency: 50272 usec (standard deviation 440 usec)
    p50 latency: 50259 usec
    p90 latency: 50866 usec
    p95 latency: 50985 usec
    p99 latency: 51249 usec
    Avg HTTP time: 50264 usec (send/recv 396 usec + response wait 49868 usec)
  Server:
    Inference count: 358
    Execution count: 358
    Successful request count: 358
    Avg request latency: 48269 usec (overhead 36 usec + queue 27 usec + compute input 510 usec + compute infer 47587 usec + compute output 108 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 19.8874 infer/sec, latency 50272 usec
```

``` 19.8874 infer/sec ``` is our target performance metric. We will try to improve it via different techniques.
</br>
This measure is valid for https://github.com/VAhafonov/SiFiGAN/tree/1b530dad16dd5c64302e796cfe24ecf7a0d934e8 state of 
repository.

## First tweak
After tweak in architecture(get rid of ModuleNetInterface and index-based running of 
nn.ModuleLists) from https://github.com/VAhafonov/SiFiGAN/commit/369bbfcdd1501e35b67fdaaf087f48195dfa4cf4 commit.
</br>
Run this command.
```bash
$ cd sifigan/nv_triton/client
$ perf_analyzer -m sifigan-pt-fp32 --input-data measurement_data/real_data_fp32.json
```
We get following results
```console
Successfully read data for 1 stream/streams with 1 step/steps.
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client:
    Request count: 387
    Throughput: 21.4984 infer/sec
    Avg latency: 46509 usec (standard deviation 2616 usec)
    p50 latency: 46229 usec
    p90 latency: 46996 usec
    p95 latency: 47352 usec
    p99 latency: 48187 usec
    Avg HTTP time: 46500 usec (send/recv 415 usec + response wait 46085 usec)
  Server:
    Inference count: 387
    Execution count: 387
    Successful request count: 387
    Avg request latency: 44429 usec (overhead 39 usec + queue 30 usec + compute input 536 usec + compute infer 43712 usec + compute output 111 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 21.4984 infer/sec, latency 46509 usec
```
``` 21.4984 infer/sec ``` is our new value of our metric. Which is 8% faster than ``` 19.8874 infer/sec ``` from 
initial measurement.

## Try to use ONNX backend
After conversation to onnx and introduction onnx model as a new model in triton server, we can do perf_analyzer 
request to that model.
```bash
$ perf_analyzer -m sifigan-onnx-fp32 --input-data measurement_data/real_data_fp32.json
```
We get following results
```console
Successfully read data for 1 stream/streams with 1 step/steps.
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client:
    Request count: 319
    Throughput: 17.7209 infer/sec
    Avg latency: 56357 usec (standard deviation 765 usec)
    p50 latency: 56240 usec
    p90 latency: 57415 usec
    p95 latency: 57779 usec
    p99 latency: 58384 usec
    Avg HTTP time: 56350 usec (send/recv 375 usec + response wait 55975 usec)
  Server:
    Inference count: 319
    Execution count: 319
    Successful request count: 319
    Avg request latency: 54388 usec (overhead 25 usec + queue 26 usec + compute input 383 usec + compute infer 53893 usec + compute output 60 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 17.7209 infer/sec, latency 56357 usec
```
We see ```17.7209 infer/sec``` value and can conclude that onnx is much slower than jit. 

## Try to use TensorRT backend
I've been able to convert SiFiGAN model to TensorRT plan and serve it via nvidia-triton but only in 
static shape mode. <br>
So I've generated TensorRT plan for static shape that is equal to shape of my testing tensor for 
performance measurement. <br> <br>
I am sure that this way with serving TensorRT with static shape could be useful in real production.
You just have to choose optimal shape and implement algorithm for smart dividing input data of any shape 
to your target static shape and smart merging networks output back to original shape. <br> <br>
Lest see performance of TensorRT model.
```bash
$ perf_analyzer -m sifigan-trt-fp32 --input-data measurement_data/real_data_fp32.json
```
We get following results
```console
Successfully read data for 1 stream/streams with 1 step/steps.
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client:
    Request count: 503
    Throughput: 27.9424 infer/sec
    Avg latency: 35747 usec (standard deviation 5663 usec)
    p50 latency: 35316 usec
    p90 latency: 36146 usec
    p95 latency: 36391 usec
    p99 latency: 40187 usec
    Avg HTTP time: 35738 usec (send/recv 444 usec + response wait 35294 usec)
  Server:
    Inference count: 503
    Execution count: 503
    Successful request count: 503
    Avg request latency: 33540 usec (overhead 29 usec + queue 26 usec + compute input 504 usec + compute infer 32822 usec + compute output 158 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 27.9424 infer/sec, latency 35747 usec
```
We see ```27.9424 infer/sec``` value that is 30% faster than jit model.

## Try to use half precision(FP16) inference
I've generated jit model in fp16 mode, served it via nv-triton and made decoding.
I've seen that model output from jit in fp32 and jit in fp16 have difference in values. But 
according to my ears wav from these tensors sound same or almost same. 
I am sure that there is some metric that can tell us how much two sounds are 
similar (like audio fingerprint or like structural metrics in image processing - SSIM or gradient metric). 
Unfortunately I don't have enough time to research and implement this kind of metric. <br>
I have repository with test sounds generated from different serving models(pt-fp32, pt-16)
you can compare them. https://github.com/VAhafonov/sounds <br>
Let's try to analyze performance of jit FP16 model.
```bash
$ perf_analyzer -m sifigan-pt-fp16 --input-data measurement_data/real_data_fp16.json
```
We get following results
```console
 Successfully read data for 1 stream/streams with 1 step/steps.
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client:
    Request count: 700
    Throughput: 38.8862 infer/sec
    Avg latency: 25709 usec (standard deviation 1128 usec)
    p50 latency: 25594 usec
    p90 latency: 25823 usec
    p95 latency: 25896 usec
    p99 latency: 26224 usec
    Avg HTTP time: 25703 usec (send/recv 186 usec + response wait 25517 usec)
  Server:
    Inference count: 700
    Execution count: 700
    Successful request count: 700
    Avg request latency: 24624 usec (overhead 28 usec + queue 22 usec + compute input 237 usec + compute infer 24279 usec + compute output 57 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 38.8862 infer/sec, latency 25709 usec
```
We see ```38.8862 infer/sec``` which is 81% faster than jit FP32 and 40% faster than TensorRT FP32 with static shape.

## ONNX backend in FP16 mode
Let's try to analyze performance of onnx FP16 model.
```bash
$ perf_analyzer -m sifigan-onnx-fp16 --input-data measurement_data/real_data_fp16.json
```
We get following results
```console
 Successfully read data for 1 stream/streams with 1 step/steps.
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client:
    Request count: 663
    Throughput: 36.8306 infer/sec
    Avg latency: 27156 usec (standard deviation 215 usec)
    p50 latency: 27144 usec
    p90 latency: 27416 usec
    p95 latency: 27526 usec
    p99 latency: 27732 usec
    Avg HTTP time: 27149 usec (send/recv 198 usec + response wait 26951 usec)
  Server:
    Inference count: 663
    Execution count: 663
    Successful request count: 663
    Avg request latency: 26014 usec (overhead 24 usec + queue 23 usec + compute input 196 usec + compute infer 25734 usec + compute output 37 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 36.8306 infer/sec, latency 27156 usec
```
We see ```36.8306 infer/sec``` value and can conclude that onnx is slower than jit in the same mode. 

## TensorRT backend in FP16 mode
Run performance analyzing for TensorRT model with static shape the same as in the FP32 mode. <br>
Let's try to analyze performance of TensorRT FP16 model.
```bash
$ perf_analyzer -m sifigan-trt-fp16 --input-data measurement_data/real_data_fp16.json
```
We get following results
```console
 Successfully read data for 1 stream/streams with 1 step/steps.
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 1
  Client:
    Request count: 1146
    Throughput: 63.6623 infer/sec
    Avg latency: 15703 usec (standard deviation 3838 usec)
    p50 latency: 15479 usec
    p90 latency: 15727 usec
    p95 latency: 15837 usec
    p99 latency: 22588 usec
    Avg HTTP time: 15698 usec (send/recv 169 usec + response wait 15529 usec)
  Server:
    Inference count: 1146
    Execution count: 1146
    Successful request count: 1146
    Avg request latency: 14682 usec (overhead 24 usec + queue 26 usec + compute input 250 usec + compute infer 14292 usec + compute output 90 usec)

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 63.6623 infer/sec, latency 15703 usec
```
We see ```63.6623 infer/sec``` which is 64% faster than jit FP16 and 128% faster than TensorRT FP32 with static shape.
