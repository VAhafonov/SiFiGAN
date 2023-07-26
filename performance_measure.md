# Performance measure
<br/>

## Measure performance of baseline (not optimized) model
### FP32 torchscript model
Steps to reproduce:
1. Run nvidia triton server
```console
sudo docker run --gpus=1 --rm --net=host -p8000:8000 -p8001:8001 -p8002:8002 \ 
-v ${PWD}/sifigan/nv_triton/server/model-repository:/models nvcr.io/nvidia/tritonserver:23.06-py3 tritonserver \ 
--model-repository=/models
```

2. Open new terminal to run performance analyzing request
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