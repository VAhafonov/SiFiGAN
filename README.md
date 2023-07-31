# Source-Filter HiFi-GAN (SiFi-GAN)

This repo provides official PyTorch implementation of [SiFi-GAN](https://arxiv.org/abs/2210.15533), a fast and pitch controllable high-fidelity neural vocoder.<br>
This repo also provides code and instruction to serve this model via Nvidia-Triton server

# How to serve model via Nvidia-Triton
## Environment setup

Install python dependencies.
```bash
cd SiFiGAN
pip install -e .
```
Install docker https://docs.docker.com/engine/install/ <br>
Install nvidia container-toolkit https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
<br>

## Prepare model for serving
### Download pretrained checkpoint.
At this example we will create new folder ```checkpoints``` at ```sifigan/nv_triton/misc``` 
and download here pretrained checkpoint.
```bash
cd sifigan/nv_triton/misc
mkdir checkpoints
cd checkpoints
# download model from dropbox
wget -O checkpoint.pkl https://www.dropbox.com/s/w3pnnmpsxvqfykx/checkpoint-1000000steps.pkl?dl=0
# go back to sifigan/nv_triton/misc dir
cd ..
```

### Prepare pretrained checkpoint for serving
You can prepare any type of model from this section and run serving for this model. <br>
You should run following scripts from ```sifigan/nv_triton/misc``` dir.

#### Prepare JIT FP32 model
```bash
python3 model_to_jit.py checkpoints/checkpoint.pkl ../server/model-repository/sifigan-pt-fp32/1/model.pt
```
`checkpoints/checkpoint.pkl` is input checkpoint path <br>
`../server/model-repository/sifigan-pt-fp32/1/model.pt` is output path for compiled jit FP32 model

#### Prepare JIT FP16 model
```bash
python3 model_to_jit.py checkpoints/checkpoint.pkl ../server/model-repository/sifigan-pt-fp16/1/model.pt --fp16=true
```
`checkpoints/checkpoint.pkl` is input checkpoint path <br>
`../server/model-repository/sifigan-pt-fp16/1/model.pt` is output path for compiled jit FP16 model <br>
`--fp16=true` indicates that we need model in FP16 precision

#### Prepare ONNX FP32 model
For ONNX generation you need example of real input. You can download it from Dropbox.

```bash
wget -O checkpoints/test_tensor.pth https://www.dropbox.com/s/8ccrv26a2t8fed9/test_tensor.pth?dl=0
```
Run onnx generation
```bash
python3 model_to_onnx.py checkpoints/checkpoint.pkl ../server/model-repository/sifigan-onnx-fp32/1/model.onnx \
checkpoints/test_tensor.pth
```
`checkpoints/checkpoint.pkl` is input checkpoint path <br>
`../server/model-repository/sifigan-onnx-fp32/1/model.onnx` is output path for onnx FP32 model
`checkpoints/test_tensor.pth` - path to the example input for the network

#### Prepare ONNX FP16 model
```bash
python3 model_to_onnx.py checkpoints/checkpoint.pkl ../server/model-repository/sifigan-onnx-fp16/1/model.onnx \
checkpoints/test_tensor.pth --fp16=true
```
`checkpoints/checkpoint.pkl` is input checkpoint path <br>
`../server/model-repository/sifigan-onnx-fp16/1/model.onnx` is output path for onnx FP16 model<br>
`checkpoints/test_tensor.pth` - path to the example input for the network <br>
`--fp16=true` indicates that we need model in FP16 precision
<br>
<br>

**WARNING!**  Section related to TensorRT generation are completely optional. <br>
Currently decoding via TensorRT models is not supported, because TensorRT works only with static shapes.
But TensorRT models could be used for performance analyze. <br>
**IMPORTANT!** In order to generate TensorRT models you have to install TensorRT.

####  Preparation step
First of all copy model configs for trt models to nvidia-triton model-repository
```bash
cp -r trt-models-conifgs/* ../server/model-repository/
```

**IMPORTANT!** If you stumbled upon an error during generation of TensorRT models, and after that you do not plan 
to use these models then clean the directory:
```bash
rm -rf ../server/model-repository/sifigan-trt-fp32
rm -rf ../server/model-repository/sifigan-trt-fp16
```

#### Prepare TensorRT FP32 model
```bash
# generate onnx model with static shape
python3 model_to_onnx.py checkpoints/checkpoint.pkl checkpoints/model.onnx checkpoints/test_tensor.pth --use_dynamic_shape=false
# generate TensorRT plan from onnx and put it in right place
python3 onnx_to_tensorrt.py checkpoints/model.onnx ../server/model-repository/sifigan-trt-fp32/1/model.plan
```

#### Prepare TensorRT FP16 model
```bash
# generate onnx model with static shape in FP16 mode
python3 model_to_onnx.py checkpoints/checkpoint.pkl checkpoints/model.onnx checkpoints/test_tensor.pth --use_dynamic_shape=false --fp16=true
# generate TensorRT plan from onnx in FP16 mode and put it in right place
python3 onnx_to_tensorrt.py checkpoints/model.onnx ../server/model-repository/sifigan-trt-fp16/1/model.plan --fp16=true
```
<br>

### Run Nvidia-Triton inference server
To run Nvidia-Triton inference server you should go back to **root directory of the repo** and 
run following command
```bash
sudo docker run --gpus=1 --rm --net=host -p8000:8000 -p8001:8001 -p8002:8002 \
-v ${PWD}/sifigan/nv_triton/server/model-repository:/models \
nvcr.io/nvidia/tritonserver:23.06-py3 tritonserver --model-repository=/models
```
You should see following three lines at the end of the previous command output.
```console
Started GRPCInferenceService at 0.0.0.0:8001
Started HTTPService at 0.0.0.0:8000
Started Metrics Service at 0.0.0.0:8002
```
This output means that nvidia-triton server is running and you can make inference requests to it.

### Run inference requests to Nvidia-Triton server
Left terminal with server running open, and open new terminal for client requests. <br>
First of all you need to prepare input data. In following example we will download input data from dropbox 
and place it under `sifigan/nv_triton/client/data/in` folder. You could place your data where you want and just 
correct paths from examples.
#### Prepare input data
```bash
# assume that we are in root folder of the repo and go to sifigan/nv_triton/client dir
cd sifigan/nv_triton/client
mkdir -p data/in
cd data/in
# download example input data from Dropbox
wget -O example_input_data.tar https://www.dropbox.com/s/qt3jkh2r3fzuge2/example_input_data.tar?dl=0
# unpack it
tar -xvf example_input_data.tar
# delete tar file
rm example_input_data.tar
# go back to sifigan/nv_triton/client dir
cd ../..
```

#### Extract all needed features
To prepare your sound files for inference you have to use `extract_features.py` script.
```bash
python3 extract_features.py --input_dir=data/in --output_dir=data/features
```
It will take all files from `--input_dir` preprocess them and save in proper format in `--output_dir` <br>
Extracting features process is based on bunch of hyperparams, they should be located in config .yaml file.
By default, extract_features.py uses [extract_features_default.yaml](sifigan%2Fnv_triton%2Fclient%2Fconfigs%2Fextract_features_default.yaml)
config file. You can modify it or even create your own config file and pass path to it via `--path_to_config` argument.

#### Run vocoding
To run vocoding from features extracted on previous step, using Sifi GAN you should use `decode.py` script.
```bash
python3 decode.py --input_dir=data/features --output_dir=data/out --model=sifigan-pt-fp32
```
It will take all extracted features from `--input_dir` do vocoding for each of them and save resulting sound files 
into `--output_dir`. <br>
`--model` indicates to which model you are making inference request. You can only use models that 
you have generated at **Prepare pretrained checkpoint for serving** step. If you have generated all type 
models from that step than following options of values of this param is open for you:
1. `sifigan-pt-fp32`
2. `sifigan-onnx-fp32`
3. `sifigan-pt-fp16`
4. `sifigan-onnx-fp16` 


Vocoding process is also based on bunch of hyperparams, they should be located in config .yaml file.
By default, decode.py uses [decode_default.yaml](sifigan%2Fnv_triton%2Fclient%2Fconfigs%2Fdecode_default.yaml)
config file. You can modify it or even create your own config file and pass path to it via `--path_to_config` argument.

#### Verifying results
After previous step there should be output sound files in `--output_dir` directory. <br>
You can listen to them and verify if everything is fine.