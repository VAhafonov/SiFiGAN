import argparse
import sys

import tensorrt as trt

from sifigan.nv_triton.misc.utils_funcs import parse_bool


def onnx_to_tensorrt_main(onnx_model_path: str, trt_model_pah: str, fp16: bool):

    print("making plan for", onnx_model_path)

    logger = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024 * 12)  # 12 gigs
    config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)

    # Create an optimization profile for dynamic input shapes
    profile = builder.create_optimization_profile()
    profile.set_shape("INPUT__0", (1, 1, 1200), (1, 1, 12000), (1, 1, 1200000))
    profile.set_shape("INPUT__1", (1, 43, 10), (1, 43, 100), (1, 43, 10000))
    profile.set_shape("INPUT__2", (1, 4, 1200), (1, 4, 12000), (1, 4, 1200000))
    config.add_optimization_profile(profile)
    print(config)

    parser = trt.OnnxParser(network, logger)
    ok = parser.parse_from_file(onnx_model_path)
    if not ok:
        sys.exit("ONNX parse error")

    plan = builder.build_serialized_network(network, config)
    with open(trt_model_pah, "wb") as fp:
        fp.write(plan)
    print("DONE", onnx_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_models_path", type=str)
    parser.add_argument("trt_models_dir", type=str)
    parser.add_argument("--fp16", type=str, default='false')
    _args = parser.parse_args()
    onnx_to_tensorrt_main(_args.onnx_models_path, _args.trt_models_dir, parse_bool(_args.fp16))
