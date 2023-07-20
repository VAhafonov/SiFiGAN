import numpy as np
import sys

import torch
import tritonclient.grpc as grpcclient


def main():
    try:
        keepalive_options = grpcclient.KeepAliveOptions(
            keepalive_time_ms=2**31 - 1,
            keepalive_timeout_ms=20000,
            keepalive_permit_without_calls=False,
            http2_max_pings_without_data=2
        )
        triton_client = grpcclient.InferenceServerClient(
            url='localhost:8001',
            verbose=False,
            keepalive_options=keepalive_options)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    # read temp data from disk
    path_to_tensor_dict = '/home/ubuntu/test_tensor.pth'
    tensor_dict = torch.load(path_to_tensor_dict, map_location='cpu')
    input_0 = tensor_dict['in_signal']
    print("0", input_0.shape)
    input_1 = tensor_dict['c']
    print("1", input_1.shape)
    input_2 = tensor_dict['dfs']
    print("2", input_2.shape)
    input_3 = tensor_dict['true_lengths']
    print("3", input_3.shape)

    model_name = "sifigan"
    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('INPUT__0', list(input_0.shape), "FP32"))
    inputs.append(grpcclient.InferInput('INPUT__1', list(input_1.shape), "FP32"))
    inputs.append(grpcclient.InferInput('INPUT__2', list(input_2.shape), "FP32"))
    inputs.append(grpcclient.InferInput('INPUT__3', list(input_3.shape), "INT64"))

    # # batch size = 1, signal length = 48000
    # input0_data = np.random.randn(1, 48000).astype(np.float32)

    # Initialize the data
    inputs[0].set_data_from_numpy(input_0.numpy())
    inputs[1].set_data_from_numpy(input_1.numpy())
    inputs[2].set_data_from_numpy(input_2.numpy())
    inputs[3].set_data_from_numpy(input_3.numpy())

    outputs.append(grpcclient.InferRequestedOutput('OUTPUT__0'))

    # Test with outputs
    results = triton_client.infer(model_name=model_name,
                                    inputs=inputs,
                                    outputs=outputs,
                                    headers={'test': '1'})

    # Get the output arrays from the results
    output0_data = results.as_numpy('OUTPUT__0')
    pytorch_output = tensor_dict['y'].numpy()
    print(output0_data.shape)
    print(pytorch_output.shape)
    # print(output0_data)


if __name__ == "__main__":
    main()