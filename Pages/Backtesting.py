import onnxruntime

def model_compute(model_name, example_inputs, device="cpu"):

    onnx_inputs = [tensor.numpy(force=True) for tensor in example_inputs]
    # Example usage
    #ort_session = onnxruntime.InferenceSession(
    #    "./image_classifier_model.onnx", providers=["CPUExecutionProvider"]
    #)

    ort_session = onnxruntime.InferenceSession(
        f"Models/{model_name}", providers= ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    )

    onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in
                         zip(ort_session.get_inputs(), onnx_inputs)}

    # ONNX Runtime returns a list of outputs
    onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]

    return onnxruntime_outputs