from pathlib import Path
import qai_hub as hub


def compile_model():
    source_model = Path("models/mobilenet_v2_embedded.onnx")
    output_model = Path("models/mobilenet_v2_qualcomm.onnx")

    if not source_model.exists():
        raise FileNotFoundError(source_model)

    device = hub.Device("Snapdragon X Elite CRD")

    print("Submitting Qualcomm AI Hub compilation...")
    print("Source:", source_model)
    print("Target:", device.name)

    job = hub.submit_compile_job(
        model=str(source_model),
        device=device,
        name="EdgeVision-MobileNetV2-Snapdragon-X-Elite",
        options="--target_runtime onnx",
    )

    print("Compilation job:", job.url)
    print("Waiting for compilation...")

    status = job.wait()
    print("Compilation status:", status)

    target_model = job.get_target_model()

    if target_model is None:
        raise RuntimeError(
            "Compilation did not produce a target model."
        )

    job.download_target_model(str(output_model))

    if not output_model.exists():
        raise RuntimeError("Compiled model was not downloaded.")

    print("\nCompilation successful!")
    print("Downloaded model:", output_model)
    print("File size:", output_model.stat().st_size, "bytes")


if __name__ == "__main__":
    compile_model()
