import rerun as rr
from pathlib import Path
import inspect
import torch
import mini_dust3r.api as api
from mini_dust3r.model import AsymmetricCroCo3DStereo

def main():
    if torch.backends.mps.is_available():
        device = "mps"
        print("Apple Silicon GPU(MPS) 사용")
    else:
        device = "cpu"
        print("CPU 사용")

    print("모델 로드 중...")
    model = AsymmetricCroCo3DStereo.from_pretrained(
        "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)

    image_dir = Path("images")

    sig = inspect.signature(api.inferece_dust3r)
    params = list(sig.parameters.keys())
    print("inferece_dust3r 파라미터:", params)

    call_kwargs = {"model": model, "device": device}
    first_param = params[0]
    call_kwargs[first_param] = image_dir

    if "batch_size" in params:
        call_kwargs["batch_size"] = 1

    print("3D 복원 실행 중...", call_kwargs)
    optimized_result = api.inferece_dust3r(**call_kwargs)

    rr.init("dust3r_demo", spawn=True)
    api.log_optimized_result(optimized_result, Path("world"))
    print("완료!")

if __name__ == "__main__":
    main()
