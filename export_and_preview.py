import rerun as rr
from pathlib import Path
import inspect
import torch
import numpy as np
import mini_dust3r.api as api
from mini_dust3r.model import AsymmetricCroCo3DStereo
import matplotlib.pyplot as plt

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"디바이스: {device}")

    model = AsymmetricCroCo3DStereo.from_pretrained(
        "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)

    image_dir = Path("images")
    sig = inspect.signature(api.inferece_dust3r)
    params = list(sig.parameters.keys())
    call_kwargs = {"model": model, "device": device, params[0]: image_dir}
    if "batch_size" in params:
        call_kwargs["batch_size"] = 1

    print("3D 복원 실행 중...")
    result = api.inferece_dust3r(**call_kwargs)

    # 점군 데이터 찾기 (여러 가능한 속성명 시도)
    pts = None
    for name in ["point_cloud", "pts3d", "points3d", "world_points"]:
        if hasattr(result, name):
            pts = np.asarray(getattr(result, name))
            print(f"점군 데이터 발견: '{name}', shape={pts.shape}")
            break

    if pts is None:
        print("점군 속성을 자동으로 못 찾았습니다. 아래 속성 목록을 확인하세요:")
        print([a for a in dir(result) if not a.startswith("_")])
        return

    pts_flat = pts.reshape(-1, 3)
    # 너무 많으면 시각화용으로 샘플링
    if len(pts_flat) > 50000:
        idx = np.random.choice(len(pts_flat), 50000, replace=False)
        pts_flat = pts_flat[idx]

    # ---- PLY 파일로 저장 (MeshLab에서 열람 가능) ----
    with open("dust3r_pointcloud.ply", "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts_flat)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts_flat:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    print(f"PLY 저장 완료: dust3r_pointcloud.ply ({len(pts_flat)}개 점)")

    # ---- 미리보기 이미지 자동 생성 (README용) ----
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts_flat[:, 0], pts_flat[:, 1], pts_flat[:, 2],
               s=0.5, c=pts_flat[:, 2], cmap='viridis')
    ax.set_title(f"DUSt3R 3D Reconstruction ({len(pts_flat)} points)")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig("result_preview.png", dpi=150)
    print("미리보기 이미지 저장 완료: result_preview.png (README에 바로 사용 가능)")

    rr.init("dust3r_demo", spawn=True)
    api.log_optimized_result(result, Path("world"))
    print("완료!")

if __name__ == "__main__":
    main()
