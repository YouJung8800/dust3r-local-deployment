import rerun as rr
from pathlib import Path
import inspect
import torch
import numpy as np
import mini_dust3r.api as api
from mini_dust3r.model import AsymmetricCroCo3DStereo

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"디바이스: {device}")

    print("모델 로드 중...")
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

    # ---- 결과 객체 안에 뭐가 들어있는지 확인 ----
    print("\n=== OptimizedResult 안의 데이터 ===")
    attrs = [a for a in dir(result) if not a.startswith("_")]
    print("사용 가능한 속성:", attrs)

    # 자주 쓰이는 속성 이름들을 시도해서 점군 정보 추출
    for name in ["point_cloud", "pts3d", "points3d", "world_points"]:
        if hasattr(result, name):
            pts = getattr(result, name)
            print(f"\n'{name}' 발견 — shape/타입: {type(pts)}")
            try:
                arr = np.asarray(pts)
                print(f"  점 개수: {arr.reshape(-1, arr.shape[-1]).shape[0]}개")
                print(f"  좌표 범위 X: {arr[..., 0].min():.3f} ~ {arr[..., 0].max():.3f}")
                print(f"  좌표 범위 Y: {arr[..., 1].min():.3f} ~ {arr[..., 1].max():.3f}")
                print(f"  좌표 범위 Z: {arr[..., 2].min():.3f} ~ {arr[..., 2].max():.3f}")
            except Exception as e:
                print(f"  (배열 변환 실패: {e})")

    # 카메라 포즈 정보 확인
    for name in ["camera_poses", "poses", "world_from_cam"]:
        if hasattr(result, name):
            print(f"\n'{name}' 발견 (카메라 포즈 정보)")

    # ---- rerun으로 실시간 뷰어 표시 (기존과 동일) ----
    rr.init("dust3r_demo", spawn=True)
    api.log_optimized_result(result, Path("world"))

    # ---- .rrd 파일로 저장 (나중에 다시 열어볼 수 있음) ----
    rr.save("dust3r_result.rrd")
    print("\n결과 저장 완료: dust3r_result.rrd")
    print("나중에 'rerun dust3r_result.rrd' 명령어로 다시 열어볼 수 있습니다.")

    print("\n완료! rerun 뷰어 창에서 3D 결과를 확인하세요.")

if __name__ == "__main__":
    main()
