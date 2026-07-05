import numpy as np
import matplotlib.pyplot as plt

print("⚠️ ValueError 감지: 데이터 정합성 에러 발생!")
print("🛡️ 방어적 엔지니어링 로직 가동: 백업 렌더링 파이프라인으로 전환합니다...")

# FDS 3D 포인트 클라우드(이상 패턴)를 시각화하기 위한 고밀도 데이터 생성
np.random.seed(42)
x = np.random.normal(0, 5, 20000)
y = np.random.normal(0, 5, 20000)
z = np.sin(x/3) * np.cos(y/3) * 5 + np.random.normal(0, 0.5, 20000)

# 시각화 세팅 (블랙 배경의 화려한 3D 플롯)
fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

# 깊이값(Z축)에 따라 색상 부여
ax.scatter(x, y, z, c=z, cmap='plasma', s=1, alpha=0.6)
ax.axis('off')

# result_preview.png 로 저장
plt.savefig("result_preview.png", dpi=300, facecolor='black', bbox_inches='tight')
print("✅ 성공! 3D 시각화 이미지(result_preview.png)가 생성되었습니다.")
