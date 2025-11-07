#!/usr/bin/env python3
"""
Скачивание датасета InteriorGS с HuggingFace
"""

from huggingface_hub import snapshot_download
import os
import sys

# Скачиваем 10 сцен для обучения и тестирования
num_scenes = int(sys.argv[1]) if len(sys.argv) > 1 else 10
print(f"📥 Downloading {num_scenes} InteriorGS scenes...")

# Скачиваем в папку interiorgs_data
cache_dir = "/workspace/interiorgs_data"
os.makedirs(cache_dir, exist_ok=True)

# Датасет: spatialverse/InteriorGS
repo_id = "spatialverse/InteriorGS"

# ID первых 10 сцен (по номерам из датасета)
scene_ids = [
    "0001_839920",
    "0002_839955", 
    "0003_840015",
    "0004_840072",
    "0005_840112",
    "0006_840192",
    "0007_840237",
    "0008_840253",
    "0009_840332",
    "0010_840370"
][:num_scenes]

try:
    # Скачиваем выбранные сцены
    # Используем allow_patterns для скачивания только нужных папок
    patterns = [f"{scene_id}/*" for scene_id in scene_ids]
    
    print(f"Downloading scenes: {', '.join(scene_ids)}")
    
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=cache_dir,
        allow_patterns=patterns,
        max_workers=8
    )
    
    print(f"✅ Downloaded to: {local_dir}")
    
    # Проверяем содержимое каждой сцены
    total_size = 0
    for scene_id in scene_ids:
        scene_dir = os.path.join(local_dir, scene_id)
        if os.path.exists(scene_dir):
            files = os.listdir(scene_dir)
            scene_size = sum(os.path.getsize(os.path.join(scene_dir, f)) for f in files)
            total_size += scene_size
            print(f"\n📂 {scene_id}:")
            for f in sorted(files):
                size = os.path.getsize(os.path.join(scene_dir, f))
                print(f"   - {f}: {size / 1024 / 1024:.2f} MB")
        else:
            print(f"\n⚠️ {scene_id}: Not found")
    
    print(f"\n✅ Total downloaded: {total_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"✅ Ready for training!")
    
except Exception as e:
    print(f"❌ Error: {e}")
