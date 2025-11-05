import os
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
import shutil
from collections import OrderedDict
import struct

# --- Утилиты для записи в бинарном формате COLMAP ---

def write_points3D_binary(points3D, path_to_file):
    with open(path_to_file, "wb") as fid:
        num_points = len(points3D)
        fid.write(struct.pack('<Q', num_points))
        for point_id, point in points3D.items():
            xyz, rgb, error = point.xyz, point.rgb, point.error
            track_len = len(point.image_ids)
            fid.write(struct.pack('<Q', point_id))
            fid.write(struct.pack('<ddd', xyz[0], xyz[1], xyz[2]))
            fid.write(struct.pack('<BBB', rgb[0], rgb[1], rgb[2]))
            fid.write(struct.pack('<d', error))
            fid.write(struct.pack('<Q', track_len))
            for image_id in point.image_ids:
                fid.write(struct.pack('<I', image_id))
            for point2D_idx in point.point2D_idxs:
                fid.write(struct.pack('<I', point2D_idx))

def write_cameras_binary(cameras, path_to_file):
    with open(path_to_file, "wb") as fid:
        num_cameras = len(cameras)
        fid.write(struct.pack('<Q', num_cameras))
        for cam_id, cam in cameras.items():
            # Используем ID модели 'PINHOLE' (0), как ожидает RaDe-GS
            model_id = 0 
            fid.write(struct.pack('<I', cam.id))
            fid.write(struct.pack('<i', model_id))
            fid.write(struct.pack('<QQ', cam.width, cam.height))
            # Записываем 4 параметра как double (d)
            for p in cam.params:
                fid.write(struct.pack('<d', p))

def write_images_binary(images, path_to_file):
    with open(path_to_file, "wb") as fid:
        num_images = len(images)
        fid.write(struct.pack('<Q', num_images))
        for img_id, img in images.items():
            q, t = img.qvec, img.tvec
            fid.write(struct.pack('<I', img.id))
            fid.write(struct.pack('<dddd', q[0], q[1], q[2], q[3]))
            fid.write(struct.pack('<ddd', t[0], t[1], t[2]))
            fid.write(struct.pack('<I', img.camera_id))
            fid.write(struct.pack('<' + str(len(img.name)) + 's', img.name.encode("utf-8")))
            fid.write(struct.pack('<B', 0)) # Null terminator
            fid.write(struct.pack('<Q', len(img.xys)))

# --- Классы-структуры данных COLMAP ---

class Point3D:
    def __init__(self, id, xyz, rgb, error, image_ids, point2D_idxs):
        self.id, self.xyz, self.rgb, self.error, self.image_ids, self.point2D_idxs = id, xyz, rgb, error, image_ids, point2D_idxs

class Camera:
    def __init__(self, id, model, width, height, params):
        self.id, self.model, self.width, self.height, self.params = id, model, width, height, params

class ImageInfo:
    def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self.id, self.qvec, self.tvec, self.camera_id, self.name, self.xys, self.point3D_ids = id, qvec, tvec, camera_id, name, xys, point3D_ids

def rotmat2qvec(R):
    q = np.empty((4, ))
    t = np.trace(R)
    if t > 0:
        t = np.sqrt(t + 1)
        q[0] = 0.5 * t
        t = 0.5 / t
        q[1] = (R[2, 1] - R[1, 2]) * t
        q[2] = (R[0, 2] - R[2, 0]) * t
        q[3] = (R[1, 0] - R[0, 1]) * t
    else:
        i = 0
        if R[1, 1] > R[0, 0]: i = 1
        if R[2, 2] > R[i, i]: i = 2
        j = (i + 1) % 3
        k = (j + 1) % 3
        t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1)
        q[i + 1] = 0.5 * t
        t = 0.5 / t
        q[0] = (R[k, j] - R[j, k]) * t
        q[j + 1] = (R[i, j] + R[j, i]) * t
        q[k + 1] = (R[i, k] + R[k, i]) * t
    return q

# --- Основной скрипт ---

def convert_spatialgen_to_rade(spatialgen_dir, rade_datadir):
    print(f"Начало конвертации данных из: {spatialgen_dir}")
    print(f"Целевая директория: {rade_datadir}")

    input_dir = os.path.join(rade_datadir, "input")
    sparse_dir = os.path.join(rade_datadir, "sparse/0")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    print("Структура папок создана.")

    npz_path = os.path.join(spatialgen_dir, "inference_results.npz")
    ply_path = os.path.join(spatialgen_dir, "global_scene_ply.ply")
    
    with np.load(npz_path, allow_pickle=True) as data:
        key = data.files[0]
        results = data[key].item()

    print("Копирование изображений...")
    all_rgbs = results['input_rgbs'] + results['target_rgbs']
    for i, rgb_data in enumerate(all_rgbs):
        Image.fromarray(rgb_data.astype(np.uint8)).save(os.path.join(input_dir, f"{i}.png"))
    print(f"Скопировано {len(all_rgbs)} изображений.")

    print("Конвертация облака точек в points3D.bin...")
    pcd = o3d.io.read_point_cloud(ply_path)
    points, colors = np.asarray(pcd.points), (np.asarray(pcd.colors) * 255).astype(np.uint8)
    points3D = {i+1: Point3D(id=i+1, xyz=points[i], rgb=colors[i], error=0.0, image_ids=[], point2D_idxs=[]) for i in range(len(points))}
    write_points3D_binary(OrderedDict(sorted(points3D.items())), os.path.join(sparse_dir, "points3D.bin"))
    print("Файл points3D.bin успешно создан.")

    print("Создание cameras.bin...")
    K = results['intrinsic']
    h, w, _ = all_rgbs[0].shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    
    # ИСПРАВЛЕНО: Используем модель PINHOLE с 4 параметрами, как ожидает RaDe-GS
    cameras = {1: Camera(id=1, model='PINHOLE', width=w, height=h, params=np.array([fx, fy, cx, cy]))}
    write_cameras_binary(OrderedDict(sorted(cameras.items())), os.path.join(sparse_dir, "cameras.bin"))
    print("Файл cameras.bin успешно создан.")

    print("Создание images.bin...")
    all_poses = np.array(results['input_poses'] + results['target_poses'])
    images = {}
    for i, c2w in enumerate(all_poses):
        w2c = np.linalg.inv(c2w)
        R, T = w2c[:3, :3], w2c[:3, 3]
        q = rotmat2qvec(R)
        images[i+1] = ImageInfo(id=i+1, qvec=q, tvec=T, camera_id=1, name=f"{i}.png", xys=[], point3D_ids=[])
    write_images_binary(OrderedDict(sorted(images.items())), os.path.join(sparse_dir, "images.bin"))
    print("Файл images.bin успешно создан.")
    
    print("\nКонвертация успешно завершена!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Конвертер данных из SpatialGen в формат RaDe-GS.")
    parser.add_argument("--spatialgen_dir", type=str, required=True, help="Путь к выходной директории сцены SpatialGen.")
    parser.add_argument("--rade_datadir", type=str, required=True, help="Путь к целевой директории данных для RaDe-GS.")
    args = parser.parse_args()
    convert_spatialgen_to_rade(args.spatialgen_dir, args.rade_datadir)
