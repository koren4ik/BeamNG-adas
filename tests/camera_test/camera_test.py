"""
camera_test.py — итерация 0 для lane keeping.

Запускает BeamNG, ставит ego на сцену, делает несколько кадров с фронтальной
камеры в /test_frames/. Это нужно, чтобы:
  1. Убедиться, что камера вообще работает (формат, ориентация, цвета).
  2. Получить реальные кадры для калибровки perspective transform offline.

После того как кадры сохранятся — прислать их Claude вместе с любыми ошибками
или странностями. На основе кадров будем настраивать ROI и src-точки.

Запускать ОТДЕЛЬНО от adas_v0_5.py — чтобы не смешивать тестирование сенсора
с основной логикой.
"""

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
import cv2
import numpy as np
import os
import time


# ─────────────────────────────────────────────────────────
# Параметры
# ─────────────────────────────────────────────────────────
BEAMNG_HOME = r'D:\Scripts\BeamNG.tech.v0.38.5.0'
OUTPUT_DIR = 'test_frames'

# Резолюция камеры. Не задирай — pipeline будет тормозить.
# Для отладки 640x360 норм. В проде можно 320x180.
CAM_W, CAM_H = 640, 360

# Сколько кадров снять и с каким интервалом
N_FRAMES = 6
INTERVAL_S = 2.5  # машина успеет проехать ~70м на 100 км/ч между кадрами


def save_frame(cam_data, path: str):
    """
    Извлекает RGB-кадр из ответа Camera.poll() и сохраняет на диск.
    """
    img = None

    # Попытка 1: словарь с ключом 'colour' или 'color'
    if isinstance(cam_data, dict):
        for key in ('colour', 'color'):
            if key in cam_data:
                img = cam_data[key]
                break

    if img is None:
        img = cam_data

    # Если PIL.Image — в numpy
    if hasattr(img, 'mode'):  # PIL Image
        img = np.array(img)

    # Если есть alpha-канал — отбросить
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # PIL/BeamNG обычно возвращают RGB; OpenCV хранит BGR.
    # Конвертируем перед записью.
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, img_bgr)
    print(f"  сохранён: {path}  shape={img.shape}  dtype={img.dtype}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bng = BeamNGpy('localhost', 64256, home=BEAMNG_HOME)
    bng.open(launch=True)

    scenario = Scenario('automation_test_track', 'camera_test')
    vehicle = Vehicle('ego', model='etk800', license='CAM-TEST')
    scenario.add_vehicle(vehicle, pos=(155.255, -285.962, 120.839), rot_quat=(0, 0, 0.705, 0.709), cling=True)
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    time.sleep(1)

    # ─── Камера ───
    # Аналогично Radar и Ultrasonic, должно быть что-то типа:
    #
    #   camera = Camera('cam', bng, vehicle, pos=..., dir=..., resolution=...,
    #                   is_render_colours=True, ...)
    #
    # Параметры, которые точно нужны:
    #   pos       — позиция относительно центра машины. Радар у нас стоит
    #               (0, -2.3, 0.8). Камера будет на крыше: (0, -1.5, 1.3).
    #               Это в системе BeamNG: -Y = вперёд, +Z = вверх.
    #   dir       — направление взгляда. Туда же, куда едем: (0, -1, 0) +
    #               чуть вниз для лучшего обзора дороги: (0, -1, -0.1).
    #   resolution — (W, H)
    # Параметры, которые могут отличаться по имени между версиями:
    #   is_render_colours / is_render_color / render_color — нужно True
    #   is_render_depth / render_depth — НЕ нужно (False)
    #   is_render_annotations — НЕ нужно (False)
    #
    camera = Camera(
        'cam_front', bng, vehicle,
        pos=(0, -1.5, 1.3),
        dir=(0, -1, -0.1),
        resolution=(CAM_W, CAM_H),
        is_render_colours=True,
        is_render_depth=False,
        is_render_annotations=False,
    )

    print(f"Камера создана: {CAM_W}x{CAM_H}")
    print(f"Снимаем {N_FRAMES} кадров с интервалом {INTERVAL_S}с\n")

    # Едем вперёд на нормальной скорости — нам нужны кадры в движении
    vehicle.control(throttle=0.5, brake=0, steering=0)

    try:
        for i in range(N_FRAMES):
            time.sleep(INTERVAL_S)

            # На первом кадре печатаем что вернул сенсор — для отладки
            cam_data = camera.poll()
            if i == 0:
                print(f"DEBUG: type(cam_data) = {type(cam_data)}")
                if isinstance(cam_data, dict):
                    print(f"DEBUG: keys = {list(cam_data.keys())}")
                    for k, v in cam_data.items():
                        print(f"DEBUG:   {k}: {type(v)}")
                print()

            path = os.path.join(OUTPUT_DIR, f'frame_{i:02d}.png')
            save_frame(cam_data, path)

    finally:
        vehicle.control(throttle=0, brake=1.0)
        time.sleep(0.5)
        print(f"\nГотово. Кадры в {OUTPUT_DIR}/")
        input("Press ENTER to close BeamNG...")
        bng.close()


if __name__ == "__main__":
    main()