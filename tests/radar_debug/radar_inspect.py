"""
radar_inspect.py — посмотреть ЧТО ИМЕННО возвращает radar.poll().

Цель: понять структуру данных. radar.poll() возвращает numpy массив,
но какие у него колонки? Только (distance, doppler)? Или (distance, doppler,
x, y, z, intensity, ...)? Это решает можно ли фильтровать по высоте.

Скрипт:
  1. Спавнит ego на ровном месте
  2. Один раз опрашивает радар
  3. ПЕЧАТАЕТ всё что есть про возвращённый массив:
     - shape
     - dtype
     - первые 10 строк
     - min/max каждой колонки
     - имена колонок если доступны
"""

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Radar
import time
import numpy as np


BEAMNG_HOME = r'D:\Scripts\BeamNG.tech.v0.38.5.0'


def main():
    bng = BeamNGpy('localhost', 64256, home=BEAMNG_HOME)
    bng.open(launch=True)

    scenario = Scenario('automation_test_track', 'radar_inspect')
    vehicle = Vehicle('ego', model='etk800', license='RADAR-INS')
    scenario.add_vehicle(
        vehicle,
        pos=(155.255, -285.962, 120.839),
        rot_quat=(0, 0, 0.705, 0.709),
        cling=True,
    )
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    time.sleep(2)  # пусть всё загрузится

    radar = Radar(
        'radar', bng, vehicle,
        pos=(0, -2.3, 0.8),
        dir=(0, -1, 0.15),
        near_far_planes=(0.5, 120.0),
        is_visualised=True,
    )

    time.sleep(1)

    # === ВОТ ГЛАВНОЕ ===
    print("=" * 70)
    print("RADAR INSPECT")
    print("=" * 70)

    # Что вернёт poll?
    data = radar.poll()

    print(f"\ntype(data) = {type(data)}")
    print(f"repr(data)[:500] = {repr(data)[:500]}")

    if isinstance(data, dict):
        print(f"\nЭто словарь. Ключи: {list(data.keys())}")
        for k, v in data.items():
            print(f"\n  [{k}]: type={type(v).__name__}")
            if hasattr(v, 'shape'):
                print(f"        shape={v.shape}, dtype={v.dtype}")
            if hasattr(v, '__len__'):
                try:
                    print(f"        len={len(v)}")
                except Exception:
                    pass
    elif hasattr(data, 'shape'):
        print(f"\nЭто массив:")
        print(f"  shape = {data.shape}")
        print(f"  dtype = {data.dtype}")
        if data.dtype.names:
            print(f"  ИМЕНОВАННЫЕ ПОЛЯ: {data.dtype.names}")
        if data.ndim == 2:
            print(f"\n  Первые 5 строк:")
            for i, row in enumerate(data[:5]):
                print(f"    [{i}] {row}")
            print(f"\n  Статистика по колонкам:")
            for c in range(data.shape[1]):
                col = data[:, c]
                print(f"    col {c}: min={col.min():.3f}, max={col.max():.3f}, mean={col.mean():.3f}")
    else:
        print(f"\nНеизвестная структура. dir(data): {[x for x in dir(data) if not x.startswith('_')]}")

    print()
    print("=" * 70)

    # На всякий случай попробуем poll_raw или другие методы у Radar
    print(f"\nМетоды объекта Radar:")
    for m in dir(radar):
        if not m.startswith('_'):
            print(f"  {m}")

    input("\nPress ENTER to close BeamNG...")
    bng.close()


if __name__ == "__main__":
    main()
