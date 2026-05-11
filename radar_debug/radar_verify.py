"""
radar_verify.py — проверка гипотезы что col 3 это elevation (угол вверх/вниз).

Если гипотеза верна — после фильтрации по |col 3| < 0.1 рад (±5°):
  - количество точек упадёт в ~10 раз
  - ближайшая точка перестанет быть «асфальтом на 1.7м»
  - и станет реальным препятствием на 5+ метров

Дополнительно проверяем col 6 (норм. intensity) — отбраковка слабых сигналов.
"""

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Radar
import time
import numpy as np
import math


BEAMNG_HOME = r'D:\Scripts\BeamNG.tech.v0.38.5.0'


def analyze(data, label):
    print(f"\n── {label} ──")
    print(f"  точек: {len(data)}")
    if len(data) == 0:
        return
    d = data[:, 0]
    print(f"  distance: min={d.min():.2f}, mean={d.mean():.2f}, max={d.max():.2f}")
    # Гистограмма по дистанции
    bins = [(0, 2), (2, 5), (5, 10), (10, 30), (30, 60), (60, 120), (120, 200)]
    for lo, hi in bins:
        cnt = ((d >= lo) & (d < hi)).sum()
        if cnt:
            pct = 100 * cnt / len(data)
            print(f"    {lo:>3}..{hi:>3}м: {cnt:5d} ({pct:5.1f}%)")
    # Ближайшая точка
    idx_close = d.argmin()
    p = data[idx_close]
    print(f"  БЛИЖАЙШАЯ: dist={p[0]:.2f}, col1={p[1]:.3f}, "
          f"azim={p[2]:.3f} рад ({math.degrees(p[2]):+.1f}°), "
          f"elev={p[3]:.3f} рад ({math.degrees(p[3]):+.1f}°)")


def main():
    bng = BeamNGpy('localhost', 64256, home=BEAMNG_HOME)
    bng.open(launch=True)

    scenario = Scenario('automation_test_track', 'radar_verify')
    vehicle = Vehicle('ego', model='etk800', license='RADAR-VER')
    scenario.add_vehicle(
        vehicle,
        pos=(155.255, -285.962, 120.839),
        rot_quat=(0, 0, 0.705, 0.709),
        cling=True,
    )
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    time.sleep(2)

    radar = Radar(
        'radar', bng, vehicle,
        pos=(0, -2.3, 0.8),
        dir=(0, -1, 0.15),
        near_far_planes=(0.5, 120.0),
        is_visualised=True,
    )

    time.sleep(1)

    data = radar.poll()
    print("=" * 70)
    print(f"Сырых точек: {len(data)}")

    # БЕЗ фильтра — что видим
    analyze(data, "БЕЗ ФИЛЬТРА")

    # Фильтр elevation ±5° (~0.087 рад)
    elev = data[:, 3]
    mask_h = np.abs(elev) < 0.087
    horizontal = data[mask_h]
    analyze(horizontal, "ТОЛЬКО ГОРИЗОНТАЛЬНЫЕ ЛУЧИ (|elev| < 5°)")

    # Ещё уже — ±2°
    mask_h2 = np.abs(elev) < 0.035
    narrow = data[mask_h2]
    analyze(narrow, "УЗКАЯ ПОЛОСА (|elev| < 2°)")

    # Только лучи которые смотрят строго вперёд (azim ±5° тоже)
    azim = data[:, 2]
    mask_fwd = (np.abs(elev) < 0.087) & (np.abs(azim) < 0.087)
    forward = data[mask_fwd]
    analyze(forward, "ПЕРЕДНИЙ КОНУС (|elev|<5° И |azim|<5°)")

    # Дополнительно — фильтр по col 6 (предполагаемая intensity)
    if data.shape[1] > 6:
        intens = data[:, 6]
        mask_strong = intens > 0.3
        strong = data[mask_strong]
        analyze(strong, "ТОЛЬКО СИЛЬНЫЕ СИГНАЛЫ (col6 > 0.3)")

        # И обе фильтрации
        mask_combo = mask_h & (intens > 0.3)
        combo = data[mask_combo]
        analyze(combo, "ГОРИЗОНТ + СИЛЬНЫЕ")

    print("\n" + "=" * 70)
    print("Анализ:")
    print(f"  Если БЕЗ фильтра ближайшая ~1.7м с elev сильно ниже 0 → это асфальт")
    print(f"  Если С ГОРИЗОНТАЛЬНЫМ фильтром ближайшая 5+ метров → гипотеза верна")
    print(f"  Если ближайшая стала ~150м (горизонт/небо) → радар реально ничего не")
    print(f"  видит впереди (что и должно быть на пустой трассе).")

    input("\nPress ENTER to close BeamNG...")
    bng.close()


if __name__ == "__main__":
    main()
