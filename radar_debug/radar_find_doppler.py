"""
radar_find_doppler.py — какая колонка в radar.poll() это доплер?

Сейчас в adas_v0_5.py мы читаем col 1 как доплер, но из radar_verify видно
что col 1 = стабильно 1.2..1.9, что НЕ похоже на доплер. Доплер должен
зависеть от скорости машины.

Этот скрипт:
  1. Спавнит ego.
  2. Включает движение вперёд (нарастающий газ).
  3. Каждую секунду печатает скорость и среднее значение КАЖДОЙ колонки
     радара. Колонка которая ЛИНЕЙНО растёт со скоростью — доплер.
"""

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Radar, Electrics, State
import time
import numpy as np


BEAMNG_HOME = r'D:\Scripts\BeamNG.tech.v0.38.5.0'


def main():
    bng = BeamNGpy('localhost', 64256, home=BEAMNG_HOME)
    bng.open(launch=True)

    scenario = Scenario('automation_test_track', 'radar_doppler')
    vehicle = Vehicle('ego', model='etk800', license='RADAR-DOP')
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

    electrics = Electrics()
    vehicle.attach_sensor('electrics', electrics)

    radar = Radar(
        'radar', bng, vehicle,
        pos=(0, -2.3, 0.8),
        dir=(0, -1, 0.15),
        near_far_planes=(0.5, 120.0),
        is_visualised=True,
    )

    time.sleep(1)
    print("=" * 80)
    print("Поехали — будем плавно разгоняться. Смотрим как меняются колонки.")
    print("=" * 80)
    print(f"{'t':>5} {'speed':>6} | {'col0':>8} {'col1':>8} {'col2':>8} {'col3':>8} {'col4':>10} {'col5':>12} {'col6':>8}")

    # Фильтр чтобы убрать шум — берём только горизонтальные сильные точки
    # (как мы выяснили, без фильтра данные забивает асфальт/небо)
    def filter_strong_horizontal(data):
        if data is None or len(data) == 0:
            return data
        elev = data[:, 3]
        intens = data[:, 6]
        mask = (np.abs(elev) < 0.087) & (intens > 0.3)
        return data[mask]

    vehicle.control(throttle=0.3, brake=0, steering=0)
    start = time.time()

    try:
        for tick in range(20):  # 20 секунд разгона
            time.sleep(1.0)
            vehicle.poll_sensors()
            t = time.time() - start
            speed = electrics.data.get('wheelspeed', 0.0)

            data = radar.poll()
            filtered = filter_strong_horizontal(data)
            if filtered is None or len(filtered) == 0:
                print(f"{t:5.1f} {speed*3.6:>6.1f} | (нет точек после фильтра)")
                continue

            means = [filtered[:, c].mean() for c in range(filtered.shape[1])]
            print(f"{t:5.1f} {speed*3.6:>6.1f} | "
                  f"{means[0]:>8.2f} {means[1]:>8.3f} {means[2]:>8.3f} "
                  f"{means[3]:>8.3f} {means[4]:>10.1f} {means[5]:>12.2e} {means[6]:>8.3f}")

    except KeyboardInterrupt:
        pass
    finally:
        vehicle.control(throttle=0, brake=1.0)
        time.sleep(0.5)

        print()
        print("=" * 80)
        print("ВЫВОД: какая колонка линейно растёт со скоростью — та и доплер.")
        print("=" * 80)
        input("Press ENTER to close...")
        bng.close()


if __name__ == "__main__":
    main()
