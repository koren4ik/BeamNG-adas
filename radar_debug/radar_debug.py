"""
radar_debug.py — диагностика радар-сенсора.

Запускает машину на сцене, НЕ управляет ею (ты рулишь сам через клавиатуру),
и логирует ВСЁ что видит радар на каждом тике:
  - все точки которые радар вернул (не только ближайшую)
  - их distance, doppler, скорость машины

Цель — понять что за «фантомы» появляются на больших картах, какие у них
характеристики, и как их отличить от реальных машин впереди.

После запуска:
  1. Откроется BeamNG, машина заспавнится на твоей карте.
  2. Можно либо рулить руками (стрелочки или WASD), либо просто стоять.
  3. Поезди ~30 секунд по местам где обычно бывают фантомы.
  4. Закрой окно или нажми Ctrl+C — лог сохранится в radar_debug_log.csv.

После этого пришли CSV на анализ.
"""

from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, Radar, State
import csv
import time


# ── НАСТРОЙКИ ──
BEAMNG_HOME = r'D:\Scripts\BeamNG.tech.v0.38.5.0'
MAP_NAME    = 'automation_test_track'      # та же что в adas_v0_5.py
SCENARIO    = 'radar_debug'

# Координаты спавна — те же что у тебя в основном проекте
SPAWN_POS      = (155.255, -285.962, 120.839)
SPAWN_ROT_QUAT = (0, 0, 0.705, 0.709)

# Параметры радара — те же что в adas_v0_5.py чтобы данные были репрезентативные.
# ВАЖНО: НЕ ставь dir=(0,0,1) — нам нужен радар который смотрит ВПЕРЁД,
# чтобы поймать фантомы которые мы хотим изучить.
RADAR_POS  = (0, -2.3, 0.8)
RADAR_DIR  = (0, -1, 0.15)
RADAR_NEAR = 0.5
RADAR_FAR  = 120.0

LOG_PATH = 'radar_debug_log.csv'
LOOP_DT  = 0.05   # 20 Гц — для диагностики хватит, меньше нагрузка на диск


def main():
    bng = BeamNGpy('localhost', 64256, home=BEAMNG_HOME)
    bng.open(launch=True)

    scenario = Scenario(MAP_NAME, SCENARIO)
    vehicle = Vehicle('ego', model='etk800', license='RADAR-DBG')
    scenario.add_vehicle(
        vehicle,
        pos=SPAWN_POS,
        rot_quat=SPAWN_ROT_QUAT,
        cling=True,
    )
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    time.sleep(1)

    # — Сенсоры —
    electrics = Electrics()
    pos_sensor = State()
    vehicle.attach_sensor('electrics', electrics)
    vehicle.attach_sensor('pos_sensor', pos_sensor)

    radar = Radar(
        'radar', bng, vehicle,
        pos=RADAR_POS,
        dir=RADAR_DIR,
        near_far_planes=(RADAR_NEAR, RADAR_FAR),
        is_visualised=True,
    )

    # — Передаём управление человеку —
    # vehicle.control() с никакими аргументами не вызываем — машина сама
    # под клавиатурой/геймпадом BeamNG.

    # — Лог —
    f = open(LOG_PATH, 'w', newline='', encoding='utf-8')
    writer = csv.writer(f)
    # На каждый ТИК пишем НЕСКОЛЬКО строк (одна на точку радара).
    # Это позволяет видеть полную картину "что видел радар", а не только
    # ближайшую точку.
    writer.writerow([
        't', 'ego_speed_kmh', 'ego_x', 'ego_y', 'ego_z',
        'n_points',         # сколько точек радар вернул на этом тике
        'point_idx',        # индекс точки в массиве (0 = ближайшая)
        'distance',         # дистанция до точки, м
        'doppler',          # доплер, м/с (положительный = сближение)
        'doppler_rel_ego',  # доплер / ego_speed — соотношение
    ])

    print(f"[RADAR-DEBUG] Лог: {LOG_PATH}")
    print(f"[RADAR-DEBUG] Карта: {MAP_NAME}, спавн: {SPAWN_POS}")
    print(f"[RADAR-DEBUG] Радар: pos={RADAR_POS}, dir={RADAR_DIR}, "
          f"range={RADAR_NEAR}..{RADAR_FAR}")
    print()
    print("Управляй машиной как обычно (BeamNG keyboard/gamepad).")
    print("Нажми Ctrl+C в этом окне когда наездишься. Лог сохранится.")
    print()

    start_time = time.time()
    tick = 0

    try:
        while True:
            time.sleep(LOOP_DT)
            tick += 1

            vehicle.poll_sensors()
            t = time.time() - start_time
            speed = electrics.data.get('wheelspeed', 0.0)
            pos = pos_sensor.data.get('pos', (0, 0, 0))

            # Опрашиваем радар — получаем массив всех точек
            data = radar.poll()

            if data is None or data.size == 0:
                # Радар ничего не увидел — пишем одну строку-заглушку
                writer.writerow([
                    f"{t:.3f}", f"{speed*3.6:.2f}",
                    f"{pos[0]:.2f}", f"{pos[1]:.2f}", f"{pos[2]:.2f}",
                    0, -1, '', '', '',
                ])
            else:
                # Сортируем по distance чтобы point_idx=0 была ближайшая
                # (как и используется в adas_v0_5.py)
                sorted_idx = data[:, 0].argsort()
                sorted_data = data[sorted_idx]
                n = len(sorted_data)

                # Логируем ВСЕ точки этого тика
                for i, point in enumerate(sorted_data):
                    dist = float(point[0])
                    dop = float(point[1])
                    rel = dop / speed if speed > 0.1 else 0.0
                    writer.writerow([
                        f"{t:.3f}", f"{speed*3.6:.2f}",
                        f"{pos[0]:.2f}", f"{pos[1]:.2f}", f"{pos[2]:.2f}",
                        n, i, f"{dist:.2f}", f"{dop:.3f}", f"{rel:.3f}",
                    ])

            # Раз в секунду — печатаем сводку в консоль чтобы было видно прогресс
            if tick % 20 == 0:
                if data is None or data.size == 0:
                    print(f"t={t:5.1f}с  speed={speed*3.6:5.1f}  радар: 0 точек")
                else:
                    closest = sorted_data[0]
                    print(f"t={t:5.1f}с  speed={speed*3.6:5.1f}  "
                          f"радар: {len(sorted_data)} точек, "
                          f"ближайшая d={closest[0]:5.1f}м  dop={closest[1]:+.2f}")

            # Sync write на диск — чтобы при Ctrl+C ничего не потерялось
            if tick % 20 == 0:
                f.flush()

    except KeyboardInterrupt:
        print("\n[RADAR-DEBUG] Ctrl+C — завершаем")
    finally:
        f.close()
        print(f"[RADAR-DEBUG] Лог сохранён: {LOG_PATH}")
        input("Press ENTER to close BeamNG...")
        bng.close()


if __name__ == "__main__":
    main()
