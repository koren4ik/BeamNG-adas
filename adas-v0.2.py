from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, Radar, Ultrasonic, State, Lidar
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
import math

bng = BeamNGpy('localhost', 64256, home=r'D:\Scripts\BeamNG.tech.v0.38.5.0')
bng.open(launch=True)

scenario = Scenario('tech_ground', 'adas_test')

vehicle = Vehicle('ego', model='etk800', license='ADAS-MAIN')
leader = Vehicle('leader', model='etk800', license='LEAD')

scenario.add_vehicle(vehicle, pos=(0, 0, 0), cling=True)
scenario.add_vehicle(leader, pos=(0, -40, 0), cling=True)

scenario.make(bng)
bng.scenario.load(scenario)
bng.scenario.start()

time.sleep(1)

LEADER_BASE_SPEED = 130 / 3.6
LEADER_AMPLITUDE = 20 / 3.6
LEADER_PERIOD = 15.0
start_time = time.time()

leader_electrics = Electrics()
leader.attach_sensor('leader_electrics', leader_electrics)

electrics = Electrics()
vehicle.attach_sensor('electrics', electrics)

# Сенсор позиции для обеих машин
pos_sensor = State()
vehicle.attach_sensor('pos_sensor', pos_sensor)

leader_pos_sensor = State()
leader.attach_sensor('leader_pos_sensor', leader_pos_sensor)

radar = Radar(
    'radar', bng, vehicle,
    pos=(0, -2.35, 0.5),
    dir=(0, -1, 0),
    near_far_planes=(0.1, 100.0),
    is_visualised=True
)

us_front = Ultrasonic(
    'us_front', bng, vehicle,
    pos=(0, -2.35, 0.5),
    dir=(0, -1, 0),
    near_far_planes=(0.1, 6.0),
    is_visualised=True
)

us_rear = Ultrasonic(
    'us_rear', bng, vehicle,
    pos=(0, 2.35, 0.5),
    dir=(0, 1, 0),
    near_far_planes=(0.1, 6.0),
    is_visualised=True
)

lidar = Lidar(
    'lidar', bng, vehicle,
    pos=(0, 0, 1.8),
    dir=(0, -1, 0),
    max_distance=50.0,
    is_360_mode=True,
    vertical_resolution=32,
    is_visualised=False  # визуализация через matplotlib, не в игре
)

SAFE_DISTANCE = 40
US_STOP_DIST = 1.5
US_CREEP_i_error = 0.0
US_CREEP_SPEED = 5 / 3.6

print("ADAS v0.2 активна")
aeb_triggered = False
leader_stopped = False

prev_leader_x = 0.0
prev_ego_x = 0.0

# Общие данные между потоками
lidar_shared = {
    'points': None,
    'ego_x': 0.0,
    'ego_y': 0.0,
    'lock': threading.Lock()
}

# --- Поток визуализации LiDAR ---
def lidar_visualization():
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('black')

    while True:
        time.sleep(0.2)

        with lidar_shared['lock']:
            points = lidar_shared['points']
            ego_x = lidar_shared['ego_x']
            ego_y = lidar_shared['ego_y']

        if points is None:
            continue

        pts = points.copy()

        # Переводим в систему координат машины
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)

        if pts.shape[0] == 0 or pts.shape[1] != 3:
            continue

        pts[:, 0] -= ego_x
        pts[:, 1] -= ego_y

        # Разделяем землю и препятствия
        ground = pts[pts[:, 2] <= 0.3]
        obstacles = pts[pts[:, 2] > 0.3]

        ax.clear()
        ax.set_facecolor('black')

        if len(ground) > 0:
            ax.scatter(ground[:, 0], ground[:, 1], s=0.3, c='gray', alpha=0.2)

        if len(obstacles) > 0:
            ax.scatter(obstacles[:, 0], obstacles[:, 1], s=2, c='red')

            cx = obstacles[:, 0].mean()
            cy = obstacles[:, 1].mean()
            dist = (cx**2 + cy**2) ** 0.5

            ax.scatter(cx, cy, s=100, c='yellow', marker='x', linewidths=2)
            ax.annotate(f'{dist:.1f}м', (cx + 1, cy + 1), color='yellow', fontsize=10)

        # Наша машина
        ax.scatter(0, 0, s=120, c='lime', marker='^')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_title(f'LiDAR | точек: {len(pts)} | препятствий: {len(obstacles)}',
                     color='white')
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y (вперёд)', color='white')
        ax.tick_params(colors='white')

        plt.pause(0.01)

# Запускаем поток визуализации
viz_thread = threading.Thread(target=lidar_visualization, daemon=True)
viz_thread.start()

# --- Основной цикл управления ---
while True:
    time.sleep(0.05)
    vehicle.poll_sensors()

    # Лидер
    elapsed = time.time() - start_time

    if elapsed > 25:
        leader.poll_sensors()
        leader_speed = leader_electrics.data.get('wheelspeed', 0)
        leader.control(throttle=0, brake=1.0)
        if leader_speed < 0.5:
            leader.control(throttle=0, brake=0, parkingbrake=1.0)
            leader_stopped = True
    else:
        leader_target = LEADER_BASE_SPEED + LEADER_AMPLITUDE * math.sin(2 * math.pi * elapsed / LEADER_PERIOD)
        leader.poll_sensors()
        leader_speed = leader_electrics.data.get('wheelspeed', 0)
        leader_error = leader_target - leader_speed

        # Удержание лидера по X=0
        leader_x = leader_pos_sensor.data.get('pos', (0, 0, 0))[0]
        dlx = leader_x - prev_leader_x
        prev_leader_x = leader_x

        leader_steer = max(-1.0, min(1.0, (leader_x * 0.05 + dlx * 0.5)))

        leader.control(
            throttle=max(0, min(1, leader_error * 0.1)),
            brake=max(0, min(1, -leader_error * 0.1)),
            steering=leader_steer
        )

    speed = electrics.data.get('wheelspeed', 0)

    # Удержание нашей машины по X=0 — вычисляем ДО использования
    ego_pos = pos_sensor.data.get('pos', (0, 0, 0))
    ego_x = ego_pos[0]
    ego_y = ego_pos[1]

    dx = ego_x - prev_ego_x  # скорость отклонения
    prev_ego_x = ego_x

    # Смягченные коэффициенты (P=0.05, D=0.5)
    steering = max(-1.0, min(1.0, (ego_x * 0.05 + dx * 0.5)))

    # Обновляем LiDAR данные для потока визуализации
    lidar_data = lidar.poll()
    if lidar_data is not None and 'pointCloud' in lidar_data:
        with lidar_shared['lock']:
            lidar_shared['points'] = lidar_data['pointCloud']
            lidar_shared['ego_x'] = ego_x
            lidar_shared['ego_y'] = ego_y

    # Радар читаем всегда
    data = radar.poll()
    distance = 9999
    doppler = 0.0
    if data is not None and data.size > 0:
        closest = data[data[:, 0].argmin()]
        distance = float(closest[0])
        doppler = float(closest[1])

    # Ультразвук читаем всегда
    front_dist = us_front.poll().get('distance', 9999)
    rear_dist = us_rear.poll().get('distance', 9999)
    us_min_dist = min(front_dist, rear_dist)

    use_ultrasonic = us_min_dist < 5.0

    if use_ultrasonic:
        print(f"[УЗ] Скорость: {speed*3.6:.1f} км/ч | Спереди: {front_dist:.1f} м | Сзади: {rear_dist:.1f} м")

        if front_dist < US_STOP_DIST or rear_dist < US_STOP_DIST:
            print("УЗ: препятствие рядом — стоп!")
            brake_intensity = max(0.3, min(1.0, (US_STOP_DIST - front_dist) * 2.0))
            vehicle.control(throttle=0, brake=brake_intensity, steering=steering)

            if speed < 0.5:
                vehicle.control(throttle=0, brake=0, parkingbrake=1.0)
                print("Машина остановлена ультразвуком!")
                break
        else:
            creep_error = US_CREEP_SPEED - speed
            US_CREEP_i_error += creep_error * 0.05

            # Ограничиваем накопленную ошибку (Anti-windup), чтобы машина не "взлетала"
            US_CREEP_i_error = max(-1.0, min(1.0, US_CREEP_i_error))

            # Kp = 0.25, Ki = 0.1 (плавное дотягивание)
            throttle_val = (creep_error * 0.25) + (US_CREEP_i_error * 0.1)

            throttle = max(0, min(0.25, throttle_val))  # Ограничим макс. газ для плавности
            brake = max(0, min(0.3, -throttle_val))     # Ограничим резкость торможения

            vehicle.control(throttle=throttle, brake=brake, parkingbrake=0, steering=steering)

    else:
        if not aeb_triggered:
            # Если лидер стоит — подъезжаем вплотную, иначе держим дистанцию
            target_dist = 3.0 if leader_stopped else SAFE_DISTANCE

            dist_error = distance - target_dist
            combined = dist_error * 0.1 - doppler * 0.5
            throttle = max(0, min(1, combined))
            brake = max(0, min(1, -combined))
            vehicle.control(throttle=throttle, brake=brake, parkingbrake=0, steering=steering)

            print(f"[ACC] Скорость: {speed*3.6:.1f} км/ч | Дистанция: {distance:.1f} м | Доплеровская скорость: {doppler*3.6:.1f} км/ч | X: {ego_x:.2f}")

input("Нажми Enter чтобы закрыть BeamNG...")
bng.close()
