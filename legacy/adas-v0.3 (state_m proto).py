from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, Radar, Ultrasonic, State, Lidar
from collections import deque
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
import math

# ─────────────────────────────────────────────
#  STATE MACHINE
# ─────────────────────────────────────────────
class ADASState(Enum):
    CRUISE = "CRUISE"   # свободная дорога — едем на целевой скорости
    FOLLOW = "FOLLOW"   # лидер обнаружен — держим time headway
    AEB    = "AEB"      # TTC критический — экстренное торможение
    CREEP  = "CREEP"    # УЗ режим — ползём до препятствия
    STOP   = "STOP"     # полная остановка

# ─────────────────────────────────────────────
#  КОНСТАНТЫ
# ─────────────────────────────────────────────
LEADER_BASE_SPEED  = 130 / 3.6
LEADER_AMPLITUDE   = 20 / 3.6
LEADER_PERIOD      = 15.0
LEADER_STOP_AT     = 25.0          # секунд до остановки лидера

TARGET_SPEED       = 140 / 3.6    # крейсерская скорость
TIME_HEADWAY       = 2.0           # секунды — дистанция = speed * headway
MIN_SAFE_DIST      = 15.0          # минимальная дистанция даже на малой скорости

TTC_AEB_THRESHOLD  = 3.0           # TTC ниже — AEB
TTC_AEB_RELEASE    = 5.0           # TTC выше — выход из AEB
RADAR_DETECT_DIST  = 90.0          # дистанция обнаружения лидера (CRUISE→FOLLOW)
RADAR_LOST_DIST    = 120.0         # дистанция потери лидера (FOLLOW→CRUISE)

US_STOP_DIST       = 1.5           # УЗ стоп если ближе
US_CREEP_SPEED     = 5 / 3.6      # скорость ползания в CREEP
CREEP_RADAR_DIST   = 15.0         # радарный порог входа в CREEP когда лидер стоит
CREEP_RADAR_STOP   = 2.0          # радарная остановка в CREEP (было 5.0 → стоп слишком рано)
CREEP_BRAKE_SPEED  = 20 / 3.6    # если в CREEP скорость выше — сначала тормозим

# PD-контроллер ACC
ACC_KP             = 0.08
ACC_KD             = 0.4
ACC_FF_DOPPLER     = 0.05

# PD-контроллер руля
STEER_KP           = 0.05
STEER_KD           = 0.5

# Фильтры — размер окна медианного фильтра
RADAR_FILTER_SIZE  = 5
US_FILTER_SIZE     = 5

AEB_CONFIRM_TICKS  = 3

# ─────────────────────────────────────────────
#  МЕДИАННЫЙ ФИЛЬТР
# ─────────────────────────────────────────────
class MedianFilter:
    def __init__(self, size):
        self.buf = deque(maxlen=size)

    def update(self, value):
        self.buf.append(value)
        return float(np.median(self.buf))

# ─────────────────────────────────────────────
#  BEAMNG SETUP
# ─────────────────────────────────────────────
bng = BeamNGpy('localhost', 64256, home=r'D:\Scripts\BeamNG.tech.v0.38.5.0')
bng.open(launch=True)

scenario = Scenario('tech_ground', 'adas_v06')

vehicle = Vehicle('ego',    model='etk800', license='ADAS-V06')
leader  = Vehicle('leader', model='etk800', license='LEAD')

scenario.add_vehicle(vehicle, pos=(0,   0, 0), cling=True)
scenario.add_vehicle(leader,  pos=(0, -40, 0), cling=True)

scenario.make(bng)
bng.scenario.load(scenario)
bng.scenario.start()

time.sleep(1)

# ─────────────────────────────────────────────
#  СЕНСОРЫ
# ─────────────────────────────────────────────
electrics        = Electrics()
leader_electrics = Electrics()
vehicle.attach_sensor('electrics',        electrics)
leader.attach_sensor('leader_electrics',  leader_electrics)

pos_sensor        = State()
leader_pos_sensor = State()
vehicle.attach_sensor('pos_sensor',        pos_sensor)
leader.attach_sensor('leader_pos_sensor',  leader_pos_sensor)

radar = Radar(
    'radar', bng, vehicle,
    pos=(0, -2.35, 0.5), dir=(0, -1, 0),
    near_far_planes=(0.1, 120.0),
    is_visualised=True
)

us_front = Ultrasonic(
    'us_front', bng, vehicle,
    pos=(0, -2.35, 0.5), dir=(0, -1, 0),
    near_far_planes=(0.1, 6.0),
    is_visualised=True
)

us_rear = Ultrasonic(
    'us_rear', bng, vehicle,
    pos=(0, 2.35, 0.5), dir=(0, 1, 0),
    near_far_planes=(0.1, 6.0),
    is_visualised=True
)

lidar = Lidar(
    'lidar', bng, vehicle,
    pos=(0, 0, 1.8), dir=(0, -1, 0),
    max_distance=50.0,
    is_360_mode=True,
    vertical_resolution=32,
    is_visualised=False
)

# ─────────────────────────────────────────────
#  ФИЛЬТРЫ СЕНСОРОВ
# ─────────────────────────────────────────────
radar_dist_filter = MedianFilter(RADAR_FILTER_SIZE)
radar_doppler_filter = MedianFilter(RADAR_FILTER_SIZE)
us_front_filter = MedianFilter(US_FILTER_SIZE)
us_rear_filter = MedianFilter(US_FILTER_SIZE)

# ─────────────────────────────────────────────
#  LIDAR ВИЗУАЛИЗАЦИЯ (отдельный поток)
# ─────────────────────────────────────────────
lidar_shared = {'points': None, 'ego_x': 0.0, 'ego_y': 0.0, 'lock': threading.Lock()}

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
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        if pts.shape[0] == 0 or pts.shape[1] != 3:
            continue

        pts[:, 0] -= ego_x
        pts[:, 1] -= ego_y

        ground = pts[pts[:, 2] <= 0.3]
        obstacles = pts[pts[:, 2]  > 0.3]

        ax.clear()
        ax.set_facecolor('black')

        if len(ground) > 0:
            ax.scatter(ground[:, 0], ground[:, 1], s=0.3, c='gray', alpha=0.2)

        if len(obstacles) > 0:
            ax.scatter(obstacles[:, 0], obstacles[:, 1], s=2, c='red')
            cx   = obstacles[:, 0].mean()
            cy   = obstacles[:, 1].mean()
            dist = math.sqrt(cx**2 + cy**2)
            ax.scatter(cx, cy, s=100, c='yellow', marker='x', linewidths=2)
            ax.annotate(f'{dist:.1f}м', (cx + 1, cy + 1), color='yellow', fontsize=10)

        ax.scatter(0, 0, s=120, c='lime', marker='^')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_title(f'LiDAR | точек: {len(pts)} | препятствий: {len(obstacles)}', color='white')
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y (вперёд)', color='white')
        ax.tick_params(colors='white')
        plt.pause(0.01)

viz_thread = threading.Thread(target=lidar_visualization, daemon=True)

# ─────────────────────────────────────────────
#  ПЕРЕМЕННЫЕ СОСТОЯНИЯ
# ─────────────────────────────────────────────

state          = ADASState.CRUISE
start_time     = time.time()
leader_stopped = False

prev_ego_x = 0.0
prev_leader_x = 0.0
prev_dist_err = 0.0    # для PD ACC
US_CREEP_i_error = 0.0
aeb_confirm_count = 0

print(" ADAS | BeamNG.tech")
print(f" Start STATE: {state.value}")
print(f" Target speed: {TARGET_SPEED*3.6:.0f} км/ч")
print(f" Time headway: {TIME_HEADWAY:.1f} с")

_LOG_W = 72

def _gear_str(elec_data):
    gear = elec_data.get('gear_index', None)
    if gear is None:
        return "?"
    if gear == 0:
        return "N"
    if gear < 0:
        return "R"
    return f"D{int(gear)}"

def _fmt_log(tag, fields):
    body = " | ".join(f"{k}: {v}" for k, v in fields)
    return f"[{tag}] {body}"

# ─────────────────────────────────────────────
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────

def compute_ttc(distance, doppler):
    # TTC = distance / relative_speed, возвращает inf если не сближаемся
    if doppler > 0.1:   # сближаемся
        return distance / doppler
    return float('inf')

def compute_safe_dist(speed):
    #Time headway: дистанция зависит от скорости
    return max(MIN_SAFE_DIST, speed * TIME_HEADWAY)

def pd_acc(dist_error, prev_dist_err, doppler_ms, dt=0.05):
    derivative = (dist_error - prev_dist_err) / dt
    ff = -doppler_ms * ACC_FF_DOPPLER
    return dist_error * ACC_KP - derivative * ACC_KD + ff

def pd_steer(ego_x, prev_ego_x, dt=0.05):
    """PD-контроллер для удержания по X=0."""
    dx = (ego_x - prev_ego_x) / dt
    return max(-1.0, min(1.0, ego_x * STEER_KP + dx * STEER_KD * dt))

def next_state(current, distance, doppler, ttc, us_min, speed, aeb_ticks, leader_stopped):

    # Логика переходов между состояниями, возвращает (новое_состояние, новый_счётчик_aeb).
    # STOP — терминальное состояние, next_state его не трогает
    if current == ADASState.STOP:
        return ADASState.STOP, 0

    # УЗ видит препятствие — всегда CREEP
    if us_min < 5.0:
        return ADASState.CREEP, 0

    # Лидер стоит и мы достаточно близко — входим в CREEP заранее
    if leader_stopped and distance < CREEP_RADAR_DIST:
        return ADASState.CREEP, 0

    # Гистерезис: разные пороги обнаружения и потери лидера
    if current == ADASState.CRUISE:
        if distance < RADAR_DETECT_DIST:
            return ADASState.FOLLOW, 0
        return ADASState.CRUISE, 0

    if current == ADASState.FOLLOW:
        # AEB только после N подтверждённых тиков подряд
        if ttc < TTC_AEB_THRESHOLD:
            new_ticks = aeb_ticks + 1
            if new_ticks >= AEB_CONFIRM_TICKS:
                return ADASState.AEB, new_ticks
            return ADASState.FOLLOW, new_ticks
        # Потеря лидера по увеличенному порогу
        if distance >= RADAR_LOST_DIST:
            return ADASState.CRUISE, 0
        return ADASState.FOLLOW, 0

    if current == ADASState.AEB:
        # Выход из AEB только если машина уже едет (не стоим)
        if ttc > TTC_AEB_RELEASE and speed > 1.0:
            return ADASState.FOLLOW, 0
        return ADASState.AEB, 0

    return current, aeb_ticks

# ─────────────────────────────────────────────
#  ОСНОВНОЙ ЦИКЛ
# ─────────────────────────────────────────────

while True:
    time.sleep(0.05)
    vehicle.poll_sensors()

    # — Лидер —
    elapsed = time.time() - start_time

    if elapsed > LEADER_STOP_AT:
        leader.poll_sensors()
        leader_speed = leader_electrics.data.get('wheelspeed', 0)
        leader.control(throttle=0, brake=1.0)
        if leader_speed < 0.5:
            leader.control(throttle=0, brake=0, parkingbrake=1.0)
            leader_stopped = True
    else:
        leader_stopped = False
        leader_target = LEADER_BASE_SPEED + LEADER_AMPLITUDE * math.sin(
            2 * math.pi * elapsed / LEADER_PERIOD)
        leader.poll_sensors()
        leader_speed = leader_electrics.data.get('wheelspeed', 0)
        leader_error = leader_target - leader_speed

        leader_x = leader_pos_sensor.data.get('pos', (0, 0, 0))[0]
        dlx = leader_x - prev_leader_x
        prev_leader_x = leader_x
        leader_steer = max(-1.0, min(1.0, leader_x * STEER_KP + dlx * STEER_KD))

        leader.control(
            throttle=max(0, min(1, leader_error * 0.1)),
            brake=max(0, min(1, -leader_error * 0.1)),
            steering=leader_steer
        )

    # — Основная машина —
    speed   = electrics.data.get('wheelspeed', 0)
    ego_pos = pos_sensor.data.get('pos', (0, 0, 0))
    ego_x   = ego_pos[0]
    ego_y   = ego_pos[1]

    # Рулевое удержание по 0x=0
    steering   = pd_steer(ego_x, prev_ego_x)
    prev_ego_x = ego_x

    # — Радар с фильтрацией —
    raw_dist, raw_doppler = 9999.0, 0.0
    radar_data = radar.poll()
    if radar_data is not None and radar_data.size > 0:
        closest     = radar_data[radar_data[:, 0].argmin()]
        raw_dist    = float(closest[0])
        raw_doppler = float(closest[1])

    distance = radar_dist_filter.update(raw_dist)
    doppler  = radar_doppler_filter.update(raw_doppler)

    # — Ультразвук с фильтрацией —
    front_dist = us_front_filter.update(us_front.poll().get('distance', 9999))
    rear_dist  = us_rear_filter.update(us_rear.poll().get('distance', 9999))
    us_min     = min(front_dist, rear_dist)

    # — TTC —
    ttc = compute_ttc(distance, doppler)

    # — Переход состояний —
    new_state, aeb_confirm_count = next_state(
        state, distance, doppler, ttc, us_min, speed, aeb_confirm_count, leader_stopped)

    if new_state != state:
        print("-" * _LOG_W)
        print(f"[STATE] {state.value:6s} - {new_state.value}  "
              f"| t={elapsed:.1f}с | {speed*3.6:.1f} км/ч | X: {ego_x:+.2f} м")
        print("-" * _LOG_W)
        # [FIX-3] Сброс prev_dist_err при каждом входе в FOLLOW
        if new_state == ADASState.FOLLOW:
            prev_dist_err = 0.0
        state = new_state

    # — Обновляем LiDAR для визуализации —
    lidar_data = lidar.poll()
    if lidar_data is not None and 'pointCloud' in lidar_data:
        with lidar_shared['lock']:
            lidar_shared['points'] = lidar_data['pointCloud']
            lidar_shared['ego_x']  = ego_x
            lidar_shared['ego_y']  = ego_y

    # ── CRUISE ──────────────────────────────
    if state == ADASState.CRUISE:
        err = TARGET_SPEED - speed
        throttle = max(0, min(1, err * 0.1))
        brake = max(0, min(1, -err * 0.1))
        vehicle.control(throttle=throttle, brake=brake, parkingbrake=0, steering=steering)
        gear_s = _gear_str(electrics.data)
        print(_fmt_log("CRUISE", [
            ("speed", f"{speed*3.6:.1f} км/ч"),
            ("target", f"{TARGET_SPEED*3.6:.0f} км/ч"),
            ("0x", f"{ego_x:+.2f} м"),
            ("gear", gear_s),
        ]))

    # ── FOLLOW ──────────────────────────────
    elif state == ADASState.FOLLOW:
        if not leader_stopped:
            # Вычисляем скорость лидера через доплер
            leader_speed_est = max(0.0, speed - doppler) if doppler > 0 else speed + abs(doppler) * 0.3
            safe_dist = max(MIN_SAFE_DIST, leader_speed_est * TIME_HEADWAY)

            # Целевая скорость = скорость лидера + мягкая коррекция по дистанции
            dist_error = distance - safe_dist
            if dist_error > 0:
                correction = min(dist_error * 0.5, 20.0)  # к скорости лидера

            target_speed = leader_speed_est + correction
            target_speed = max(0, min(TARGET_SPEED, target_speed))

            speed_error = target_speed - speed
            throttle = max(0, min(1, speed_error * 0.1))
            brake = max(0, min(1, -speed_error * 0.1))

            prev_dist_err = dist_error  # сохраняем для лога
            combined = 0  # не используется

        else:
            # Лидер стоит — скоростной контроллер
            raw_target = max(0.0, (distance - 2.0) ** 0.7 * 0.9)
            target_v = min(raw_target, CREEP_BRAKE_SPEED - 0.5 / 3.6)
            speed_err = target_v - speed
            throttle = max(0, min(0.4, speed_err * 0.15))
            brake = max(0, min(1.0, -speed_err * 0.3))
            safe_dist = 8.0
            combined = 0
            prev_dist_err = 0.0
        vehicle.control(throttle=throttle, brake=brake, parkingbrake=0, steering=steering)
        gear_s = _gear_str(electrics.data)
        print(_fmt_log("FOLLOW", [
            ("speed", f"{speed*3.6:.1f} км/ч"),
            ("dist", f"{distance:.1f} м"),
            ("safe_dist", f"{safe_dist:.1f} м"),
            ("ttc", f"{ttc:.1f} с" if ttc != float('inf') else " inf с"),
            ("doppler_s", f"{doppler*3.6:+.1f} км/ч"),
            ("0x", f"{ego_x:+.2f} м"),
            ("gear", gear_s),
        ]))

    # ── AEB ─────────────────────────────────
    elif state == ADASState.AEB:
        vehicle.control(throttle=0, brake=1.0, parkingbrake=0, steering=steering)
        gear_s = _gear_str(electrics.data)
        print(_fmt_log("AEB", [
            ("speed", f"{speed*3.6:.1f} км/ч"),
            ("dist", f"{distance:.1f} м"),
            ("ttc", f"{ttc:5.1f} с" if ttc != float('inf') else "inf с"),
            ("0x", f"{ego_x:+.2f} м"),
            ("gear", gear_s),
        ]))

    # ── CREEP ───────────────────────────────
    elif state == ADASState.CREEP:
        # Если лидер уехал далеко и УЗ не видит — выходим в FOLLOW
        if us_min >= 5.0 and distance > 10.0:
            state = ADASState.FOLLOW
            prev_dist_err = 0.0
            post_aeb_braking = True  # плавный вход
            continue
        gear_s = _gear_str(electrics.data)
        print(_fmt_log("CREEP", [
            ("speed", f"{speed*3.6:.1f} км/ч"),
            ("radar_s", f"{distance:.1f} м"),
            ("us_front", f"{front_dist:.1f} м"),
            ("us_back", f"{rear_dist:.1f} м"),
            ("0x", f"{ego_x:+.2f} м"),
            ("gear", gear_s),
        ]))

        # Если вошли в CREEP на высокой скорости — сначала гасим до CREEP_BRAKE_SPEED
        if speed > CREEP_BRAKE_SPEED:
            # Чем быстрее едем — тем сильнее тормозим. На 60 км/ч = полный тормоз.
            brake_intensity = min(1.0, 0.3 + (speed - CREEP_BRAKE_SPEED) / (40/3.6 - CREEP_BRAKE_SPEED) * 0.7)
            vehicle.control(throttle=0, brake=brake_intensity, parkingbrake=0, steering=steering)
        else:
            # Приоритет остановки: УЗ (точнее), затем радар
            us_stop = front_dist < US_STOP_DIST or rear_dist < US_STOP_DIST
            radar_stop = leader_stopped and distance < CREEP_RADAR_STOP

            if us_stop or radar_stop:
                vehicle.control(throttle=0, brake=1.0, parkingbrake=0, steering=steering)
                if speed < 0.3:
                    vehicle.control(throttle=0, brake=0, parkingbrake=1.0)
                    state = ADASState.STOP
            else:
                # Скорость ползания пропорциональна дистанции — плавно замедляемся при подъезде
                # Коэффициент 0.6 и минимум 3 км/ч дают более уверенное движение
                ref_dist = front_dist if front_dist < 9000 else distance
                target_creep = min(US_CREEP_SPEED, max(3.0 / 3.6, (ref_dist - US_STOP_DIST) * 0.6))
                creep_error = target_creep - speed
                US_CREEP_i_error += creep_error * 0.05
                US_CREEP_i_error  = max(-1.0, min(1.0, US_CREEP_i_error))
                throttle_val = creep_error * 0.25 + US_CREEP_i_error * 0.1
                vehicle.control(
                    throttle=max(0, min(0.25, throttle_val)),
                    brake=max(0, min(0.5, -throttle_val)),
                    parkingbrake=0, steering=steering
                )

    # ── STOP ────────────────────────────────
    elif state == ADASState.STOP:
        vehicle.control(throttle=0, brake=0, parkingbrake=1.0, steering=steering)
        print("=" * _LOG_W)
        print(f"Car stopped! | t={elapsed:.1f}с | X: {ego_x:+.2f} м")
        print("=" * _LOG_W)
        break

input("Press ENTER to continue...")
bng.close()