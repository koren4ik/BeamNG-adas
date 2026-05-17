from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, Radar, Ultrasonic, State, Camera
from collections import deque
from dataclasses import dataclass
from enum import Enum
import argparse
import csv
import math
import time
import numpy as np
from lane_detection import LaneDetector     # Lane keeping detector (отдельный модуль)
from visualizer import Visualizer


# STATE MACHINE
class ADASState(Enum):
    CRUISE = "CRUISE"
    FOLLOW = "FOLLOW"
    AEB    = "AEB"
    CREEP  = "CREEP"
    STOP   = "STOP"


# КОНСТАНТЫ
@dataclass(frozen=True)
class Config:
    # LEADER (для тестов) —
    LEADER_BASE_SPEED:  float = 10 / 3.6
    LEADER_AMPLITUDE:   float = 20 / 3.6
    LEADER_PERIOD:      float = 10.0
    LEADER_STOP_AT:     float = 30.0

    # EGO
    TARGET_SPEED:       float = 64 / 3.6
    # SAFETY_HEADWAY — жёсткий минимум, ниже него регулятор тормозит резко.
    # COMFORT_HEADWAY — целевая дистанция, регулятор стремится к ней.
    SAFETY_HEADWAY:     float = 2.0   # сек — НИЖЕ ЭТОГО НЕЛЬЗЯ
    COMFORT_HEADWAY:    float = 2.5   # сек — целевая для регулятора
    MIN_SAFE_DIST:      float = 10.0
    MIN_COMFORT_DIST:   float = 12.5  # = MIN_SAFE_DIST * (COMFORT/SAFETY)

    # TTC / переключения
    TTC_AEB_THRESHOLD:  float = 3.0
    TTC_AEB_RELEASE:    float = 5.0
    RADAR_DETECT_DIST:  float = 90.0
    RADAR_LOST_DIST:    float = 120.0

    # Фильтр лучей радара. Сенсор возвращает массив ~36k точек на тик — это попадания каждого луча конуса (±34° по вертикали и горизонтали).
    # Без фильтра argmin находит ближайшую точку = асфальт под бампером или часть кузова ego.
    # Нам нужны точки которые реально соответствуют объектам впереди:
    #   - elevation (вертикальный угол луча) близко к 0 — не земля, не небо
    #   - azimuth (горизонтальный угол) близко к 0 — впереди, не сбоку
    #   - intensity достаточно сильная — не помеха
    # Углы в радианах. 0.087 рад ≈ 5°, 0.26 рад ≈ 15°.
    RADAR_ELEV_MAX:     float = 0.087    # ±5° по вертикали
    RADAR_AZIM_MAX:     float = 0.26     # ±15° по горизонтали (шире чтобы не терять цель на лёгком повороте)
    RADAR_INTENS_MIN:   float = 0.3      # минимальная нормализованная интенсивность

    # Doppler-фильтр: игнорируем статичные объекты (знаки, столбы, здания).
    # Объект считается статичным если его доплер ≈ ego_speed:
    #   статичный знак: doppler = ego_speed (мы к нему приближаемся со своей скоростью)
    #   движущийся вместе с нами лидер: doppler ≈ 0
    # Если |doppler - ego_speed| < TOL → объект статичный, отбрасываем.
    # 2.0 м/с ≈ 7 км/ч допустимого "дрейфа" чтобы не отбраковать
    # медленно-движущийся лидер или статичный объект при колебаниях измерения.
    # Применяется только когда ego САМ движется (>1 м/с), иначе на остановке
    # все доплеры около нуля и фильтр убил бы реального стоящего лидера.
    RADAR_DOPPLER_STATIC_TOL: float = 2.0   # м/с
    RADAR_MIN_EGO_SPEED:      float = 1.0   # м/с — порог активации фильтра

    # — УЗ / CREEP —
    US_STOP_DIST:       float = 1.5
    US_CREEP_SPEED:     float = 5 / 3.6
    CREEP_RADAR_DIST:   float = 15.0
    CREEP_RADAR_STOP:   float = 2.0
    CREEP_BRAKE_SPEED:  float = 20 / 3.6
    US_TRIGGER_DIST:    float = 5.0   # когда УЗ начинает считать «препятствие близко»

    # ACC (cascade): внешний контур считает target_speed из дистанции target_v = leader_v + v_corr(distance - target_dist)
    # GAIN АСИММЕТРИЧНЫЙ: при distance < target_dist (мы ближе целевой)
    # Реакция в ACC_DIST_KP_CLOSE раз сильнее. Безопасность важнее быстрого сокращения дистанции!
    ACC_DIST_KP_OPEN:   float = 0.4   # м/с на 1м (когда мы дальше цели)
    ACC_DIST_KP_CLOSE:  float = 0.8   # м/с на 1м (когда мы ближе цели — резче!)
    ACC_DIST_CORR_MAX:  float = 8.0   # максимальная коррекция к скорости лидера, м/с

    # Внутренний PI-контроллер на ошибке скорости (управляет throttle/brake)
    ACC_SPEED_KP:       float = 0.35  # резкий газ при отставании
    ACC_SPEED_KI:       float = 0.05
    ACC_SPEED_I_MAX:    float = 0.5
    ACC_SPEED_DEADZONE: float = 0.5   # м/с — внутри этой зоны throttle=brake=0 (анти-дребезг)
    ACC_BRAKE_GAIN:     float = 0.5   # тормозим в 2 раза мягче чем газуем (комфорт)

    # — PID-контроллер руля (D в нашем случае идёт нах*й)
    # На низких скоростях нужен агрессивный KP чтобы быстро возвращать машину к центру (в манёврах, CREEP, парковке).
    # На высоких — мягкий, потому что тот же руль даёт в разы больший боковой сдвиг за единицу времени, и высокий KP вызывает «змейку».
    STEER_KP_AT_LOW:    float = 0.4    # @ STEER_V_LOW
    STEER_KP_AT_HIGH:   float = 0.18   # @ STEER_V_HIGH (бывшее 0.08 не держало полосу)
    STEER_V_LOW:        float = 10 / 3.6  # м/с
    STEER_V_HIGH:       float = 60 / 3.6  # м/с

    # KD на ВСЕХ скоростях низкий: D работает на разности offset-ов между кадрами, а сам offset шумит (ступеньки квантования, периодические
    # выбросы из детектора). KD=0.3 на скачке offset 0.2‑м даёт ВЕСЬ ход руля - дёрганые рывки.
    # Низкий KD означает что D почти не работает, но лучше иметь стабильный руль без D, чем дёрганый с ним.
    STEER_KD_AT_LOW:    float = 0.05
    STEER_KD_AT_HIGH:   float = 0.05

    # I и I_max — статичные, на высокой скорости интегратор всё равно мало даёт
    STEER_KI:           float = 0.02
    STEER_I_MAX:        float = 0.2

    # Deadzone: при |error| меньше этого значения руль не дёргается.
    STEER_DEADZONE:     float = 0.02   # м (2 см)

    # — Inertia при потере полосы —
    # Сколько тиков подряд держим последний валидный error при отказе детектора.
    # При LOOP_DT=0.02с (50 Гц), 25 тиков = 0.5с.
    # Логика: на короткое выпадение (тени, плохая разметка, артефакты) —
    # продолжаем рулить «по последнему хорошему». На длинное (детектор
    # окончательно потерял полосу) — плавно отпускаем руль к нулю.
    # На повороте это даёт шанс «допройти» поворот вслепую если детектор
    # отвалился на пару кадров.
    STEER_INERTIA_TICKS:    int   = 25
    # Decay-фактор после превышения порога: error *= STEER_INERTIA_DECAY каждый тик.
    # 0.93 значит за 25 тиков (после порога) → 0.16 от исходного → почти 0.
    STEER_INERTIA_DECAY:    float = 0.93

    # — Lookahead steering —
    # Рулим не по offset под бампером, а по offset на LOOKAHEAD_M метров впереди
    # (по полиному полосы). Это даёт «предвидение» — машина начинает крутить руль
    # ДО входа в поворот, а не реагирует когда уже в нём.
    #
    # Геометрия: на bird's-eye 1 пиксель = LANE_WIDTH_M / lane_width_px метров.
    # Bird's-eye высота 400 пикселей соответствует ~30м перед машиной
    # (зависит от калибровки perspective). Так что LOOKAHEAD_M=8м это
    # примерно 100 пикселей вверх от y_eval=399.
    #
    # Lookahead distance scheduled по скорости: на 10 км/ч смотрим близко
    # (большой lookahead даёт большие коррекции на мелочи), на 60 км/ч смотрим
    # далеко (нужно время среагировать на поворот).
    LOOKAHEAD_M_AT_LOW:     float = 4.0    # @ STEER_V_LOW
    LOOKAHEAD_M_AT_HIGH:    float = 12.0   # @ STEER_V_HIGH

    # — Curvature-based speed limiting —
    # Считаем радиус кривизны полосы по полиному, и из него — максимальную
    # безопасную скорость в повороте через формулу v = sqrt(a_lat * R).
    # Где a_lat — комфортное боковое ускорение пассажира.
    #
    # Реальные значения:
    #   2.0 м/с² — очень комфортно, как такси
    #   3.5 м/с² — обычный городской поворот
    #   5.0 м/с² — спортивно, пассажир может прижаться
    # У нас 3.0 — баланс между скоростью и тем чтобы не вылететь.
    LAT_ACCEL_MAX:          float = 3.0    # м/с²
    # Минимум на который можно урезать скорость по curvature.
    # Иначе на очень крутом повороте машина встанет.
    CURVE_MIN_SPEED:        float = 15 / 3.6   # м/с

    # Сглаживание выхода руля
    # Slew rate тоже scheduled: на низкой скорости можно резче поворачивать
    # (нужно для манёвров), на высокой — плавнее (комфорт + безопасность).
    STEER_SLEW_AT_LOW:  float = 1.5
    STEER_SLEW_AT_HIGH: float = 0.6
    # LPF tau — постоянная во времени фильтра. Больше τ = плавнее руль, но медленнее реакция.
    STEER_LPF_TAU:      float = 0.25

    # Throttle/Brake mapping для скоростных регуляторов CRUISE
    SPEED_KP:           float = 0.1   # коэф. пропорционала throttle/brake = err * KP

    # CREEP throttle PI
    CREEP_KP:           float = 0.25
    CREEP_KI:           float = 0.1
    CREEP_DT_GAIN:      float = 0.05  # коэф. интегрирования (как в v0.4: err * 0.05)
    CREEP_I_MAX:        float = 1.0
    CREEP_THROTTLE_MAX: float = 0.25
    CREEP_BRAKE_MAX:    float = 0.5

    # Фильтры
    RADAR_FILTER_SIZE:  int   = 5
    US_FILTER_SIZE:     int   = 5

    # AEB
    AEB_CONFIRM_TICKS:  int   = 3

    # Настройка цикла
    LOOP_DT:            float = 0.02
    DT_MIN:             float = 0.01  # клипуем dt чтобы не делить на ноль
    DT_MAX:             float = 0.2

    # Радар
    RADAR_NO_TARGET:    float = 9999.0  # сторожевое значение «нет цели»

    # Логирование
    LOG_W:              int   = 72
    CSV_PATH:           str   = "adas_log.csv"

CFG = Config()


#  МЕДИАННЫЙ ФИЛЬТР
class MedianFilter:
    def __init__(self, size: int):
        self.buf = deque(maxlen=size)

    def update(self, value: float) -> float:
        self.buf.append(value)
        return float(np.median(self.buf))

    def reset(self):
        self.buf.clear()


#  PID КОНТРОЛЛЕР с anti-windup (clamping) - интегратор не накапливается, если выход уже в насыщении, а ошибка толкает его ещё дальше в то же насыщение.
class PIDController:
    def __init__(self, kp: float, ki: float, kd: float,
                 out_min: float = -1.0, out_max: float = 1.0,
                 i_max: float | None = None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i_max = i_max if i_max is not None else (out_max - out_min)
        self.integral: float = 0.0
        self.prev_error: float | None = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def update(self, error: float, dt: float) -> float:
        # P
        p = self.kp * error

        # D — на первом тике 0, чтобы не было derivative kick
        if self.prev_error is None:
            d = 0.0
        else:
            d = self.kd * (error - self.prev_error) / dt
        self.prev_error = error

        # I — anti-windup: считаем «предсказанный» выход без новой инкрементации
        unsat = p + d + self.ki * self.integral
        push_high = unsat >= self.out_max and error > 0
        push_low  = unsat <= self.out_min and error < 0
        if not (push_high or push_low):
            self.integral += error * dt
            self.integral = max(-self.i_max, min(self.i_max, self.integral))

        out = p + d + self.ki * self.integral
        return max(self.out_min, min(self.out_max, out))


#  ИЗМЕРЕНИЯ (для логирования - сейчас не используется)
@dataclass
class Measurements:
# Снимок сенсоров на текущем тике.
    t:            float = 0.0
    dt:           float = CFG.LOOP_DT
    speed:        float = 0.0
    ego_x:        float = 0.0
    ego_y:        float = 0.0
    distance:     float = CFG.RADAR_NO_TARGET
    doppler:      float = 0.0
    has_target:   bool  = False
    front_dist:   float = CFG.RADAR_NO_TARGET
    rear_dist:    float = CFG.RADAR_NO_TARGET
    us_min:       float = CFG.RADAR_NO_TARGET
    ttc:          float = float('inf')
    leader_stopped: bool = False
    # Lane keeping
    lane_offset_m:  float = 0.0      # offset под бампером (для логирования)
    lane_valid:     bool  = False
    lane_reason:    str   = ""
    # Расширенные данные lane (для lookahead control и curvature speed)
    lane_offset_ahead_m: float = 0.0  # offset на LOOKAHEAD_M метров впереди
    lane_curvature_r:    float = 1e6   # радиус кривизны полосы (м, большой = прямая)


#  ВСПОМОГАТЕЛЬНЫЕ
def compute_ttc(distance: float, doppler: float) -> float:
# TTC = distance / closing_speed. inf если не сближаемся.
    if doppler > 0.1:
        return distance / doppler
    return float('inf')


def estimate_leader_speed(ego_speed: float, doppler: float) -> float:
    # Доплер положителен при сближении (ego быстрее лидера). leader_v = ego_v - doppler — корректно для обеих сторон знака.
    return max(0.0, ego_speed - doppler)


def lerp_by_speed(speed: float, v_low: float, v_high: float,
                  val_low: float, val_high: float) -> float:
    """
    Линейная интерполяция параметра по скорости.
    speed ≤ v_low → val_low (clamped).
    speed ≥ v_high → val_high (clamped).
    Между ними — линейно.
    Используется для gain scheduling регуляторов.
    """
    if speed <= v_low:
        return val_low
    if speed >= v_high:
        return val_high
    t = (speed - v_low) / (v_high - v_low)
    return val_low + t * (val_high - val_low)


def curvature_radius(fit, y_eval, mx_per_px=1.0, my_per_px=1.0) -> float:
    """
    Радиус кривизны параболы x = a*y² + b*y + c в точке y_eval.

    ВАЖНО: fit получен в ПИКСЕЛЬНЫХ координатах bird's-eye. Чтобы получить
    радиус в МЕТРАХ, нужно пересчитать коэффициенты:
        x_m = mx_per_px * x_px
        y_m = my_per_px * y_px
        a_m = mx_per_px / (my_per_px²) * a_px
        b_m = mx_per_px / my_per_px * b_px

    Формула: R = (1 + (dx/dy)²)^(3/2) / |d²x/dy²|

    Возвращает БОЛЬШОЕ число для прямой (|a|→0), маленькое для крутого
    поворота. На прямой клиппуется к 1e6.
    """
    a_px, b_px, _ = fit
    # Пересчитываем коэффициенты под метры
    a_m = mx_per_px / (my_per_px ** 2) * a_px
    b_m = mx_per_px / my_per_px * b_px
    y_m = y_eval * my_per_px
    if abs(a_m) < 1e-7:
        return 1e6  # практически прямая
    return (1.0 + (2*a_m*y_m + b_m)**2)**1.5 / abs(2*a_m)


def offset_at_y(left_fit, right_fit, y_eval, px_per_meter, bev_w):
    """
    Считает offset (в метрах) машины относительно центра полосы
    на уровне y_eval (в координатах bird's-eye).

    y_eval = низ BEV (BEV_H-1) → offset под бампером (как сейчас).
    y_eval меньше → смотрим ДАЛЬШЕ впереди по дороге.

    Знак: offset > 0 = машина (центр кадра) ПРАВЕЕ центра полосы.
    """
    lx = left_fit[0]*y_eval**2  + left_fit[1]*y_eval  + left_fit[2]
    rx = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_center_px = (lx + rx) / 2
    car_pos_px = bev_w / 2
    offset_px = car_pos_px - lane_center_px
    return offset_px / px_per_meter


def _extract_camera_rgb(cam_data) -> np.ndarray | None:
    if cam_data is None:
        return None
    img = None
    if isinstance(cam_data, dict):
        for key in ('colour', 'color'):
            if key in cam_data:
                img = cam_data[key]
                break
    if img is None:
        img = cam_data
    if hasattr(img, 'mode'):  # PIL.Image
        img = np.array(img)
    if not isinstance(img, np.ndarray):
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim != 3 or img.shape[2] != 3:
        return None
    return img


def gear_str(elec_data: dict) -> str:
    gear = elec_data.get('gear_index', None)
    if gear is None:
        return "?"
    if gear == 0:
        return "N"
    if gear < 0:
        return "R"
    return f"D{int(gear)}"


def fmt_log(tag: str, fields: list[tuple[str, str]]) -> str:
    body = " | ".join(f"{k}: {v}" for k, v in fields)
    return f"[{tag}] {body}"


# STATE MACHINE — чистая функция (тестируемая без BeamNG)
def next_state(current: ADASState, m: Measurements,
               aeb_ticks: int) -> tuple[ADASState, int]:
    # STOP — терминальное
    if current == ADASState.STOP:
        return ADASState.STOP, 0

    # УЗ препятствие — всегда CREEP
    if m.us_min < CFG.US_TRIGGER_DIST:
        return ADASState.CREEP, 0

    # Лидер стоит и близко — заранее в CREEP
    if m.leader_stopped and m.has_target and m.distance < CFG.CREEP_RADAR_DIST:
        return ADASState.CREEP, 0

    if current == ADASState.CRUISE:
        if m.has_target and m.distance < CFG.RADAR_DETECT_DIST:
            return ADASState.FOLLOW, 0
        return ADASState.CRUISE, 0

    if current == ADASState.FOLLOW:
        # AEB-инкремент только пока ttc низкий; иначе — сброс
        if m.ttc < CFG.TTC_AEB_THRESHOLD:
            new_ticks = aeb_ticks + 1
            if new_ticks >= CFG.AEB_CONFIRM_TICKS:
                return ADASState.AEB, new_ticks
            return ADASState.FOLLOW, new_ticks
        # Потеря лидера
        if (not m.has_target) or m.distance >= CFG.RADAR_LOST_DIST:
            return ADASState.CRUISE, 0
        return ADASState.FOLLOW, 0  # сброс aeb_ticks на 0

    if current == ADASState.AEB:
        if m.ttc > CFG.TTC_AEB_RELEASE and m.speed > 1.0:
            return ADASState.FOLLOW, 0
        return ADASState.AEB, 0

    if current == ADASState.CREEP:
        # Выход CREEP - FOLLOW: УЗ свободен и радар видит цель далеко
        if m.us_min >= CFG.US_TRIGGER_DIST and m.has_target and m.distance > 10.0:
            return ADASState.FOLLOW, 0
        # Если радар вообще цель потерял и УЗ свободен — в CRUISE
        if m.us_min >= CFG.US_TRIGGER_DIST and not m.has_target:
            return ADASState.CRUISE, 0
        return ADASState.CREEP, 0

    return current, aeb_ticks


# ADAS CONTROLLER — инкапсулирует всё состояние машины
class ADASController:
    def __init__(self, vehicle, sensors_bundle):
        self.vehicle = vehicle
        (self.electrics, self.pos_sensor, self.radar,
         self.us_front, self.us_rear, self.camera) = sensors_bundle

        # Регуляторы
        # KP/KD руля устанавливаются динамически в compute_steering() через gain scheduling по скорости. Здесь — стартовые значения.
        self.steer_pid = PIDController(
            CFG.STEER_KP_AT_LOW, CFG.STEER_KI, CFG.STEER_KD_AT_LOW,
            out_min=-1.0, out_max=1.0, i_max=CFG.STEER_I_MAX,
        )
        # ACC: cascade. Внешний контур (target_speed из дистанции) — без памяти. Внутренний — PI на ошибке скорости.
        self.acc_speed_pid = PIDController(
            CFG.ACC_SPEED_KP, CFG.ACC_SPEED_KI, kd=0.0,
            out_min=-1.0, out_max=1.0, i_max=CFG.ACC_SPEED_I_MAX,
        )
        # CREEP — простой PI на ошибке скорости
        self.creep_pid = PIDController(
            CFG.CREEP_KP, CFG.CREEP_KI, kd=0.0,
            out_min=-CFG.CREEP_BRAKE_MAX, out_max=CFG.CREEP_THROTTLE_MAX,
            i_max=CFG.CREEP_I_MAX,
        )

        # Lane detection
        self.lane_detector = LaneDetector(debug=False)
        # Статистика для лога
        self.lane_stats = {'ok': 0, 'fallback': 0, 'invalid': 0}

        # Фильтры
        self.radar_dist_filter = MedianFilter(CFG.RADAR_FILTER_SIZE)
        self.radar_doppler_filter = MedianFilter(CFG.RADAR_FILTER_SIZE)
        self.us_front_filter = MedianFilter(CFG.US_FILTER_SIZE)
        self.us_rear_filter = MedianFilter(CFG.US_FILTER_SIZE)

        # Состояние
        self.state: ADASState = ADASState.CRUISE
        self.aeb_ticks: int = 0
        self.t_prev: float = time.time()
        # Предыдущее значение руля для slew rate limiter и LPF
        self.prev_steering: float = 0.0
        # Inertia при отказе детектора полосы:
        #   last_valid_error — последний error который мы получили при lane_valid=True
        #   invalid_ticks — счётчик последовательных тиков с lane_valid=False
        self.last_valid_error: float = 0.0
        self.invalid_ticks: int = 0
        # Для визуализатора и lookahead — последний кадр и lane_result
        self.last_frame_rgb = None
        self.last_lane_result = None

    #СБОР ИЗМЕРЕНИЙ
    def measure(self, t_now: float, leader_stopped: bool) -> Measurements:
        m = Measurements()
        m.t = t_now
        dt_raw = t_now - self.t_prev
        m.dt = max(CFG.DT_MIN, min(CFG.DT_MAX, dt_raw))
        self.t_prev = t_now

        m.speed = self.electrics.data.get('wheelspeed', 0.0)
        ego_pos = self.pos_sensor.data.get('pos', (0.0, 0.0, 0.0))
        m.ego_x = ego_pos[0]
        m.ego_y = ego_pos[1]

        # Радар — отделяем «нет цели» от «цель далеко».
        # Сенсор возвращает ~36k точек/тик (все лучи конуса). Без фильтра
        # ближайшая точка это ВСЕГДА асфальт под бампером (elevation в минус)
        # или часть кузова ego (elevation в плюс). Фильтр оставляет только
        # горизонтальные передние сильные сигналы — реальные объекты.
        # Структура колонок (выяснено через radar_inspect / radar_find_doppler):
        #   col 0 — distance (м)
        #   col 1 — doppler (м/с, положительный = сближение)
        #   col 2 — azimuth (рад)
        #   col 3 — elevation (рад)
        #   col 6 — нормализованная интенсивность (0..1)
        radar_data = self.radar.poll()
        if radar_data is not None and radar_data.size > 0:
            elev   = radar_data[:, 3]
            azim   = radar_data[:, 2]
            intens = radar_data[:, 6]
            mask = (
                (np.abs(elev)   < CFG.RADAR_ELEV_MAX)
                & (np.abs(azim) < CFG.RADAR_AZIM_MAX)
                & (intens > CFG.RADAR_INTENS_MIN)
            )
            filtered = radar_data[mask]

            # Дополнительный фильтр: убираем СТАТИЧНЫЕ объекты (столбы, знаки,
            # здания). У статичного объекта doppler ≈ ego_speed (потому что
            # мы к нему приближаемся со своей скоростью). У движущегося с нами
            # лидера doppler ≈ 0. Если |doppler - ego_speed| мало — это знак,
            # игнорируем.
            # ВАЖНО: фильтр работает только когда ego САМ движется. Иначе
            # все доплеры около 0 и мы бы отбраковали стоящего лидера
            # (что критично для CREEP-сценария).
            if (filtered.size > 0
                    and m.speed > CFG.RADAR_MIN_EGO_SPEED):
                obj_doppler = filtered[:, 1]
                moving_mask = (
                    np.abs(obj_doppler - m.speed) > CFG.RADAR_DOPPLER_STATIC_TOL
                )
                filtered = filtered[moving_mask]
        else:
            filtered = None

        if filtered is not None and filtered.size > 0:
            closest = filtered[filtered[:, 0].argmin()]
            raw_dist = float(closest[0])
            raw_doppler = float(closest[1])
            m.distance = self.radar_dist_filter.update(raw_dist)
            m.doppler = self.radar_doppler_filter.update(raw_doppler)
            m.has_target = True
        else:
            # После фильтра ничего не осталось — впереди нет цели.
            m.distance = CFG.RADAR_NO_TARGET
            m.doppler = 0.0
            m.has_target = False

        # УЗ
        m.front_dist = self.us_front_filter.update(
            self.us_front.poll().get('distance', CFG.RADAR_NO_TARGET))
        m.rear_dist = self.us_rear_filter.update(
            self.us_rear.poll().get('distance', CFG.RADAR_NO_TARGET))
        m.us_min = min(m.front_dist, m.rear_dist)

        # TTC и лидер
        m.ttc = compute_ttc(m.distance, m.doppler) if m.has_target else float('inf')
        m.leader_stopped = leader_stopped

        # Lane detection — обработка кадра с фронтальной камеры.
        try:
            cam_data = self.camera.poll()
            frame_rgb = _extract_camera_rgb(cam_data)
            if frame_rgb is not None:
                lane_result = self.lane_detector.detect(frame_rgb)
                self.last_lane_result = lane_result
                self.last_frame_rgb = frame_rgb
                if lane_result.valid:
                    m.lane_offset_m = lane_result.offset_m
                    m.lane_valid = True
                    m.lane_reason = lane_result.reason
                    if 'fallback' in lane_result.reason:
                        self.lane_stats['fallback'] += 1
                    else:
                        self.lane_stats['ok'] += 1

                    # Считаем lookahead-offset и curvature.
                    # Они нужны для steering control и curvature speed limiting.
                    if (lane_result.left_fit is not None
                            and lane_result.right_fit is not None):
                        from lane_detection import BEV_W, BEV_H, LANE_WIDTH_M
                        # Lookahead distance scheduled по скорости:
                        # на 10 км/ч → 4м, на 60 км/ч → 12м
                        lookahead_m = lerp_by_speed(
                            m.speed, CFG.STEER_V_LOW, CFG.STEER_V_HIGH,
                            CFG.LOOKAHEAD_M_AT_LOW, CFG.LOOKAHEAD_M_AT_HIGH)
                        # Размеры в метрах на пиксель BEV.
                        # mx — поперёк (ширина полосы / её пиксельный размер)
                        # my — вдоль (BEV_H пикселей ≈ 30м перед машиной).
                        lf = lane_result.left_fit
                        rf = lane_result.right_fit
                        y_bot = BEV_H - 1
                        lx_b = lf[0]*y_bot**2 + lf[1]*y_bot + lf[2]
                        rx_b = rf[0]*y_bot**2 + rf[1]*y_bot + rf[2]
                        lane_width_px = max(1.0, rx_b - lx_b)
                        mx_per_px = LANE_WIDTH_M / lane_width_px
                        my_per_px = 30.0 / BEV_H   # калибровочная константа
                        px_per_meter = 1.0 / mx_per_px
                        # Lookahead в пикселях:
                        lookahead_px = lookahead_m / my_per_px
                        y_ahead = max(0, y_bot - lookahead_px)
                        # Offset на y_ahead (в метрах)
                        m.lane_offset_ahead_m = offset_at_y(
                            lf, rf, y_ahead, px_per_meter, BEV_W)
                        # Curvature в МЕТРАХ
                        r_left = curvature_radius(lf, y_ahead,
                                                  mx_per_px, my_per_px)
                        r_right = curvature_radius(rf, y_ahead,
                                                   mx_per_px, my_per_px)
                        m.lane_curvature_r = (r_left + r_right) / 2
                else:
                    m.lane_valid = False
                    m.lane_reason = lane_result.reason
                    self.lane_stats['invalid'] += 1
        except Exception as e:
            m.lane_valid = False
            m.lane_reason = f"exception: {e}"
            self.lane_stats['invalid'] += 1

        return m

    #ПЕРЕХОД СОСТОЯНИЙ + ПРОБРОС reset() РЕГУЛЯТОРОВ
    def transition(self, m: Measurements):
        new_state, new_ticks = next_state(self.state, m, self.aeb_ticks)
        if new_state != self.state:
            print("-" * CFG.LOG_W)
            print(f"[STATE] {self.state.value:6s} → {new_state.value}  "
                  f"| t={m.t:.1f}с | {m.speed*3.6:.1f} км/ч | X: {m.ego_x:+.2f} м")
            print("-" * CFG.LOG_W)

            # Сбрасываем регуляторы при входе в новый режим, чтобы интегралы
            # и prev_error не «протекли» из прошлой логики
            if new_state == ADASState.FOLLOW:
                self.acc_speed_pid.reset()
            if new_state == ADASState.CREEP:
                self.creep_pid.reset()
            # AEB-счётчик [BUG-5] — гарантированно сбрасываем при выходе из FOLLOW
            if self.state == ADASState.FOLLOW and new_state != ADASState.FOLLOW:
                new_ticks = 0
            self.state = new_state
        self.aeb_ticks = new_ticks

    #РУЛЬ (PID (помним про D) + gain scheduling + dead-zone + LPF + slew rate)
    def compute_steering(self, m: Measurements) -> float:
        # 0. GAIN SCHEDULING — обновляем коэффициенты под текущую скорость
        self.steer_pid.kp = lerp_by_speed(
            m.speed, CFG.STEER_V_LOW, CFG.STEER_V_HIGH,
            CFG.STEER_KP_AT_LOW, CFG.STEER_KP_AT_HIGH)
        self.steer_pid.kd = lerp_by_speed(
            m.speed, CFG.STEER_V_LOW, CFG.STEER_V_HIGH,
            CFG.STEER_KD_AT_LOW, CFG.STEER_KD_AT_HIGH)
        slew_rate = lerp_by_speed(
            m.speed, CFG.STEER_V_LOW, CFG.STEER_V_HIGH,
            CFG.STEER_SLEW_AT_LOW, CFG.STEER_SLEW_AT_HIGH)

        # — ВЫБОР SETPOINT с INERTIA при отказе детектора —
        # При lane_valid=True — рулим по смещению, запоминаем последний error.
        # При lane_valid=False:
        #   • первые N тиков (STEER_INERTIA_TICKS) — держим last_valid_error
        #     как есть. Это «память» руля: на короткие отказы (тени, артефакты)
        #     мы не сбрасываемся в 0, а продолжаем рулить как только что.
        #   • после N тиков — медленно decay'им к нулю (STEER_INERTIA_DECAY),
        #     чтобы избежать «зависшего» руля при долгом отказе детектора.
        # ЗНАК: при lane_offset_m > 0 (машина вправо от центра) нужен steering < 0
        # для возврата → error = -lane_offset_m.
        # — ВЫБОР SETPOINT — LOOKAHEAD CONTROL —
        # Рулим не по текущему offset (под бампером), а по offset на LOOKAHEAD_M
        # метров впереди (по нашему полиному). Это даёт «предвидение» поворота:
        # ego начинает крутить руль ДО входа в поворот, а не реагирует когда
        # уже в нём.
        # ЗНАК: при offset_ahead > 0 (центр полосы впереди СЛЕВА от машины) → steer < 0.
        if m.lane_valid:
            error = -m.lane_offset_ahead_m
            self.last_valid_error = error
            self.invalid_ticks = 0
        else:
            self.invalid_ticks += 1
            if self.invalid_ticks <= CFG.STEER_INERTIA_TICKS:
                # Короткий отказ — держим последний хороший error
                error = self.last_valid_error
            else:
                # Долгий отказ — decay'им
                self.last_valid_error *= CFG.STEER_INERTIA_DECAY
                error = self.last_valid_error

        # Dead-zone: внутри ±STEER_DEADZONE считаем что мы «в нуле».
        if abs(error) < CFG.STEER_DEADZONE:
            error = 0.0

        # 1. Сырой выход PID
        raw = self.steer_pid.update(error, m.dt)

        # 2. Low-pass filter
        if CFG.STEER_LPF_TAU > 0:
            alpha = m.dt / (CFG.STEER_LPF_TAU + m.dt)
            filtered = alpha * raw + (1.0 - alpha) * self.prev_steering
        else:
            filtered = raw

        # 3. Slew rate limiter (динамический)
        max_delta = slew_rate * m.dt
        delta = filtered - self.prev_steering
        if delta > max_delta:
            filtered = self.prev_steering + max_delta
        elif delta < -max_delta:
            filtered = self.prev_steering - max_delta

        # Финальный clip и сохранение
        filtered = max(-1.0, min(1.0, filtered))
        self.prev_steering = filtered
        return filtered

    #CRUISE
    def control_cruise(self, m: Measurements, steering: float):
        # Curvature-based speed limiting: на повороте снижаем целевую скорость.
        # Формула v_max = sqrt(a_lat * R), где
        #   a_lat — комфортное боковое ускорение (CFG.LAT_ACCEL_MAX)
        #   R — радиус кривизны полосы впереди
        # Это даёт классическое поведение «автомобиль чувствует поворот и
        # сбрасывает скорость», как у штатных ACC + LKA систем.
        target_speed = CFG.TARGET_SPEED
        if m.lane_valid and m.lane_curvature_r < 1e5:
            v_max_curve = math.sqrt(CFG.LAT_ACCEL_MAX * m.lane_curvature_r)
            v_max_curve = max(CFG.CURVE_MIN_SPEED, v_max_curve)
            target_speed = min(CFG.TARGET_SPEED, v_max_curve)

        err = target_speed - m.speed
        throttle = max(0.0, min(1.0, err * CFG.SPEED_KP))
        brake    = max(0.0, min(1.0, -err * CFG.SPEED_KP))
        self.vehicle.control(throttle=throttle, brake=brake,
                             parkingbrake=0, steering=steering)
        return throttle, brake

    #FOLLOW (cascade ACC: дистанция - target_speed - throttle/brake)
    def control_follow(self, m: Measurements, steering: float):
        """
        Cascade-схема:
          1) Внешний контур (без памяти): из дистанции и скорости лидера
             вычисляем target_speed.
          2) Внутренний PI: из ошибки скорости (target − ego) вычисляем
             комбинированную команду.
          3) Маппинг в throttle/brake с dead-zone (анти-дребезг) и
             асимметричным усилением тормоза (комфорт).
        """
        if not m.leader_stopped:
            # Внешний контур: target_speed
            leader_v = estimate_leader_speed(m.speed, m.doppler)

            # Две дистанции: safety (минимум) и comfort (target регулятора)
            safety_dist  = max(CFG.MIN_SAFE_DIST,    leader_v * CFG.SAFETY_HEADWAY)
            comfort_dist = max(CFG.MIN_COMFORT_DIST, leader_v * CFG.COMFORT_HEADWAY)

            # Ошибка относительно COMFORT (а не safety!) — стремимся к комфортной дистанции, оставляя запас до safety.
            dist_error = m.distance - comfort_dist

            # Асимметричный gain: ниже целевой реагируем резче
            kp = CFG.ACC_DIST_KP_OPEN if dist_error >= 0 else CFG.ACC_DIST_KP_CLOSE
            v_corr = max(-CFG.ACC_DIST_CORR_MAX,
                         min(CFG.ACC_DIST_CORR_MAX, dist_error * kp))

            target_v = leader_v + v_corr
            target_v = max(0.0, min(CFG.TARGET_SPEED, target_v))

            # Внутренний контур: PI на ошибке скорости
            speed_err = target_v - m.speed
            cmd = self.acc_speed_pid.update(speed_err, m.dt)

            # Маппинг в throttle/brake
            # ВАЖНО: проверка на нарушение SAFETY имеет приоритет.
            # Если зашли ниже safety_dist — гарантированно тормозим, независимо от того, что говорит PI скорости.
            if m.distance < safety_dist and m.doppler > -0.5:
                # Сближаемся или почти не отдаляемся, и ниже safety — экстренный комфортный тормоз
                throttle, brake = 0.0, min(1.0, CFG.ACC_BRAKE_GAIN * 1.5)
            elif abs(speed_err) < CFG.ACC_SPEED_DEADZONE:
                throttle, brake = 0.0, 0.0
            elif cmd >= 0:
                throttle = min(1.0, cmd)
                brake = 0.0
            else:
                # Не тормозим, если лидер удаляется — это бессмысленно.
                if m.doppler < -0.5:
                    throttle, brake = 0.0, 0.0
                else:
                    throttle = 0.0
                    brake = min(1.0, -cmd * CFG.ACC_BRAKE_GAIN)

            # Возвращаем оба значения для логирования и анализа
            safe_dist = comfort_dist

        else:
            # Лидер стоит — мягкий съезд по дистанции (квадратный корень)
            raw_target = max(0.0, (m.distance - 2.0) ** 0.7 * 0.9) if m.has_target else 0.0
            target_v = min(raw_target, CFG.CREEP_BRAKE_SPEED - 0.5 / 3.6)
            speed_err = target_v - m.speed
            throttle = max(0.0, min(0.4, speed_err * 0.15))
            brake    = max(0.0, min(1.0, -speed_err * 0.3))
            safe_dist = 8.0
            safety_dist = comfort_dist = 8.0

        self.vehicle.control(throttle=throttle, brake=brake,
                             parkingbrake=0, steering=steering)
        return throttle, brake, safe_dist, safety_dist, comfort_dist

    # AEB
    def control_aeb(self, m: Measurements, steering: float):
        self.vehicle.control(throttle=0, brake=1.0, parkingbrake=0, steering=steering)
        return 0.0, 1.0

    # CREEP
    def control_creep(self, m: Measurements, steering: float):
        radar_lying = (m.has_target and m.distance < 4.0 and m.front_dist > 4.0)

        # Фаза 1 — гасим скорость до CREEP_BRAKE_SPEED
        if m.speed > CFG.CREEP_BRAKE_SPEED:
            denom = max(1e-3, (40 / 3.6 - CFG.CREEP_BRAKE_SPEED))
            brake_intensity = min(1.0, 0.3 + (m.speed - CFG.CREEP_BRAKE_SPEED) / denom * 0.7)
            self.vehicle.control(throttle=0, brake=brake_intensity,
                                 parkingbrake=0, steering=steering)
            return 0.0, brake_intensity, radar_lying

        # Фаза 2 — ползём
        us_stop = (m.front_dist < CFG.US_STOP_DIST or m.rear_dist < CFG.US_STOP_DIST)
        radar_stop = (not radar_lying) and m.leader_stopped \
                     and m.has_target and m.distance < CFG.CREEP_RADAR_STOP

        if us_stop or radar_stop:
            self.vehicle.control(throttle=0, brake=1.0,
                                 parkingbrake=0, steering=steering)
            if m.speed < 0.3:
                self.vehicle.control(throttle=0, brake=0, parkingbrake=1.0)
                self.state = ADASState.STOP
            return 0.0, 1.0, radar_lying

        # Целевая скорость ползания
        if radar_lying:
            target_creep = CFG.US_CREEP_SPEED
        else:
            ref_dist = m.front_dist if m.front_dist < 9000 else m.distance
            target_creep = min(CFG.US_CREEP_SPEED,
                               max(3.0 / 3.6, (ref_dist - CFG.US_STOP_DIST) * 0.6))

        creep_error = target_creep - m.speed
        cmd = self.creep_pid.update(creep_error, m.dt)
        throttle = max(0.0, min(CFG.CREEP_THROTTLE_MAX, cmd))
        brake    = max(0.0, min(CFG.CREEP_BRAKE_MAX, -cmd))
        self.vehicle.control(throttle=throttle, brake=brake,
                             parkingbrake=0, steering=steering)
        return throttle, brake, radar_lying

    # STOP
    def control_stop(self, m: Measurements, steering: float):
        self.vehicle.control(throttle=0, brake=0, parkingbrake=1.0, steering=steering)


#  ЛИДЕР (PID руля + простой пропорциональный спид-контроллер)
class LeaderDriver:
    def __init__(self, leader_vehicle, leader_electrics, leader_pos_sensor):
        self.veh = leader_vehicle
        self.elec = leader_electrics
        self.pos = leader_pos_sensor
        self.steer_pid = PIDController(
            CFG.STEER_KP_AT_LOW, CFG.STEER_KI, CFG.STEER_KD_AT_LOW,
            out_min=-1.0, out_max=1.0, i_max=CFG.STEER_I_MAX)
        self.t_prev = time.time()
        self.stopped = False

    def step(self, t_now: float, elapsed: float) -> bool:
        dt = max(CFG.DT_MIN, min(CFG.DT_MAX, t_now - self.t_prev))
        self.t_prev = t_now

        if elapsed > CFG.LEADER_STOP_AT:
            self.veh.poll_sensors()
            leader_speed = self.elec.data.get('wheelspeed', 0)
            self.veh.control(throttle=0, brake=1.0)
            if leader_speed < 0.5:
                self.veh.control(throttle=0, brake=0, parkingbrake=1.0)
                self.stopped = True
            return self.stopped

        self.stopped = False
        target = CFG.LEADER_BASE_SPEED + CFG.LEADER_AMPLITUDE * math.sin(
            2 * math.pi * elapsed / CFG.LEADER_PERIOD)
        self.veh.poll_sensors()
        speed = self.elec.data.get('wheelspeed', 0)
        err = target - speed

        leader_x = self.pos.data.get('pos', (0, 0, 0))[0]
        steer = self.steer_pid.update(leader_x, dt)  # знак как в v0.4

        self.veh.control(
            throttle=max(0, min(1, err * CFG.SPEED_KP)),
            brake=max(0, min(1, -err * CFG.SPEED_KP)),
            steering=steer,
        )
        return False


#  CSV-логирование
class CsvLogger:
    HEADER = ['t', 'state', 'speed_kmh', 'ego_x', 'distance', 'has_target',
              'doppler', 'ttc', 'us_min', 'throttle', 'brake', 'steering',
              'safety_dist', 'comfort_dist',
              'lane_offset', 'lane_valid', 'lane_reason']

    def __init__(self, path: str):
        self.f = open(path, 'w', newline='', encoding='utf-8')
        self.w = csv.writer(self.f)
        self.w.writerow(self.HEADER)

    def write(self, m: Measurements, state: ADASState,
              throttle: float, brake: float, steering: float,
              safety_dist: float = 0.0, comfort_dist: float = 0.0):
        self.w.writerow([
            f"{m.t:.3f}", state.value, f"{m.speed*3.6:.2f}",
            f"{m.ego_x:.3f}", f"{m.distance:.2f}", int(m.has_target),
            f"{m.doppler:.2f}",
            f"{m.ttc:.2f}" if m.ttc != float('inf') else "inf",
            f"{m.us_min:.2f}", f"{throttle:.3f}", f"{brake:.3f}", f"{steering:.3f}",
            f"{safety_dist:.2f}", f"{comfort_dist:.2f}",
            f"{m.lane_offset_m:.3f}", int(m.lane_valid), m.lane_reason,
        ])
        self.f.flush()

    def close(self):
        self.f.close()


#  MAIN
def main():
    # Аргументы запуска
    parser = argparse.ArgumentParser(description="ADAS v1.0 — Lane Keeping Assistant")
    parser.add_argument('--visualize', action='store_true',
                        help='Включить окно визуализации (камера + BEV + HUD)')
    args = parser.parse_args()

    bng = BeamNGpy('localhost', 64256, home=r'D:\Scripts\BeamNG.tech.v0.38.5.0')
    bng.open(launch=True)

    scenario = Scenario('automation_test_track', 'adas_v05')
    vehicle = Vehicle('ego',    model='etk800', license='ADAS-V05')
    leader  = Vehicle('leader', model='etk800', license='LEAD')
    scenario.add_vehicle(vehicle, pos=(155.715, -289.962, 120.839), rot_quat=(0, 0, 0.705, 0.709), cling=True)
    scenario.add_vehicle(leader, pos=(0, 0, 0), cling=True)
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    time.sleep(1)

    # — Сенсоры ego —
    electrics = Electrics()
    pos_sensor = State()
    vehicle.attach_sensor('electrics', electrics)
    vehicle.attach_sensor('pos_sensor', pos_sensor)

    # Радар сконфигурирован как «настоящий» автомобильный ACC long-range:
    #   • horizontal FOV ±12° (типичный для ACC)
    #   • vertical FOV ±3° (узкий — не цепляем землю/небо)
    #   • range 3..100м (под ACC, не нужен ближе чем 3м — это для УЗ)
    # Раньше дефолты давали ±34° конус, и ближайшие лучи попадали в асфальт.
    # Теперь физически отсеяно на уровне сенсора, плюс фильтр в measure()
    # как защита второго эшелона.
    radar = Radar(
        'radar', bng, vehicle,
        pos=(0, -2.3, 0.8),
        dir=(0, -1, 0),
        field_of_view_y=6.0,                 # вертикальный FOV, градусы (±3°)
        half_angle_deg=12.0,      # горизонтальный half-angle (итого ±12°)
        range_min=3.0,            # ближе не интересует — это работа УЗ
        range_max=100.0,
        near_far_planes=(0.5, 120.0),
        is_visualised=True,
    )
    us_front = Ultrasonic('us_front', bng, vehicle,
                          pos=(0, -2.35, 0.5), dir=(0, -1, 0),
                          near_far_planes=(0.1, 6.0), is_visualised=True)
    us_rear = Ultrasonic('us_rear', bng, vehicle,
                         pos=(0, 2.35, 0.5), dir=(0, 1, 0),
                         near_far_planes=(0.1, 6.0), is_visualised=True)
    #Камера для lane keeping. Параметры синхронизированы с camera_test.py (те же что использовались для калибровки lane_detection).
    camera = Camera(
        'cam_front', bng, vehicle,
        pos=(0, -1.5, 1.3),
        dir=(0, -1, -0.1),
        resolution=(640, 360),
        is_render_colours=True,
        is_render_depth=False,
        is_render_annotations=False,
    )

    # Сенсоры лидера
    leader_electrics = Electrics()
    leader_pos_sensor = State()
    leader.attach_sensor('leader_electrics', leader_electrics)
    leader.attach_sensor('leader_pos_sensor', leader_pos_sensor)

    # Контроллеры
    adas = ADASController(vehicle,
                          (electrics, pos_sensor, radar, us_front, us_rear, camera))
    leader_drv = LeaderDriver(leader, leader_electrics, leader_pos_sensor)
    logger = CsvLogger(CFG.CSV_PATH)
    viz = Visualizer(enabled=args.visualize)

    print(" ADAS v1 (beta) by koren4ik | BeamNG.tech | ACC + AEB + LKA")
    print(f" Start STATE: {adas.state.value}")
    print(f" Target speed: {CFG.TARGET_SPEED*3.6:.0f} км/ч")
    print(f" Headway: safety={CFG.SAFETY_HEADWAY:.1f}с, comfort={CFG.COMFORT_HEADWAY:.1f}с")
    print(f" CSV log: {CFG.CSV_PATH}")
    print(f" Lane setpoint: при lane_valid → lane_offset_m, иначе fallback → ego_x")
    print(f" Visualizer: {'ON' if args.visualize else 'OFF (запусти с --visualize)'}")

    start_time = time.time()
    next_tick = start_time

    try:
        while True:
            now = time.time()
            sleep_for = next_tick - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            next_tick += CFG.LOOP_DT

            t_now = time.time()
            elapsed = t_now - start_time

            # LEADER
            leader_stopped = leader_drv.step(t_now, elapsed)

            # EGO
            vehicle.poll_sensors()
            m = adas.measure(t_now, leader_stopped)

            # Переход состояний
            adas.transition(m)

            # Управление в зависимости от состояния
            steering = adas.compute_steering(m)
            throttle, brake = 0.0, 0.0
            safety_dist, comfort_dist = 0.0, 0.0  # для CSV — заполняются только в FOLLOW

            if adas.state == ADASState.CRUISE:
                throttle, brake = adas.control_cruise(m, steering)
                lane_info = (f"{m.lane_offset_m:+.2f}м"
                             if m.lane_valid else "n/a")
                print(fmt_log("CRUISE", [
                    ("speed",  f"{m.speed*3.6:.1f} км/ч"),
                    ("target", f"{CFG.TARGET_SPEED*3.6:.0f} км/ч"),
                    ("lane",   lane_info),
                    ("0x",     f"{m.ego_x:+.2f} м"),
                    ("gear",   gear_str(electrics.data)),
                ]))

            elif adas.state == ADASState.FOLLOW:
                throttle, brake, safe_dist, safety_dist, comfort_dist = \
                    adas.control_follow(m, steering)
                # Помечаем нарушение SAFETY чтобы было видно в консоли
                safety_marker = "  ⚠SAFETY" if m.distance < safety_dist else ""
                lane_info = (f"{m.lane_offset_m:+.2f}м"
                             if m.lane_valid else "n/a")
                print(fmt_log("FOLLOW", [
                    ("speed",     f"{m.speed*3.6:.1f} км/ч"),
                    ("dist",      f"{m.distance:.1f} м{safety_marker}"),
                    ("safety",    f"{safety_dist:.1f} м"),
                    ("comfort",   f"{comfort_dist:.1f} м"),
                    ("ttc",       f"{m.ttc:.1f} с" if m.ttc != float('inf') else " inf с"),
                    ("doppler_s", f"{m.doppler*3.6:+.1f} км/ч"),
                    ("lane",      lane_info),
                    ("0x",        f"{m.ego_x:+.2f} м"),
                    ("gear",      gear_str(electrics.data)),
                ]))

            elif adas.state == ADASState.AEB:
                throttle, brake = adas.control_aeb(m, steering)
                print(fmt_log("AEB", [
                    ("speed", f"{m.speed*3.6:.1f} км/ч"),
                    ("dist",  f"{m.distance:.1f} м"),
                    ("ttc",   f"{m.ttc:5.1f} с" if m.ttc != float('inf') else "inf с"),
                    ("0x",    f"{m.ego_x:+.2f} м"),
                    ("gear",  gear_str(electrics.data)),
                ]))

            elif adas.state == ADASState.CREEP:
                throttle, brake, radar_lying = adas.control_creep(m, steering)
                print(fmt_log("CREEP", [
                    ("speed",    f"{m.speed*3.6:.1f} км/ч"),
                    ("radar_s",  f"{m.distance:.1f} м{'  [BAD]' if radar_lying else ''}"),
                    ("us_front", f"{m.front_dist:.1f} м"),
                    ("us_back",  f"{m.rear_dist:.1f} м"),
                    ("0x",       f"{m.ego_x:+.2f} м"),
                    ("gear",     gear_str(electrics.data)),
                ]))

            elif adas.state == ADASState.STOP:
                adas.control_stop(m, steering)
                logger.write(m, adas.state, 0.0, 0.0, steering)
                viz.update(adas.last_frame_rgb, adas.last_lane_result,
                           m, adas.state, 0.0, 0.0, steering)
                print("=" * CFG.LOG_W)
                print(f"Car stopped! | t={elapsed:.1f}с | X: {m.ego_x:+.2f} м")
                print("=" * CFG.LOG_W)
                break

            logger.write(m, adas.state, throttle, brake, steering,
                         safety_dist, comfort_dist)

            viz.update(adas.last_frame_rgb, adas.last_lane_result,
                       m, adas.state, throttle, brake, steering)

    finally:
        viz.close()
        logger.close()
        # Финальная статистика lane detection
        total = sum(adas.lane_stats.values())
        if total > 0:
            print()
            print("─" * 60)
            print("Lane detection статистика:")
            for k, v in adas.lane_stats.items():
                pct = 100 * v / total
                print(f"  {k:10s}: {v:5d} тиков ({pct:5.1f}%)")
            print("─" * 60)
        input("Press ENTER to continue...")
        bng.close()


if __name__ == "__main__":
    main()