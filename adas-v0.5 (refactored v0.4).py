from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, Radar, Ultrasonic, State
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import csv
import math
import time

import numpy as np


# ═════════════════════════════════════════════════════════════════
#  STATE MACHINE
# ═════════════════════════════════════════════════════════════════
class ADASState(Enum):
    CRUISE = "CRUISE"
    FOLLOW = "FOLLOW"
    AEB    = "AEB"
    CREEP  = "CREEP"
    STOP   = "STOP"


# ═════════════════════════════════════════════════════════════════
#  КОНСТАНТЫ
# ═════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Config:
    # — Лидер (для тестов) —
    LEADER_BASE_SPEED:  float = 110 / 3.6
    LEADER_AMPLITUDE:   float = 20 / 3.6
    LEADER_PERIOD:      float = 10.0
    LEADER_STOP_AT:     float = 30.0

    # — Ego —
    TARGET_SPEED:       float = 130 / 3.6
    # SAFETY_HEADWAY — жёсткий минимум, ниже него регулятор тормозит резко.
    # COMFORT_HEADWAY — целевая дистанция, регулятор стремится к ней.
    # Запас между ними нужен чтобы небольшие колебания не приводили к
    # нарушениям SAFETY_HEADWAY. В промышленных ACC обычно зазор ~25%.
    SAFETY_HEADWAY:     float = 2.0   # сек — НИЖЕ ЭТОГО НЕЛЬЗЯ
    COMFORT_HEADWAY:    float = 2.5   # сек — целевая для регулятора
    MIN_SAFE_DIST:      float = 10.0
    MIN_COMFORT_DIST:   float = 12.5  # = MIN_SAFE_DIST * (COMFORT/SAFETY)

    # — TTC / переключения —
    TTC_AEB_THRESHOLD:  float = 3.0
    TTC_AEB_RELEASE:    float = 5.0
    RADAR_DETECT_DIST:  float = 90.0
    RADAR_LOST_DIST:    float = 120.0

    # — УЗ / CREEP —
    US_STOP_DIST:       float = 1.5
    US_CREEP_SPEED:     float = 5 / 3.6
    CREEP_RADAR_DIST:   float = 15.0
    CREEP_RADAR_STOP:   float = 2.0
    CREEP_BRAKE_SPEED:  float = 20 / 3.6
    US_TRIGGER_DIST:    float = 5.0   # когда УЗ начинает считать «препятствие близко»

    # — ACC (cascade): внешний контур считает target_speed из дистанции —
    #   target_v = leader_v + v_corr(distance - target_dist)
    # Гейн АСИММЕТРИЧНЫЙ: при distance < target_dist (мы ближе целевой)
    # реакция в ACC_DIST_KP_CLOSE раз сильнее. Безопасность важнее
    # быстрого догона.
    ACC_DIST_KP_OPEN:   float = 0.4   # м/с на 1м (когда мы дальше цели)
    ACC_DIST_KP_CLOSE:  float = 0.8   # м/с на 1м (когда мы ближе цели — резче!)
    ACC_DIST_CORR_MAX:  float = 8.0   # макс. коррекция к скорости лидера, м/с
    # — Внутренний PI на ошибке скорости (управляет throttle/brake) —
    ACC_SPEED_KP:       float = 0.35  # резкий газ при отставании
    ACC_SPEED_KI:       float = 0.05
    ACC_SPEED_I_MAX:    float = 0.5
    ACC_SPEED_DEADZONE: float = 0.5   # м/с — внутри этой зоны throttle=brake=0 (анти-дребезг)
    ACC_BRAKE_GAIN:     float = 0.5   # тормозим в 2 раза мягче чем газуем (комфорт)

    # — PID руля —
    # KD низкий специально: позиция ego_x квантуется сенсором ступеньками
    # ~1мм, и при KD=0.5 D-член шумел на ±0.01 руля каждый тик («змейка»).
    # При KD=0.1 D всё ещё успевает гасить колебания, но не реагирует
    # на сенсорный шум.
    STEER_KP:           float = 0.05
    STEER_KI:           float = 0.003
    STEER_KD:           float = 0.1
    STEER_I_MAX:        float = 0.2
    # Dead-zone: при |ego_x| меньше этого значения руль не дёргается.
    # Гасит лимит-цикл вокруг setpoint=0.
    STEER_DEADZONE:     float = 0.02   # м (2 см)
    # — Сглаживание выхода руля —
    # Slew rate: максимальная скорость изменения руля в единицах руля в секунду.
    # 1.0 значит «от полного левого до полного правого за 2 секунды».
    # Значения 0.5..1.5 типичны для комфортного автопилота.
    STEER_SLEW_RATE:    float = 0.8
    # Low-pass filter на выход руля. tau = постоянная времени в секундах.
    # alpha = dt / (tau + dt). tau=0 отключает фильтр.
    # Чем больше tau, тем плавнее, но и медленнее реакция.
    # 0.15..0.30 — разумный диапазон.
    STEER_LPF_TAU:      float = 0.2

    # — Throttle/Brake mapping для скоростных регуляторов CRUISE —
    SPEED_KP:           float = 0.1   # коэф пропорционала throttle/brake = err * KP

    # — CREEP throttle PI —
    CREEP_KP:           float = 0.25
    CREEP_KI:           float = 0.1
    CREEP_DT_GAIN:      float = 0.05  # коэф интегрирования (как в v0.4: err * 0.05)
    CREEP_I_MAX:        float = 1.0
    CREEP_THROTTLE_MAX: float = 0.25
    CREEP_BRAKE_MAX:    float = 0.5

    # — Фильтры —
    RADAR_FILTER_SIZE:  int   = 5
    US_FILTER_SIZE:     int   = 5

    # — AEB —
    AEB_CONFIRM_TICKS:  int   = 3

    # — Цикл —
    LOOP_DT:            float = 0.05
    DT_MIN:             float = 0.01  # клипуем dt чтобы не делить на ноль
    DT_MAX:             float = 0.2

    # — Радар —
    RADAR_NO_TARGET:    float = 9999.0  # сторожевое значение «нет цели»

    # — Лог —
    LOG_W:              int   = 72
    CSV_PATH:           str   = "adas_log.csv"

CFG = Config()


# ═════════════════════════════════════════════════════════════════
#  МЕДИАННЫЙ ФИЛЬТР
# ═════════════════════════════════════════════════════════════════
class MedianFilter:
    def __init__(self, size: int):
        self.buf = deque(maxlen=size)

    def update(self, value: float) -> float:
        self.buf.append(value)
        return float(np.median(self.buf))

    def reset(self):
        self.buf.clear()


# ═════════════════════════════════════════════════════════════════
#  PID КОНТРОЛЛЕР с anti-windup (clamping)
# ═════════════════════════════════════════════════════════════════
class PIDController:
    """
    Универсальный PID. Anti-windup методом clamping:
    интегратор не накапливается, если выход уже в насыщении,
    а ошибка толкает его ещё дальше в то же насыщение.

    Особенности:
      • На первом вызове D=0 (нет предыдущей ошибки → нет производной).
      • prev_error сохраняется между вызовами; reset() обнуляет состояние.
      • i_max ограничивает абсолютное значение интеграла (доп. защита от windup).
    """

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


# ═════════════════════════════════════════════════════════════════
#  ИЗМЕРЕНИЯ
# ═════════════════════════════════════════════════════════════════
@dataclass
class Measurements:
    """Снимок сенсоров на текущем тике."""
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


# ═════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ
# ═════════════════════════════════════════════════════════════════
def compute_ttc(distance: float, doppler: float) -> float:
    """TTC = distance / closing_speed. inf если не сближаемся."""
    if doppler > 0.1:
        return distance / doppler
    return float('inf')


def estimate_leader_speed(ego_speed: float, doppler: float) -> float:
    """
    Доплер положителен при сближении (ego быстрее лидера).
    leader_v = ego_v - doppler — корректно для обеих сторон знака.
    """
    return max(0.0, ego_speed - doppler)


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


# ═════════════════════════════════════════════════════════════════
#  STATE MACHINE — чистая функция (тестируемая без BeamNG)
# ═════════════════════════════════════════════════════════════════
def next_state(current: ADASState, m: Measurements,
               aeb_ticks: int) -> tuple[ADASState, int]:
    """
    Возвращает (новое_состояние, новый_счётчик_aeb_ticks).
    Логика идентична v0.4, но с фиксом [BUG-4]: aeb_ticks сбрасывается
    в 0 на любом тике, где условие AEB не выполнено.
    """
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
        # AEB-инкремент только пока ttc низкий; иначе — сброс [BUG-4]
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
        # Выход CREEP→FOLLOW: УЗ свободен и радар видит цель далеко
        if m.us_min >= CFG.US_TRIGGER_DIST and m.has_target and m.distance > 10.0:
            return ADASState.FOLLOW, 0
        # Если радар вообще цель потерял и УЗ свободен — в CRUISE
        if m.us_min >= CFG.US_TRIGGER_DIST and not m.has_target:
            return ADASState.CRUISE, 0
        return ADASState.CREEP, 0

    return current, aeb_ticks


# ═════════════════════════════════════════════════════════════════
#  ADAS CONTROLLER — инкапсулирует всё состояние машины
# ═════════════════════════════════════════════════════════════════
class ADASController:
    def __init__(self, vehicle, sensors_bundle):
        self.vehicle = vehicle
        self.electrics, self.pos_sensor, self.radar, self.us_front, self.us_rear = sensors_bundle

        # — Регуляторы —
        self.steer_pid = PIDController(
            CFG.STEER_KP, CFG.STEER_KI, CFG.STEER_KD,
            out_min=-1.0, out_max=1.0, i_max=CFG.STEER_I_MAX,
        )
        # ACC: cascade. Внешний контур (target_speed из дистанции) — без памяти,
        # чистая алгебра. Внутренний — PI на ошибке скорости.
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

        # — Фильтры —
        self.radar_dist_filter = MedianFilter(CFG.RADAR_FILTER_SIZE)
        self.radar_doppler_filter = MedianFilter(CFG.RADAR_FILTER_SIZE)
        self.us_front_filter = MedianFilter(CFG.US_FILTER_SIZE)
        self.us_rear_filter = MedianFilter(CFG.US_FILTER_SIZE)

        # — Состояние —
        self.state: ADASState = ADASState.CRUISE
        self.aeb_ticks: int = 0
        self.t_prev: float = time.time()
        # Предыдущее значение руля для slew rate limiter и LPF
        self.prev_steering: float = 0.0

    # ───────── СБОР ИЗМЕРЕНИЙ ─────────
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

        # Радар — отделяем «нет цели» от «цель далеко» [BUG-радарный мерцающий]
        radar_data = self.radar.poll()
        if radar_data is not None and radar_data.size > 0:
            closest = radar_data[radar_data[:, 0].argmin()]
            raw_dist = float(closest[0])
            raw_doppler = float(closest[1])
            m.distance = self.radar_dist_filter.update(raw_dist)
            m.doppler = self.radar_doppler_filter.update(raw_doppler)
            m.has_target = True
        else:
            # Цели нет — фильтр НЕ подкармливаем мусором. Просто отдаём
            # большое расстояние и нулевой доплер, has_target=False.
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
        return m

    # ───────── ПЕРЕХОД СОСТОЯНИЙ + ПРОБРОС reset() РЕГУЛЯТОРОВ ─────────
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

    # ───────── РУЛЬ (PID + dead-zone + LPF + slew rate) ─────────
    def compute_steering(self, m: Measurements) -> float:
        """
        Многоступенчатое сглаживание выхода руля:
          1. PID считает «сырое» желаемое значение.
          2. Low-pass filter (1-го порядка): out = α·raw + (1-α)·prev,
             где α = dt / (τ + dt). Сглаживает высокочастотные изменения.
          3. Slew rate limiter: ограничивает |Δsteering/dt|.
             Жёсткий потолок на скорость поворота руля.

        LPF и slew rate — комплементарные:
          • LPF мягко гасит ВСЕ изменения, но больше всего быстрые.
          • Slew rate — жёсткий потолок «быстрее этого нельзя».
        Вместе они дают плавное и предсказуемое движение руля.
        """
        # ВАЖНО про знак: в этой конфигурации BeamNG/сцены положительный
        # steering двигает машину в ту сторону, в которую растёт ego_x —
        # использование `error = ego_x` (без минуса) работает как
        # отрицательная обратная связь.
        error = m.ego_x

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

        # 3. Slew rate limiter
        max_delta = CFG.STEER_SLEW_RATE * m.dt
        delta = filtered - self.prev_steering
        if delta > max_delta:
            filtered = self.prev_steering + max_delta
        elif delta < -max_delta:
            filtered = self.prev_steering - max_delta

        # Финальный clip и сохранение
        filtered = max(-1.0, min(1.0, filtered))
        self.prev_steering = filtered
        return filtered

    # ───────── CRUISE ─────────
    def control_cruise(self, m: Measurements, steering: float):
        err = CFG.TARGET_SPEED - m.speed
        throttle = max(0.0, min(1.0, err * CFG.SPEED_KP))
        brake    = max(0.0, min(1.0, -err * CFG.SPEED_KP))
        self.vehicle.control(throttle=throttle, brake=brake,
                             parkingbrake=0, steering=steering)
        return throttle, brake

    # ───────── FOLLOW (cascade ACC: дистанция → target_speed → throttle/brake) ─────────
    def control_follow(self, m: Measurements, steering: float):
        """
        Cascade-схема:
          1) Внешний контур (без памяти): из дистанции и скорости лидера
             вычисляем target_speed.
          2) Внутренний PI: из ошибки скорости (target − ego) вычисляем
             комбинированную команду.
          3) Маппинг в throttle/brake с dead-zone (анти-дребезг) и
             асимметричным усилением тормоза (комфорт).

        Принципиально:
          • Если лидер удаляется (doppler<0) — никогда не тормозим.
            Открытие дистанции не повод глушить ускорение.
          • target_speed клипуется в [0, TARGET_SPEED] — не превышаем
            крейсерскую даже если лидер очень далеко.
          • PI на ошибке скорости устойчив к ступенькам в дистанции
            (медианный фильтр квантует distance), потому что они
            влияют на target_speed мягко через ACC_DIST_KP.
        """
        if not m.leader_stopped:
            # ── Внешний контур: target_speed ──
            leader_v = estimate_leader_speed(m.speed, m.doppler)

            # Две дистанции: safety (минимум) и comfort (target регулятора)
            safety_dist  = max(CFG.MIN_SAFE_DIST,    leader_v * CFG.SAFETY_HEADWAY)
            comfort_dist = max(CFG.MIN_COMFORT_DIST, leader_v * CFG.COMFORT_HEADWAY)

            # Ошибка относительно COMFORT (а не safety!) — стремимся
            # к комфортной дистанции, оставляя запас до safety.
            dist_error = m.distance - comfort_dist

            # Асимметричный гейн: ниже целевой реагируем резче
            kp = CFG.ACC_DIST_KP_OPEN if dist_error >= 0 else CFG.ACC_DIST_KP_CLOSE
            v_corr = max(-CFG.ACC_DIST_CORR_MAX,
                         min(CFG.ACC_DIST_CORR_MAX, dist_error * kp))

            target_v = leader_v + v_corr
            target_v = max(0.0, min(CFG.TARGET_SPEED, target_v))

            # ── Внутренний контур: PI на ошибке скорости ──
            speed_err = target_v - m.speed
            cmd = self.acc_speed_pid.update(speed_err, m.dt)

            # ── Маппинг в throttle/brake ──
            # ВАЖНО: проверка на нарушение SAFETY имеет приоритет.
            # Если зашли ниже safety_dist — гарантированно тормозим,
            # независимо от того, что говорит PI скорости.
            if m.distance < safety_dist and m.doppler > -0.5:
                # Сближаемся или почти не отдаляемся, и ниже safety —
                # экстренный комфортный тормоз (жёсткий AEB сработает
                # отдельно через state-машину при низком TTC)
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
            safe_dist = comfort_dist  # для backward compat в логе

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

    # ───────── AEB ─────────
    def control_aeb(self, m: Measurements, steering: float):
        self.vehicle.control(throttle=0, brake=1.0, parkingbrake=0, steering=steering)
        return 0.0, 1.0

    # ───────── CREEP ─────────
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

    # ───────── STOP ─────────
    def control_stop(self, m: Measurements, steering: float):
        self.vehicle.control(throttle=0, brake=0, parkingbrake=1.0, steering=steering)


# ═════════════════════════════════════════════════════════════════
#  ЛИДЕР (PID руля + простой пропорциональный спид-контроллер)
# ═════════════════════════════════════════════════════════════════
class LeaderDriver:
    """Управление эталонным лидером для тестов."""
    def __init__(self, leader_vehicle, leader_electrics, leader_pos_sensor):
        self.veh = leader_vehicle
        self.elec = leader_electrics
        self.pos = leader_pos_sensor
        self.steer_pid = PIDController(
            CFG.STEER_KP, CFG.STEER_KI, CFG.STEER_KD,
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


# ═════════════════════════════════════════════════════════════════
#  CSV-ЛОГГЕР
# ═════════════════════════════════════════════════════════════════
class CsvLogger:
    HEADER = ['t', 'state', 'speed_kmh', 'ego_x', 'distance', 'has_target',
              'doppler', 'ttc', 'us_min', 'throttle', 'brake', 'steering',
              'safety_dist', 'comfort_dist']

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
        ])
        self.f.flush()

    def close(self):
        self.f.close()


# ═════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    bng = BeamNGpy('localhost', 64256, home=r'D:\Scripts\BeamNG.tech.v0.38.5.0')
    bng.open(launch=True)

    scenario = Scenario('tech_ground', 'adas_v05')
    vehicle = Vehicle('ego',    model='etk800', license='ADAS-V05')
    leader  = Vehicle('leader', model='etk800', license='LEAD')
    scenario.add_vehicle(vehicle, pos=(0,   0, 0), cling=True)
    scenario.add_vehicle(leader,  pos=(0, -40, 0), cling=True)
    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    time.sleep(1)

    # — Сенсоры ego —
    electrics = Electrics()
    pos_sensor = State()
    vehicle.attach_sensor('electrics', electrics)
    vehicle.attach_sensor('pos_sensor', pos_sensor)

    radar = Radar('radar', bng, vehicle,
                  pos=(0, -2.3, 0.8), dir=(0, -1, 0.15),
                  near_far_planes=(0.5, 120.0), is_visualised=True)
    us_front = Ultrasonic('us_front', bng, vehicle,
                          pos=(0, -2.35, 0.5), dir=(0, -1, 0),
                          near_far_planes=(0.1, 6.0), is_visualised=True)
    us_rear = Ultrasonic('us_rear', bng, vehicle,
                         pos=(0, 2.35, 0.5), dir=(0, 1, 0),
                         near_far_planes=(0.1, 6.0), is_visualised=True)

    # — Сенсоры лидера —
    leader_electrics = Electrics()
    leader_pos_sensor = State()
    leader.attach_sensor('leader_electrics', leader_electrics)
    leader.attach_sensor('leader_pos_sensor', leader_pos_sensor)

    # — Контроллеры —
    adas = ADASController(vehicle, (electrics, pos_sensor, radar, us_front, us_rear))
    leader_drv = LeaderDriver(leader, leader_electrics, leader_pos_sensor)
    logger = CsvLogger(CFG.CSV_PATH)

    print(" ADAS v0.5 | BeamNG.tech")
    print(f" Start STATE: {adas.state.value}")
    print(f" Target speed: {CFG.TARGET_SPEED*3.6:.0f} км/ч")
    print(f" Headway: safety={CFG.SAFETY_HEADWAY:.1f}с, comfort={CFG.COMFORT_HEADWAY:.1f}с")
    print(f" CSV log: {CFG.CSV_PATH}")

    start_time = time.time()
    next_tick = start_time

    try:
        while True:
            # — Стабильный шаг цикла: компенсируем длительность итерации —
            now = time.time()
            sleep_for = next_tick - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            next_tick += CFG.LOOP_DT

            t_now = time.time()
            elapsed = t_now - start_time

            # — Лидер —
            leader_stopped = leader_drv.step(t_now, elapsed)

            # — Ego —
            vehicle.poll_sensors()
            m = adas.measure(t_now, leader_stopped)

            # — Переход состояний —
            adas.transition(m)

            # — Управление в зависимости от состояния —
            steering = adas.compute_steering(m)
            throttle, brake = 0.0, 0.0
            safety_dist, comfort_dist = 0.0, 0.0  # для CSV — заполняются только в FOLLOW

            if adas.state == ADASState.CRUISE:
                throttle, brake = adas.control_cruise(m, steering)
                print(fmt_log("CRUISE", [
                    ("speed",  f"{m.speed*3.6:.1f} км/ч"),
                    ("target", f"{CFG.TARGET_SPEED*3.6:.0f} км/ч"),
                    ("0x",     f"{m.ego_x:+.2f} м"),
                    ("gear",   gear_str(electrics.data)),
                ]))

            elif adas.state == ADASState.FOLLOW:
                throttle, brake, safe_dist, safety_dist, comfort_dist = \
                    adas.control_follow(m, steering)
                # Помечаем нарушение SAFETY чтобы было видно в консоли
                safety_marker = "  ⚠SAFETY" if m.distance < safety_dist else ""
                print(fmt_log("FOLLOW", [
                    ("speed",     f"{m.speed*3.6:.1f} км/ч"),
                    ("dist",      f"{m.distance:.1f} м{safety_marker}"),
                    ("safety",    f"{safety_dist:.1f} м"),
                    ("comfort",   f"{comfort_dist:.1f} м"),
                    ("ttc",       f"{m.ttc:.1f} с" if m.ttc != float('inf') else " inf с"),
                    ("doppler_s", f"{m.doppler*3.6:+.1f} км/ч"),
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
                print("=" * CFG.LOG_W)
                print(f"Car stopped! | t={elapsed:.1f}с | X: {m.ego_x:+.2f} м")
                print("=" * CFG.LOG_W)
                break

            logger.write(m, adas.state, throttle, brake, steering,
                         safety_dist, comfort_dist)

    finally:
        logger.close()
        input("Press ENTER to continue...")
        bng.close()


if __name__ == "__main__":
    main()