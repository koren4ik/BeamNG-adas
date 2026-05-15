"""
visualizer.py — визуализация работы ADAS в отдельном окне.

Композитное окно 1024×600 с:
  • кадр с камеры (слева, ~640×360) с overlay полосы
  • bird's-eye view (правый верх, 240×240)
  • HUD с метриками (правый низ)
  • шкала offset внизу (полная ширина)

Использование:
    viz = Visualizer(enabled=True)
    ...
    viz.update(frame_rgb, lane_result, measurements, state, throttle, brake, steering)

В adas_v0_5.py включается флагом --visualize или константой VIS_ENABLED.

ВАЖНО: цикл обновления throttled до VIS_RATE_HZ чтобы не нагружать
основной цикл управления (который крутится на 50 Гц).
"""

import time
import cv2
import numpy as np


# ── Параметры ──
VIS_W, VIS_H = 1024, 600        # размер общего окна
CAM_W, CAM_H = 640, 360         # размер кадра камеры в окне
BEV_SIZE = 240                  # размер bird's-eye в окне
VIS_RATE_HZ = 20                # частота отрисовки (Гц)


# Цвета (BGR — OpenCV)
COLOR_BG       = (25, 25, 30)         # фон
COLOR_PANEL    = (40, 40, 45)         # фон панелей
COLOR_TEXT     = (220, 220, 220)
COLOR_LABEL    = (140, 140, 140)
COLOR_OK       = (80, 220, 80)        # зелёный
COLOR_WARN     = (40, 180, 240)       # оранжевый
COLOR_BAD      = (60, 80, 240)        # красный
COLOR_LANE     = (80, 220, 80)        # заливка полосы
COLOR_CENTER   = (255, 180, 60)       # центр полосы (голубой)
COLOR_CAR      = (60, 80, 240)        # машина (красная)


class Visualizer:
    def __init__(self, enabled: bool = True, window_name: str = "ADAS"):
        self.enabled = enabled
        self.window_name = window_name
        self._last_render_t: float = 0.0
        self._min_interval = 1.0 / VIS_RATE_HZ
        self._closed = False

        if enabled:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, VIS_W, VIS_H)

    def close(self):
        if self.enabled and not self._closed:
            cv2.destroyWindow(self.window_name)
            self._closed = True

    def update(self, frame_rgb, lane_result, m, state, throttle, brake, steering):
        """
        Главный entry point — вызывается каждый тик основного цикла.
        Внутри сама проверяет throttling и пропускает если рано.

        Параметры:
          frame_rgb     — последний кадр с камеры (H,W,3) RGB или None
          lane_result   — LaneResult из lane_detector или None
          m             — Measurements
          state         — ADASState
          throttle/brake/steering — текущие управляющие команды
        """
        if not self.enabled:
            return

        # Throttling: рисуем не чаще VIS_RATE_HZ
        now = time.time()
        if now - self._last_render_t < self._min_interval:
            return
        self._last_render_t = now

        canvas = self._render(frame_rgb, lane_result, m, state,
                              throttle, brake, steering)
        cv2.imshow(self.window_name, canvas)
        # waitKey(1) обязателен чтобы окно вообще обновлялось;
        # возвращает код клавиши если нажата
        cv2.waitKey(1)

    # ──────────────────────────────────────────────
    def _render(self, frame_rgb, lane_result, m, state,
                throttle, brake, steering) -> np.ndarray:
        canvas = np.full((VIS_H, VIS_W, 3), COLOR_BG, dtype=np.uint8)

        # Раскладка:
        #   камера   240×360+    bev (240×240)
        #                        hud (240×~340)
        #   offset bar на всю ширину, ниже камеры
        #
        # cam_h=360, vis_h=600 → offset bar высотой 60 в y=380..440
        # bev высотой 240 в y=10..250, hud в y=265..600

        self._draw_camera_panel(canvas, frame_rgb, lane_result, m,
                                x=10, y=10)
        self._draw_bev_panel(canvas, lane_result,
                             x=10 + CAM_W + 15, y=10)
        # HUD: высота = от y=265 до низа окна (всего ~325px) — много места для 7 строк
        self._draw_hud(canvas, m, state, throttle, brake, steering,
                       x=10 + CAM_W + 15, y=10 + BEV_SIZE + 15)
        # Offset bar: под камерой (y=380), на всю ширину минус правая колонка
        self._draw_offset_bar(canvas, m,
                              x=10, y=10 + CAM_H + 15,
                              w=CAM_W)

        return canvas

    # ──────────────────────────────────────────────
    def _draw_camera_panel(self, canvas, frame_rgb, lane_result, m, x, y):
        # Фон панели
        cv2.rectangle(canvas, (x-2, y-2), (x+CAM_W+2, y+CAM_H+2), COLOR_PANEL, 2)

        if frame_rgb is None:
            cv2.putText(canvas, "No camera", (x+20, y+CAM_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_LABEL, 1)
            return

        # Ресайз если нужно, OpenCV хочет BGR
        if frame_rgb.shape[1] != CAM_W or frame_rgb.shape[0] != CAM_H:
            frame_rgb = cv2.resize(frame_rgb, (CAM_W, CAM_H))
        cam_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Overlay полосы (если есть валидный детект и его полиномы)
        if (lane_result is not None and lane_result.valid
                and lane_result.left_fit is not None
                and lane_result.right_fit is not None):
            cam_bgr = self._overlay_lane(cam_bgr, lane_result)

        # Поместить на canvas
        canvas[y:y+CAM_H, x:x+CAM_W] = cam_bgr

        # Метка
        cv2.putText(canvas, "FRONT CAMERA", (x+8, y+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _overlay_lane(self, cam_bgr, lane_result):
        """Накладывает зелёную заливку полосы на оригинальный кадр.
        Полиномы и BEV → обратный perspective transform → на исходник."""
        # Для inverse transform нам нужен M_inv. Его держит детектор внутри,
        # но мы получаем здесь только результат. Самый простой путь —
        # использовать bev_img из lane_result и развернуть его обратно.
        # Но это требует matrix... сделаем проще: нарисуем overlay на самом BEV
        # и просто покажем стрелку направления на исходнике.

        # Минималистичный overlay: рисуем стрелку показывающую куда смотрит
        # центр полосы относительно машины. Плюс текст с offset.
        h, w = cam_bgr.shape[:2]
        # Стрелка от центра низа кадра в сторону центра полосы.
        # Грубая аппроксимация: горизонтальное смещение стрелки на верху
        # пропорционально offset_m (50 пикселей на 1м).
        cx = w // 2
        y_start = h - 30
        y_end = h - 130
        x_end = cx + int(-lane_result.offset_m * 80)
        # ↑ минус: машина смещена вправо → стрелка указывает влево (куда вернуть)

        cv2.arrowedLine(cam_bgr, (cx, y_start), (x_end, y_end),
                        COLOR_OK, 3, tipLength=0.2)
        return cam_bgr

    # ──────────────────────────────────────────────
    def _draw_bev_panel(self, canvas, lane_result, x, y):
        cv2.rectangle(canvas, (x-2, y-2),
                      (x+BEV_SIZE+2, y+BEV_SIZE+2), COLOR_PANEL, 2)

        if lane_result is not None and lane_result.bev_img is not None:
            bev = lane_result.bev_img
            bev_resized = cv2.resize(bev, (BEV_SIZE, BEV_SIZE))
            canvas[y:y+BEV_SIZE, x:x+BEV_SIZE] = bev_resized

            # Если есть полиномы — рисуем линии полосы на BEV
            if (lane_result.valid
                    and lane_result.left_fit is not None
                    and lane_result.right_fit is not None):
                scale = BEV_SIZE / bev.shape[0]
                ploty = np.linspace(0, bev.shape[0]-1, 30)
                lf = lane_result.left_fit
                rf = lane_result.right_fit
                lx = lf[0]*ploty**2 + lf[1]*ploty + lf[2]
                rx = rf[0]*ploty**2 + rf[1]*ploty + rf[2]
                for i in range(len(ploty)-1):
                    p1 = (int(lx[i]*scale)+x, int(ploty[i]*scale)+y)
                    p2 = (int(lx[i+1]*scale)+x, int(ploty[i+1]*scale)+y)
                    cv2.line(canvas, p1, p2, COLOR_CENTER, 2)
                    p1 = (int(rx[i]*scale)+x, int(ploty[i]*scale)+y)
                    p2 = (int(rx[i+1]*scale)+x, int(ploty[i+1]*scale)+y)
                    cv2.line(canvas, p1, p2, COLOR_CENTER, 2)
        else:
            cv2.putText(canvas, "No BEV", (x+20, y+BEV_SIZE//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LABEL, 1)

        cv2.putText(canvas, "BIRD'S-EYE", (x+8, y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ──────────────────────────────────────────────
    def _draw_hud(self, canvas, m, state, throttle, brake, steering, x, y):
        # Размер HUD-панели: от текущей y до низа окна
        w = VIS_W - x - 10
        h = VIS_H - y - 10
        cv2.rectangle(canvas, (x-2, y-2), (x+w, y+h), COLOR_PANEL, 2)

        # Заголовок
        cv2.putText(canvas, "STATUS", (x+8, y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Подготовим строки данных
        speed_kmh = m.speed * 3.6
        state_color = self._state_color(state)
        valid_str = "OK" if m.lane_valid else "NO LANE"
        valid_color = COLOR_OK if m.lane_valid else COLOR_BAD

        lines = [
            ("SPEED",   f"{speed_kmh:5.1f} km/h", COLOR_TEXT),
            ("STATE",   str(state.value),         state_color),
            ("LANE",    valid_str,                valid_color),
            ("OFFSET",  f"{m.lane_offset_m:+.2f} m" if m.lane_valid else "--",
             COLOR_TEXT),
            ("STEER",   f"{steering:+.3f}",       COLOR_TEXT),
            ("THROTTLE",f"{throttle:.2f}",        COLOR_OK if throttle > 0 else COLOR_LABEL),
            ("BRAKE",   f"{brake:.2f}",           COLOR_BAD if brake > 0 else COLOR_LABEL),
        ]

        line_y = y + 50
        for label, value, color in lines:
            cv2.putText(canvas, label, (x+12, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_LABEL, 1)
            cv2.putText(canvas, value, (x+95, line_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            line_y += 27

    def _state_color(self, state):
        s = state.value if hasattr(state, 'value') else str(state)
        if s in ('AEB', 'STOP'): return COLOR_BAD
        if s in ('CREEP', 'FOLLOW'): return COLOR_WARN
        return COLOR_OK

    # ──────────────────────────────────────────────
    def _draw_offset_bar(self, canvas, m, x, y, w):
        # Полоска отображает позицию машины в полосе:
        # центр горизонтальной линии = центр полосы.
        # Зелёная зона ±0.3м, жёлтая ±0.8м, красная дальше.
        h = 50
        cv2.rectangle(canvas, (x-2, y-2), (x+w, y+h), COLOR_PANEL, 2)
        cv2.putText(canvas, "LANE POSITION", (x+8, y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_LABEL, 1)

        # Полоса от -1.5м до +1.5м
        bar_y = y + 35
        bar_x0 = x + 20
        bar_x1 = x + w - 20
        bar_w = bar_x1 - bar_x0

        def offset_to_x(off):
            # off=-1.5 → bar_x0, off=+1.5 → bar_x1
            t = (off + 1.5) / 3.0
            t = max(0.0, min(1.0, t))
            return int(bar_x0 + t * bar_w)

        # Зоны
        x_safe_l = offset_to_x(-0.3)
        x_safe_r = offset_to_x(+0.3)
        x_warn_l = offset_to_x(-0.8)
        x_warn_r = offset_to_x(+0.8)

        # Красные края
        cv2.rectangle(canvas, (bar_x0, bar_y-6), (x_warn_l, bar_y+6), COLOR_BAD, -1)
        cv2.rectangle(canvas, (x_warn_r, bar_y-6), (bar_x1, bar_y+6), COLOR_BAD, -1)
        # Жёлтые зоны
        cv2.rectangle(canvas, (x_warn_l, bar_y-6), (x_safe_l, bar_y+6), COLOR_WARN, -1)
        cv2.rectangle(canvas, (x_safe_r, bar_y-6), (x_warn_r, bar_y+6), COLOR_WARN, -1)
        # Зелёная центральная
        cv2.rectangle(canvas, (x_safe_l, bar_y-6), (x_safe_r, bar_y+6), COLOR_OK, -1)

        # Чёрные тики на отметках
        for off in [-1.0, -0.5, 0.0, +0.5, +1.0]:
            tx = offset_to_x(off)
            cv2.line(canvas, (tx, bar_y-8), (tx, bar_y+8), (0, 0, 0), 1)
            cv2.putText(canvas, f"{off:+.1f}", (tx-12, bar_y+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_LABEL, 1)

        # Маркер машины
        if m.lane_valid:
            mx = offset_to_x(m.lane_offset_m)
            cv2.circle(canvas, (mx, bar_y), 8, COLOR_CAR, -1)
            cv2.circle(canvas, (mx, bar_y), 8, (255, 255, 255), 1)
        else:
            # Если нет валидного offset — показываем серый кружок в центре
            cv2.circle(canvas, (offset_to_x(0), bar_y), 8, COLOR_LABEL, -1)
