"""
lane_detection.py — обнаружение полосы на кадре с фронтальной камеры.

Pipeline:
  1. ROI crop     — нижние ~55% кадра
  2. Bird's-eye   — perspective transform к виду сверху
  3. Бинаризация  — выделение пикселей разметки (HLS white+yellow)
  4. Sliding window или prior-based search — группировка пикселей по линиям
  5. Polynomial fit — аппроксимация x = a·y² + b·y + c для каждой линии
  6. Sanity check — проверка адекватности результата
  7. Offset       — смещение машины от центра полосы в метрах

Класс LaneDetector хранит prior (полиномы прошлого кадра) для inertia —
это сильно ускоряет и стабилизирует детекцию.

ИНТЕГРАЦИЯ В adas_v0_5.py:
  detector = LaneDetector()
  ...
  result = detector.detect(camera_frame_rgb)
  if result.valid:
      error = result.offset_m  # в compute_steering вместо m.ego_x
"""

from dataclasses import dataclass
import cv2
import numpy as np


# ═════════════════════════════════════════════════════════════════
#  КАЛИБРОВКА — ЭТИ ЧИСЛА ТЫ КРУТИШЬ ПОД РЕАЛЬНЫЕ КАДРЫ
# ═════════════════════════════════════════════════════════════════
# Все координаты задаются в долях кадра (0.0..1.0), чтобы не зависеть
# от резолюции. Под (W, H) умножатся внутри.

# ROI: где начинается «дорога». Всё выше — небо, обрезаем.
# ROI_BOTTOM меньше 1.0 — обрезаем капот ego-машины.
ROI_TOP_FRAC    = 0.45      # верхняя граница ROI (по высоте кадра)
ROI_BOTTOM_FRAC = 0.95      # нижняя граница ROI — отсекаем капот

# Perspective transform: 4 точки трапеции в системе ROI (после crop).
# Калибровано вручную под езду по правой полосе:
#   - левая граница = центральный пунктир (разделитель встречек)
#   - правая граница = правая обочина (белая сплошная)
# Если на твоих кадрах геометрия немного другая — крути в первую очередь
# верхние точки (нижние стабильнее).
# Порядок: верх-лево, верх-право, низ-право, низ-лево.
SRC_POINTS_FRAC = [
    (0.40, 0.10),   # верх-лево
    (0.64, 0.10),   # верх-право
    (1, 0.65),  # низ-право
    (0.05, 0.65),   # низ-лево
]

# Bird's-eye output. Чем больше — тем точнее, но медленнее.
BEV_W, BEV_H = 400, 400

# Бинаризация: HLS пороги для белой и жёлтой разметки.
# (H, L, S) min/max. Lightness — основной канал для белого.
WHITE_HLS_MIN = (0,   200, 0)
WHITE_HLS_MAX = (255, 255, 255)
YELLOW_HLS_MIN = (15,  30, 115)
YELLOW_HLS_MAX = (35, 204, 255)

# Sliding window
N_WINDOWS  = 9
WIN_MARGIN = 50    # полуширина окна, пикселей
MIN_PIX    = 50    # минимум пикселей чтобы пересчитать центр окна

# Prior-based search: шире окна, но только в полосе вокруг прошлого полинома
PRIOR_MARGIN = 60

# Confidence линии: считаем линию НАДЁЖНОЙ если у неё много пикселей
# и они разнесены по высоте BEV (не один штрих в одном месте).
# Используется для логики fallback: если одна линия надёжная а другая нет,
# слабую перестраиваем относительно сильной + сохранённой ширины полосы.
# ВАЖНО: пунктир даёт меньше пикселей чем сплошная — пороги поставлены
# с учётом этого.
LINE_MIN_PIXELS         = 100    # мин. пикселей для "надёжной" линии
LINE_MIN_Y_COVERAGE     = 0.5    # мин. покрытие по Y (доля высоты BEV)
LINE_STRENGTH_RATIO     = 4.0    # одна линия в N раз сильнее другой → fallback

# Parallelism check: реальные полосы геометрически почти параллельны.
# Если коэффициенты при y² у двух полиномов сильно расходятся — одна из
# линий построена на шуме (типичный случай: пунктир потерян, polyfit
# изогнул кривую через случайные пиксели).
# Допустимая разница в коэффициенте 'a' между левой и правой линиями.
PARALLELISM_A_DIFF_MAX  = 0.0008

# Sanity check
LANE_WIDTH_M     = 3.5    # реальная ширина полосы (BeamNG обычно ~3.5м)
LANE_WIDTH_MIN_M = 2.5    # ниже — отказ
LANE_WIDTH_MAX_M = 5.0    # выше — отказ
MAX_OFFSET_JUMP_M = 0.5   # макс. изменение offset за 1 кадр

# Сглаживание offset_m через LPF (как в руле)
OFFSET_LPF_TAU = 0.15  # сек


# ═════════════════════════════════════════════════════════════════
#  РЕЗУЛЬТАТ ДЕТЕКЦИИ
# ═════════════════════════════════════════════════════════════════
@dataclass
class LaneResult:
    valid: bool                    # удалось ли найти полосу
    offset_m: float = 0.0          # смещение машины от центра полосы [м]
    lane_width_m: float = 0.0      # ширина полосы [м]
    left_fit: np.ndarray = None    # коэффициенты [a, b, c] левого полинома
    right_fit: np.ndarray = None   # коэффициенты правого полинома
    debug_img: np.ndarray = None   # картинка с overlay для отладки
    reason: str = ""               # если invalid — почему
    # Промежуточные картинки для отладки (заполняются всегда, даже при отказе,
    # чтобы можно было увидеть на каком шаге сломалось)
    bev_img:    np.ndarray = None  # bird's-eye view (BGR)
    binary_img: np.ndarray = None  # бинаризация (uint8 mask)


# ═════════════════════════════════════════════════════════════════
#  ОСНОВНОЙ КЛАСС
# ═════════════════════════════════════════════════════════════════
class LaneDetector:
    def __init__(self, debug: bool = False):
        self.debug = debug

        # Prior — полиномы с прошлого успешного кадра
        self.prev_left_fit:  np.ndarray = None
        self.prev_right_fit: np.ndarray = None
        self.prev_offset_m:  float = 0.0

        # Сохранённая ширина полосы в пикселях с последнего ВЫСОКОДОВЕРНОГО кадра
        # (где обе линии были надёжно найдены). Используется для fallback
        # когда одна из линий теряется.
        self.lane_width_px_prior: float | None = None

        # Кэш матриц perspective transform — пересчитывается при первом detect
        self._M = None
        self._M_inv = None
        self._cached_size = None

    # ─────────────────────────────────────────────────────────
    def reset(self):
        """Сброс prior. Вызывать при потере полосы или смене сцены."""
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.prev_offset_m = 0.0
        # lane_width_px_prior НЕ сбрасываем — он живёт дольше и помогает
        # восстановиться. Сброс его — только если очень надо извне.

    # ─────────────────────────────────────────────────────────
    def _build_transform(self, frame_w: int, frame_h: int):
        """Один раз посчитать матрицы perspective transform для данного размера."""
        if self._cached_size == (frame_w, frame_h):
            return

        roi_top = int(frame_h * ROI_TOP_FRAC)
        roi_bot = int(frame_h * ROI_BOTTOM_FRAC)
        roi_h = roi_bot - roi_top
        roi_w = frame_w

        src = np.float32([
            (x_frac * roi_w, y_frac * roi_h)
            for x_frac, y_frac in SRC_POINTS_FRAC
        ])
        dst = np.float32([
            (0, 0), (BEV_W, 0), (BEV_W, BEV_H), (0, BEV_H),
        ])
        self._M     = cv2.getPerspectiveTransform(src, dst)
        self._M_inv = cv2.getPerspectiveTransform(dst, src)
        self._roi_top = roi_top
        self._roi_bot = roi_bot
        self._cached_size = (frame_w, frame_h)

    # ─────────────────────────────────────────────────────────
    def _binarize(self, bev_bgr: np.ndarray) -> np.ndarray:
        """HLS-бинаризация: выделяем белую и жёлтую разметку."""
        hls = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HLS)
        white  = cv2.inRange(hls, WHITE_HLS_MIN,  WHITE_HLS_MAX)
        yellow = cv2.inRange(hls, YELLOW_HLS_MIN, YELLOW_HLS_MAX)
        binary = cv2.bitwise_or(white, yellow)
        # Морфология: закрываем мелкие дыры, убираем шум
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    # ─────────────────────────────────────────────────────────
    def _sliding_window(self, binary: np.ndarray):
        """Sliding window от низа к верху. Возвращает (lx, ly, rx, ry)."""
        # Гистограмма нижней половины — стартовые X
        hist = np.sum(binary[binary.shape[0]//2:, :], axis=0)
        midpoint = len(hist) // 2
        left_cur  = int(np.argmax(hist[:midpoint]))
        right_cur = int(np.argmax(hist[midpoint:])) + midpoint

        win_h = binary.shape[0] // N_WINDOWS
        nonzero = binary.nonzero()
        ny, nx = np.array(nonzero[0]), np.array(nonzero[1])

        left_idx_all, right_idx_all = [], []

        for w in range(N_WINDOWS):
            y_low  = binary.shape[0] - (w+1) * win_h
            y_high = binary.shape[0] - w * win_h

            xl_low,  xl_high = left_cur  - WIN_MARGIN, left_cur  + WIN_MARGIN
            xr_low,  xr_high = right_cur - WIN_MARGIN, right_cur + WIN_MARGIN

            in_left = ((ny >= y_low) & (ny < y_high) &
                       (nx >= xl_low) & (nx < xl_high)).nonzero()[0]
            in_right = ((ny >= y_low) & (ny < y_high) &
                        (nx >= xr_low) & (nx < xr_high)).nonzero()[0]

            left_idx_all.append(in_left)
            right_idx_all.append(in_right)

            if len(in_left) > MIN_PIX:
                left_cur = int(np.mean(nx[in_left]))
            if len(in_right) > MIN_PIX:
                right_cur = int(np.mean(nx[in_right]))

        left_idx_all  = np.concatenate(left_idx_all)
        right_idx_all = np.concatenate(right_idx_all)
        return nx[left_idx_all], ny[left_idx_all], \
               nx[right_idx_all], ny[right_idx_all]

    # ─────────────────────────────────────────────────────────
    def _prior_search(self, binary: np.ndarray):
        """
        Если есть полиномы с прошлого кадра — ищем пиксели в полосе вокруг них.
        Быстрее sliding window и стабильнее.
        """
        nonzero = binary.nonzero()
        ny, nx = np.array(nonzero[0]), np.array(nonzero[1])

        lf, rf = self.prev_left_fit, self.prev_right_fit
        left_curve  = lf[0]*ny**2 + lf[1]*ny + lf[2]
        right_curve = rf[0]*ny**2 + rf[1]*ny + rf[2]

        in_left  = (np.abs(nx - left_curve)  < PRIOR_MARGIN).nonzero()[0]
        in_right = (np.abs(nx - right_curve) < PRIOR_MARGIN).nonzero()[0]

        return nx[in_left], ny[in_left], nx[in_right], ny[in_right]

    # ─────────────────────────────────────────────────────────
    def _line_strength(self, ly: np.ndarray) -> dict:
        """
        Оценивает 'силу' линии по её Y-координатам пикселей.
        Сильная линия = много пикселей + они разнесены по высоте BEV
        (не один штрих в одном месте).
        """
        n_pix = len(ly)
        if n_pix == 0:
            return {'n_pix': 0, 'y_coverage': 0.0, 'is_strong': False}

        y_coverage = (ly.max() - ly.min()) / BEV_H
        is_strong = (n_pix >= LINE_MIN_PIXELS and
                     y_coverage >= LINE_MIN_Y_COVERAGE)
        return {'n_pix': n_pix, 'y_coverage': y_coverage, 'is_strong': is_strong}

    # ─────────────────────────────────────────────────────────
    def _draw_debug(self, bev_bgr, binary, left_fit, right_fit,
                    lane_center_px, car_pos_px):
        """Накладывает найденную полосу на bird's-eye для отладки."""
        ploty = np.linspace(0, BEV_H-1, BEV_H)
        lf = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        rf = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        overlay = np.zeros_like(bev_bgr)
        pts_l = np.array([np.transpose(np.vstack([lf, ploty]))])
        pts_r = np.array([np.flipud(np.transpose(np.vstack([rf, ploty])))])
        pts = np.hstack((pts_l, pts_r))
        cv2.fillPoly(overlay, np.int32([pts]), (0, 200, 0))

        out = cv2.addWeighted(bev_bgr, 1.0, overlay, 0.4, 0)

        # Линия центра полосы (синяя) и центр кадра (красная)
        cv2.line(out, (int(lane_center_px), 0), (int(lane_center_px), BEV_H),
                 (255, 100, 0), 2)
        cv2.line(out, (int(car_pos_px), 0), (int(car_pos_px), BEV_H),
                 (0, 0, 255), 2)
        return out

    # ─────────────────────────────────────────────────────────
    def detect(self, frame_rgb: np.ndarray) -> LaneResult:
        """
        Главный entry point. Принимает кадр RGB, возвращает LaneResult.
        """
        # OpenCV любит BGR
        if frame_rgb.ndim != 3:
            return LaneResult(valid=False, reason="frame is not 3-channel")
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        H, W = frame_bgr.shape[:2]
        self._build_transform(W, H)

        # 1. ROI
        roi = frame_bgr[self._roi_top:self._roi_bot, :]

        # 2. Bird's-eye
        bev = cv2.warpPerspective(roi, self._M, (BEV_W, BEV_H))

        # 3. Бинаризация
        binary = self._binarize(bev)

        # Сохраняем для возможной диагностики (любой возврат добавит их)
        self._last_bev = bev
        self._last_binary = binary

        def _invalid(reason: str) -> LaneResult:
            """Helper для отказов: автоматически приклеивает bev и binary."""
            return LaneResult(valid=False, reason=reason,
                              bev_img=bev, binary_img=binary)

        # 4. Поиск пикселей: prior, если есть, иначе sliding window
        if self.prev_left_fit is not None and self.prev_right_fit is not None:
            lx, ly, rx, ry = self._prior_search(binary)
            # Если prior не нашёл достаточно — fallback на sliding window
            if len(lx) < 100 or len(rx) < 100:
                lx, ly, rx, ry = self._sliding_window(binary)
        else:
            lx, ly, rx, ry = self._sliding_window(binary)

        # Sanity: вообще ничего не нашли — отказ
        if len(lx) < 50 and len(rx) < 50:
            self.reset()
            return _invalid(f"both lines empty: L={len(lx)} R={len(rx)}")

        # Если ОДНА линия совсем пустая, но fallback недоступен — тоже отказ
        if (len(lx) < 50 or len(rx) < 50) and self.lane_width_px_prior is None:
            self.reset()
            return _invalid(f"one line empty, no width prior: L={len(lx)} R={len(rx)}")

        # 5. Оцениваем СИЛУ каждой линии до построения полиномов
        left_str  = self._line_strength(ly)
        right_str = self._line_strength(ry)

        # ── СТРАТЕГИЯ ──
        # Хотим избежать ситуации: одна линия найдена надёжно (много пикселей,
        # широкое покрытие по Y), другая — плохо (несколько пикселей в одном
        # штрихе). Полином по слабой линии будет выдумкой, и offset поедет.
        #
        # Решение: если одна линия СИЛЬНО надёжнее другой (ratio пикселей >
        # LINE_STRENGTH_RATIO), и при этом у нас есть сохранённая ширина полосы
        # с прошлых кадров — слабую линию ВОССТАНАВЛИВАЕМ как сдвиг сильной
        # на эту ширину.
        used_fallback = None    # 'left_from_right' / 'right_from_left' / None

        n_l, n_r = left_str['n_pix'], right_str['n_pix']

        # "Правая сильно лучше" — это когда либо левая пустая, либо правая в N раз
        # больше левой. Аналогично для "левая сильно лучше".
        much_more_right = (n_l == 0) or (n_r / max(n_l, 1) > LINE_STRENGTH_RATIO)
        much_more_left  = (n_r == 0) or (n_l / max(n_r, 1) > LINE_STRENGTH_RATIO)

        # Условие fallback: сильный дисбаланс + есть сохранённая ширина полосы
        can_fallback = self.lane_width_px_prior is not None

        if can_fallback and (much_more_right and right_str['is_strong']):
            # Доверяем правой, левую перестраиваем
            try:
                right_fit = np.polyfit(ry, rx, 2)
            except (np.linalg.LinAlgError, TypeError) as e:
                return _invalid(f"polyfit right failed: {e}")
            # left_fit = right_fit, сдвинутый на -lane_width_px по X
            left_fit = right_fit.copy()
            left_fit[2] -= self.lane_width_px_prior
            used_fallback = 'left_from_right'

        elif can_fallback and (much_more_left and left_str['is_strong']):
            # Доверяем левой, правую перестраиваем
            try:
                left_fit = np.polyfit(ly, lx, 2)
            except (np.linalg.LinAlgError, TypeError) as e:
                return _invalid(f"polyfit left failed: {e}")
            right_fit = left_fit.copy()
            right_fit[2] += self.lane_width_px_prior
            used_fallback = 'right_from_left'

        else:
            # Стандартный путь: обе линии достаточно надёжны (или нет
            # сохранённой ширины — первый запуск)
            try:
                left_fit  = np.polyfit(ly, lx, 2)
                right_fit = np.polyfit(ry, rx, 2)
            except (np.linalg.LinAlgError, TypeError) as e:
                return _invalid(f"polyfit failed: {e}")

            # Sanity: ОЧЕНЬ сильный дисбаланс БЕЗ fallback — отказ.
            min_n = min(n_l, n_r)
            if min_n < 20:
                self.reset()
                return _invalid(f"weak line no fallback: L={n_l} R={n_r}")

        # 5b. Parallelism check: реальные полосы параллельны, у них близкая
        # кривизна. Если коэффициенты при y² (старший член параболы) сильно
        # расходятся — одна из линий построена на шуме (типичный случай:
        # пунктир потерян, polyfit изогнул дугу через случайные пиксели).
        # Это работает только в стандартной ветке (не fallback) — в fallback
        # коэффициенты a-b-c слепляются по построению, проверка избыточна.
        if used_fallback is None:
            a_diff = abs(left_fit[0] - right_fit[0])
            if a_diff > PARALLELISM_A_DIFF_MAX:
                # Линии не параллельны. Если есть prior — пробуем fallback,
                # выбирая ту линию у которой больше пикселей как "достоверную".
                if self.lane_width_px_prior is not None:
                    if n_r >= n_l:
                        # Доверяем правой
                        left_fit = right_fit.copy()
                        left_fit[2] -= self.lane_width_px_prior
                        used_fallback = 'left_from_right_parallel'
                    else:
                        right_fit = left_fit.copy()
                        right_fit[2] += self.lane_width_px_prior
                        used_fallback = 'right_from_left_parallel'
                else:
                    # Нет prior — отказ
                    self.reset()
                    return _invalid(f"non-parallel lines, no prior: a_diff={a_diff:.5f}")

        # 6. Считаем offset на уровне переднего бампера (низ BEV)
        y_eval = BEV_H - 1
        lx_at = left_fit[0]*y_eval**2  + left_fit[1]*y_eval  + left_fit[2]
        rx_at = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]

        lane_width_px = rx_at - lx_at
        if lane_width_px <= 0:
            self.reset()
            return _invalid("lines crossed (left right of right)")

        # Sanity ширины (как раньше)
        rel_width = lane_width_px / BEV_W
        if not (0.3 < rel_width < 1.5):
            self.reset()
            return _invalid(f"lane width out of range: {rel_width:.2f} of BEV "
                            f"(L_x={lx_at:.0f}, R_x={rx_at:.0f})")

        px_per_meter = lane_width_px / LANE_WIDTH_M
        lane_center_px = (lx_at + rx_at) / 2
        car_pos_px = BEV_W / 2
        offset_px = car_pos_px - lane_center_px
        offset_m_raw = offset_px / px_per_meter

        # УЖЕСТОЧЁННЫЙ sanity: offset > половины ширины полосы означает что
        # машина "вне полосы" по показаниям детектора. Это типичный признак
        # того что детектор перепрыгнул на соседнюю полосу. Отказ.
        # Раньше было > LANE_WIDTH_M (3.5м) — слишком мягко.
        if abs(offset_m_raw) > LANE_WIDTH_M / 2:
            self.reset()
            return _invalid(f"offset > half lane: {offset_m_raw:+.2f}m")

        # Sanity: резкий скачок offset → не доверяем
        if self.prev_left_fit is not None:
            if abs(offset_m_raw - self.prev_offset_m) > MAX_OFFSET_JUMP_M:
                # Не сохраняем prior, но возвращаем последнее доверенное значение
                return LaneResult(valid=True, offset_m=self.prev_offset_m,
                                  lane_width_m=LANE_WIDTH_M,
                                  reason=f"jump rejected, kept prev",
                                  bev_img=bev, binary_img=binary)

        # Сохраняем prior
        self.prev_left_fit  = left_fit
        self.prev_right_fit = right_fit
        self.prev_offset_m  = offset_m_raw

        # lane_width_px_prior обновляем когда расчёт был СТАНДАРТНЫЙ
        # (без fallback) — значит обе линии нашлись и прошли все sanity
        # включая parallelism check и rel_width. Это означает что lane_width_px
        # вычислена правдоподобно и её можно запомнить.
        # Раньше было дополнительное требование is_strong для обеих, но это
        # слишком строго для пунктира и prior почти не накапливался.
        if used_fallback is None:
            # Сглаживаем prior: 70% старого + 30% нового, чтобы не дёргался
            if self.lane_width_px_prior is None:
                self.lane_width_px_prior = lane_width_px
            else:
                self.lane_width_px_prior = 0.7 * self.lane_width_px_prior + 0.3 * lane_width_px

        result = LaneResult(
            valid=True,
            offset_m=offset_m_raw,
            lane_width_m=LANE_WIDTH_M,
            left_fit=left_fit,
            right_fit=right_fit,
            reason=f"fallback={used_fallback}" if used_fallback else "ok",
            bev_img=bev,
            binary_img=binary,
        )
        if self.debug:
            result.debug_img = self._draw_debug(
                bev, binary, left_fit, right_fit,
                lane_center_px, car_pos_px,
            )
        return result


# ═════════════════════════════════════════════════════════════════
#  Sanity-test: запускаем на синтетической картинке из туториала
# ═════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python lane_detection.py <image_path>")
        print("       Прогоняет pipeline на одной картинке и показывает результат.")
        sys.exit(0)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Не удалось загрузить {sys.argv[1]}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = LaneDetector(debug=True)
    result = detector.detect(img_rgb)

    print(f"valid: {result.valid}")
    print(f"reason: {result.reason}")
    if result.valid:
        print(f"offset_m: {result.offset_m:+.3f}")
        print(f"lane_width_m: {result.lane_width_m:.2f}")

    if result.debug_img is not None:
        cv2.imshow("Lane debug", result.debug_img)
        print("\nНажми любую клавишу в окне для выхода.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()