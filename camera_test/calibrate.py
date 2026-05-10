"""
calibrate.py — визуальный отладчик для калибровки lane_detection.py.

Принимает кадр, прогоняет pipeline пошагово, сохраняет промежуточные
картинки в папку debug/. Это намного полезнее чем просто запускать
lane_detection.py — видно ГДЕ ИМЕННО ломается.

Запуск:
    python calibrate.py test_frames/frame_00.png

Что смотреть в debug/:
    01_original.png       — исходный кадр со src-точками (зелёная трапеция)
    02_roi.png            — после ROI crop
    03_birdseye.png       — bird's-eye view (ЗДЕСЬ линии должны быть вертикальные!)
    04_binary.png         — бинаризация (ЗДЕСЬ должны быть видны ОБЕ линии)
    05_windows.png        — sliding windows + найденные пиксели
    06_polynomials.png    — аппроксимирующие кривые
    07_final.png          — финал с полосой и offset

ИНТЕРПРЕТАЦИЯ:
    • На 03 линии не вертикальные → крутить SRC_POINTS_FRAC верхние точки
    • На 03 линии вертикальные, но НЕ параллельные → крутить нижние
    • На 04 линии слабо видны / много шума → крутить WHITE_HLS_MIN
    • На 04 видны хорошо, но 05 показывает что окна "ушли" → крутить WIN_MARGIN
    • На 07 lane width в метрах не похожа на правду → крутить LANE_WIDTH_M
"""

import cv2
import numpy as np
import os
import sys

import lane_detection as ld


def calibrate(image_path: str, out_dir: str = 'debug'):
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Не удалось загрузить {image_path}")
        return

    H, W = img.shape[:2]
    print(f"Кадр: {W}x{H}")

    # ─── Шаг 1: исходник + src-точки ───
    roi_top = int(H * ld.ROI_TOP_FRAC)
    roi_bot = int(H * ld.ROI_BOTTOM_FRAC)
    roi_h = roi_bot - roi_top

    src_in_orig = np.array([
        (int(x_frac * W), int(y_frac * roi_h) + roi_top)
        for x_frac, y_frac in ld.SRC_POINTS_FRAC
    ], dtype=np.int32)

    vis1 = img.copy()
    cv2.polylines(vis1, [src_in_orig], True, (0, 255, 0), 2)
    cv2.line(vis1, (0, roi_top), (W, roi_top), (0, 165, 255), 1)  # ROI top
    cv2.line(vis1, (0, roi_bot-1), (W, roi_bot-1), (0, 165, 255), 1)  # ROI bot
    cv2.imwrite(f'{out_dir}/01_original.png', vis1)

    # ─── Шаг 2: ROI ───
    roi = img[roi_top:roi_bot, :]
    cv2.imwrite(f'{out_dir}/02_roi.png', roi)

    # ─── Шаг 3: bird's-eye ───
    src = np.float32([
        (x_frac * W, y_frac * roi_h)
        for x_frac, y_frac in ld.SRC_POINTS_FRAC
    ])
    dst = np.float32([
        (0, 0), (ld.BEV_W, 0), (ld.BEV_W, ld.BEV_H), (0, ld.BEV_H),
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    bev = cv2.warpPerspective(roi, M, (ld.BEV_W, ld.BEV_H))
    cv2.imwrite(f'{out_dir}/03_birdseye.png', bev)

    # ─── Шаг 4: бинаризация ───
    hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
    white  = cv2.inRange(hls, ld.WHITE_HLS_MIN,  ld.WHITE_HLS_MAX)
    yellow = cv2.inRange(hls, ld.YELLOW_HLS_MIN, ld.YELLOW_HLS_MAX)
    binary = cv2.bitwise_or(white, yellow)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f'{out_dir}/04_binary.png', binary)

    # Гистограмма — отдельным файлом для понимания
    hist = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    if hist.max() > 0:
        hist_img = np.zeros((200, ld.BEV_W, 3), dtype=np.uint8)
        for x, h in enumerate(hist):
            h_norm = int((h / hist.max()) * 195)
            cv2.line(hist_img, (x, 200), (x, 200 - h_norm), (255, 255, 255), 1)
        midpoint = len(hist) // 2
        left_x = int(np.argmax(hist[:midpoint]))
        right_x = int(np.argmax(hist[midpoint:])) + midpoint
        cv2.line(hist_img, (left_x, 0), (left_x, 200), (0, 0, 255), 1)
        cv2.line(hist_img, (right_x, 0), (right_x, 200), (0, 255, 0), 1)
        cv2.line(hist_img, (midpoint, 0), (midpoint, 200), (255, 255, 0), 1)
        cv2.imwrite(f'{out_dir}/04b_histogram.png', hist_img)
        print(f"  Гистограмма: пик слева x={left_x}, справа x={right_x}, midpoint={midpoint}")

    # ─── Шаг 5: sliding window ───
    win_h = binary.shape[0] // ld.N_WINDOWS
    nonzero = binary.nonzero()
    ny, nx = np.array(nonzero[0]), np.array(nonzero[1])

    vis5 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    if hist.max() == 0:
        print("  ❌ Бинаризация дала пустую картинку — pipeline не пойдёт дальше")
        cv2.imwrite(f'{out_dir}/05_windows.png', vis5)
        return

    midpoint = len(hist) // 2
    left_cur = int(np.argmax(hist[:midpoint]))
    right_cur = int(np.argmax(hist[midpoint:])) + midpoint

    left_idx_all, right_idx_all = [], []
    for w in range(ld.N_WINDOWS):
        y_low  = binary.shape[0] - (w+1) * win_h
        y_high = binary.shape[0] - w * win_h
        xl_low, xl_high = left_cur - ld.WIN_MARGIN, left_cur + ld.WIN_MARGIN
        xr_low, xr_high = right_cur - ld.WIN_MARGIN, right_cur + ld.WIN_MARGIN

        cv2.rectangle(vis5, (xl_low, y_low), (xl_high, y_high), (0, 255, 0), 2)
        cv2.rectangle(vis5, (xr_low, y_low), (xr_high, y_high), (0, 255, 0), 2)

        in_left = ((ny >= y_low) & (ny < y_high) &
                   (nx >= xl_low) & (nx < xl_high)).nonzero()[0]
        in_right = ((ny >= y_low) & (ny < y_high) &
                    (nx >= xr_low) & (nx < xr_high)).nonzero()[0]

        left_idx_all.append(in_left)
        right_idx_all.append(in_right)

        if len(in_left) > ld.MIN_PIX:
            left_cur = int(np.mean(nx[in_left]))
        if len(in_right) > ld.MIN_PIX:
            right_cur = int(np.mean(nx[in_right]))

    left_idx_all = np.concatenate(left_idx_all)
    right_idx_all = np.concatenate(right_idx_all)
    vis5[ny[left_idx_all], nx[left_idx_all]] = [255, 0, 0]
    vis5[ny[right_idx_all], nx[right_idx_all]] = [0, 0, 255]
    cv2.imwrite(f'{out_dir}/05_windows.png', vis5)
    print(f"  Sliding window: левая={len(left_idx_all)}пикс, правая={len(right_idx_all)}пикс")

    # ─── Шаг 6: полиномы ───
    if len(left_idx_all) < 50 or len(right_idx_all) < 50:
        print(f"  ❌ Слишком мало пикселей для полинома")
        return

    lx, ly = nx[left_idx_all], ny[left_idx_all]
    rx, ry = nx[right_idx_all], ny[right_idx_all]
    left_fit = np.polyfit(ly, lx, 2)
    right_fit = np.polyfit(ry, rx, 2)

    vis6 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    ploty = np.linspace(0, ld.BEV_H-1, ld.BEV_H)
    lf = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rf = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    for y, x in zip(ploty.astype(int), lf.astype(int)):
        if 0 <= x < vis6.shape[1]:
            cv2.circle(vis6, (x, y), 2, (255, 200, 0), -1)
    for y, x in zip(ploty.astype(int), rf.astype(int)):
        if 0 <= x < vis6.shape[1]:
            cv2.circle(vis6, (x, y), 2, (0, 200, 255), -1)
    cv2.imwrite(f'{out_dir}/06_polynomials.png', vis6)

    # ─── Шаг 7: финал ───
    y_eval = ld.BEV_H - 1
    lx_at = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    rx_at = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_width_px = rx_at - lx_at

    if lane_width_px <= 0:
        print(f"  ❌ Линии перепутаны: левая={lx_at:.0f} > правая={rx_at:.0f}")
        return

    px_per_meter = lane_width_px / ld.LANE_WIDTH_M
    lane_center_px = (lx_at + rx_at) / 2
    car_pos_px = ld.BEV_W / 2
    offset_px = car_pos_px - lane_center_px
    offset_m = offset_px / px_per_meter

    rel_width = lane_width_px / ld.BEV_W

    print(f"\n  ──── РЕЗУЛЬТАТ ────")
    print(f"  левая линия в нижнем ряду:  X = {lx_at:.1f}px")
    print(f"  правая линия в нижнем ряду: X = {rx_at:.1f}px")
    print(f"  ширина полосы:              {lane_width_px:.1f}px = {rel_width*100:.1f}% от BEV")
    print(f"  scale:                      {px_per_meter:.1f} px/м (при LANE_WIDTH_M={ld.LANE_WIDTH_M})")
    print(f"  смещение машины:            {offset_m:+.3f}м")

    if rel_width < 0.3:
        print(f"  ⚠ ширина полосы < 30% BEV — pipeline отвергнет результат")
        print(f"     Скорее всего sliding window нашёл одну линию вместо двух.")
    elif rel_width > 1.5:
        print(f"  ⚠ ширина полосы > 150% BEV — pipeline отвергнет результат")

    # Финальная картинка
    overlay = np.zeros_like(bev)
    pts_l = np.array([np.transpose(np.vstack([lf, ploty]))])
    pts_r = np.array([np.flipud(np.transpose(np.vstack([rf, ploty])))])
    pts = np.hstack((pts_l, pts_r))
    cv2.fillPoly(overlay, np.int32([pts]), (0, 200, 0))
    final_bev = cv2.addWeighted(bev, 1.0, overlay, 0.4, 0)
    cv2.line(final_bev, (int(lane_center_px), 0), (int(lane_center_px), ld.BEV_H), (255, 100, 0), 2)
    cv2.line(final_bev, (int(car_pos_px), 0), (int(car_pos_px), ld.BEV_H), (0, 0, 255), 2)
    cv2.putText(final_bev, f"offset: {offset_m:+.2f}m", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(f'{out_dir}/07_final.png', final_bev)

    print(f"\n  Все шаги в папке {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python calibrate.py <image_path>")
        sys.exit(1)
    calibrate(sys.argv[1])
