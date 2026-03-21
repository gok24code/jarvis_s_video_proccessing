import cv2
import mediapipe as mp
import numpy as np
import os
import time
import math

# --- MEDIAPIPE KURULUM ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_result = None

def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# --- PARMAK DURUMU ---
def get_finger_states(hand_landmarks):
    lm = hand_landmarks
    return {
        'basparmak': lm[4].x < lm[3].x,
        'isaret':    lm[8].y < lm[6].y,
        'orta':      lm[12].y < lm[10].y,
        'yuzuk':     lm[16].y < lm[14].y,
        'serce':     lm[20].y < lm[18].y,
    }

def count_fingers(fingers):
    return sum(fingers.values())

def detect_gesture(fingers):
    total = count_fingers(fingers)
    ac = fingers
    if total == 0:
        return "YUMRUK"
    if total == 5:
        return "ACIK_EL"
    if ac['isaret'] and not ac['orta'] and not ac['yuzuk'] and not ac['serce']:
        return "BIR_PARMAK"
    if ac['isaret'] and ac['orta'] and not ac['yuzuk'] and not ac['serce']:
        return "IKI_PARMAK"
    if total == 3 and ac['isaret'] and ac['orta'] and ac['yuzuk']:
        return "UC_PARMAK"
    return f"UNKNOWN"

# --- KOMUT BİLGİLERİ ---
GESTURE_INFO = {
    "YUMRUK":    {"label": "YUMRUK",   "icon": "0",  "color": (0, 80, 255)},
    "ACIK_EL":   {"label": "ACIK EL",    "icon": "1", "color": (0, 220, 180)},
    "BIR_PARMAK":{"label": "ISARET",      "icon": "2",   "color": (0, 255, 100)},
    "IKI_PARMAK":{"label": "IKI PARMAK", "icon": "3",   "color": (0, 200, 255)},
    "UC_PARMAK": {"label": "UC PARMAK",     "icon": "4",   "color": (180, 100, 255)},
    "UNKNOWN":   {"label": "---",      "icon": "???",   "color": (80, 80, 80)},
}


#verilecek komutlar buraya gelecek
KOMUTLAR = {
    "YUMRUK":     lambda: print("YUMRUK"),
    "ACIK_EL":    lambda: print("EL ACIK"),
    "BIR_PARMAK": lambda: print("ISARET"),
    "IKI_PARMAK": lambda: print("IKI PARMAK"),
    "UC_PARMAK":  lambda: print("UC PARMAK"),
}

def execute_command(gesture):
    if gesture in KOMUTLAR:
        KOMUTLAR[gesture]()

# --- ÇİZİM FONKSİYONLARI ---

def draw_scanlines(frame, alpha=0.03):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    for y in range(0, h, 4):
        cv2.line(overlay, (0, y), (w, y), (0, 0, 0), 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_corner_brackets(frame, x, y, w, h, color, size=18, thickness=2):
    # Sol üst
    cv2.line(frame, (x, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + size), color, thickness)
    # Sağ üst
    cv2.line(frame, (x + w, y), (x + w - size, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + size), color, thickness)
    # Sol alt
    cv2.line(frame, (x, y + h), (x + size, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - size), color, thickness)
    # Sağ alt
    cv2.line(frame, (x + w, y + h), (x + w - size, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - size), color, thickness)

def draw_hud_panel(frame, x, y, w, h, color, alpha=0.35):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    draw_corner_brackets(frame, x, y, w, h, color, size=12, thickness=1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (*color[::-1], 60), 1)

def draw_text_hud(frame, text, x, y, scale=0.5, color=(0, 255, 160), thickness=1, font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)

def draw_gesture_ring(frame, cx, cy, radius, color, progress=1.0, t=0):
    # Dış parlayan halka
    for i in range(3):
        alpha_ring = 0.15 - i * 0.04
        ov = frame.copy()
        cv2.circle(ov, (cx, cy), radius + i * 3, color, 1)
        cv2.addWeighted(ov, alpha_ring, frame, 1 - alpha_ring, 0, frame)

    # Ana halka
    cv2.circle(frame, (cx, cy), radius, color, 1)

    # Dönen tarama çizgisi
    angle = (t * 3) % 360
    rad = math.radians(angle)
    ex = int(cx + radius * math.cos(rad))
    ey = int(cy + radius * math.sin(rad))
    cv2.line(frame, (cx, cy), (ex, ey), color, 1)

    # Merkez nokta
    cv2.circle(frame, (cx, cy), 3, color, -1)

def draw_landmark_skeleton(frame, hand_landmarks, h, w, color):
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    # Bağlantı çizgileri
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], color, 1)

    # Landmark noktaları
    fingertips = [4, 8, 12, 16, 20]
    for i, pt in enumerate(pts):
        if i in fingertips:
            cv2.circle(frame, pt, 5, color, -1)
            cv2.circle(frame, pt, 7, color, 1)
        elif i == 0:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        else:
            cv2.circle(frame, pt, 3, color, -1)

def draw_hand_outline(frame, hand_landmarks, h, w, color):
    points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks], dtype=np.int32)
    hull = cv2.convexHull(points)

    # Glow efekti
    for i in range(4):
        glow_ov = frame.copy()
        cv2.polylines(glow_ov, [hull], True, color, 3 + i * 2)
        cv2.addWeighted(glow_ov, 0.06, frame, 0.94, 0, frame)

    cv2.polylines(frame, [hull], True, color, 1)

    # Dolgu
    fill_ov = frame.copy()
    cv2.fillPoly(fill_ov, [hull], color)
    cv2.addWeighted(fill_ov, 0.08, frame, 0.92, 0, frame)

def draw_top_bar(frame, w, gesture, color, t):
    # Üst bar
    bar_ov = frame.copy()
    cv2.rectangle(bar_ov, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.addWeighted(bar_ov, 0.6, frame, 0.4, 0, frame)
    cv2.line(frame, (0, 38), (w, 38), color, 1)

    # Sol: sistem adı
    draw_text_hud(frame, "JARVIS GESTURE CONTROL", 12, 24, 0.45, color)

    # Sağ: saat damgası + FPS placeholder
    ts = time.strftime("%H:%M:%S")
    draw_text_hud(frame, f"SYS:{ts}", w - 160, 24, 0.4, color)

    # Orta: mevcut jest
    info = GESTURE_INFO.get(gesture, GESTURE_INFO["UNKNOWN"])
    label = info["label"]
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
    tx = (w - text_size[0]) // 2
    draw_text_hud(frame, label, tx, 24, 0.5, color)

def draw_bottom_bar(frame, h, w, hand_count, gesture_history, color):
    bar_ov = frame.copy()
    cv2.rectangle(bar_ov, (0, h - 34), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar_ov, 0.6, frame, 0.4, 0, frame)
    cv2.line(frame, (0, h - 34), (w, h - 34), color, 1)

    draw_text_hud(frame, f"HANDS:{hand_count}", 12, h - 12, 0.38, color)

    # Jest geçmişi
    hist_x = 120
    for i, g in enumerate(gesture_history[-6:]):
        info = GESTURE_INFO.get(g, GESTURE_INFO["UNKNOWN"])
        alpha_val = 0.3 + i * 0.12
        c = tuple(int(v * alpha_val) for v in info["color"])
        draw_text_hud(frame, info["icon"], hist_x + i * 52, h - 12, 0.38, c)

    draw_text_hud(frame, "Q:QUIT", w - 80, h - 12, 0.38, color)

def draw_side_panel(frame, h, w, gesture, color):
    # Sağ panel
    px, py, pw, ph = w - 130, 50, 120, 200
    draw_hud_panel(frame, px, py, pw, ph, color)
    draw_text_hud(frame, "GESTURES", px + 8, py + 18, 0.35, color)
    cv2.line(frame, (px + 4, py + 24), (px + pw - 4, py + 24), color, 1)

    gestures_list = [
        ("YUMRUK",     "0"),
        ("ACIK_EL",    "1"),
        ("BIR_PARMAK", "2"),
        ("IKI_PARMAK", "3"),
        ("UC_PARMAK",  "4"),
    ]
    for i, (g, lbl) in enumerate(gestures_list):
        gy = py + 44 + i * 28
        is_active = gesture == g
        if is_active:
            ov = frame.copy()
            cv2.rectangle(ov, (px + 2, gy - 12), (px + pw - 2, gy + 8), color, -1)
            cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)
            draw_text_hud(frame, f"> {lbl}", px + 8, gy, 0.35, (255, 255, 255))
        else:
            draw_text_hud(frame, f"  {lbl}", px + 8, gy, 0.32, (60, 120, 80))

def draw_grid_overlay(frame, color, alpha=0.04):
    h, w = frame.shape[:2]
    ov = frame.copy()
    step = 60
    for x in range(0, w, step):
        cv2.line(ov, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(ov, (0, y), (w, y), color, 1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

# --- MODEL ---
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=on_result
)

# --- ANA DÖNGÜ ---
cap = cv2.VideoCapture(0)
timestamp = 0
prev_gesture = None
gesture_history = []
frame_count = 0
start_time = time.time()
fps = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Ayna modu
        h, w = frame.shape[:2]
        timestamp += 1
        frame_count += 1

        # FPS hesapla
        elapsed = time.time() - start_time
        if elapsed > 0.5:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        t = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, timestamp)

        # Grid arka plan
        draw_grid_overlay(frame, (0, 60, 30))

        current_gesture = "UNKNOWN"
        hand_count = 0

        if latest_result:
            hand_count = len(latest_result.hand_landmarks)
            for hand_landmarks in latest_result.hand_landmarks:
                fingers = get_finger_states(hand_landmarks)
                current_gesture = detect_gesture(fingers)

                info = GESTURE_INFO.get(current_gesture, GESTURE_INFO["UNKNOWN"])
                color = info["color"]

                # El merkezi
                cx = int(np.mean([lm.x for lm in hand_landmarks]) * w)
                cy = int(np.mean([lm.y for lm in hand_landmarks]) * h)

                # Outline + skeleton
                draw_hand_outline(frame, hand_landmarks, h, w, color)
                draw_landmark_skeleton(frame, hand_landmarks, h, w, color)

                # Dönen tarama halkası
                draw_gesture_ring(frame, cx, cy, 65, color, t=t)

                # El üstü etiket
                draw_corner_brackets(frame, cx - 45, cy - 80, 90, 25, color, size=8)
                draw_text_hud(frame, info["label"], cx - 38, cy - 62, 0.38, (255, 255, 255))

        # Komut tetikle
        if current_gesture != prev_gesture:
            execute_command(current_gesture)
            if current_gesture != "UNKNOWN":
                gesture_history.append(current_gesture)
            prev_gesture = current_gesture

        # HUD rengi (el varsa aktif renk, yoksa dim yeşil)
        if hand_count > 0 and latest_result:
            last_info = GESTURE_INFO.get(current_gesture, GESTURE_INFO["UNKNOWN"])
            hud_color = last_info["color"]
        else:
            hud_color = (0, 140, 80)

        # Tarama çizgisi efekti
        draw_scanlines(frame)

        # UI katmanları
        draw_top_bar(frame, w, current_gesture, hud_color, t)
        draw_bottom_bar(frame, h, w, hand_count, gesture_history, hud_color)
        draw_side_panel(frame, h, w, current_gesture, hud_color)

        # FPS
        draw_text_hud(frame, f"FPS:{fps:.0f}", w - 80, 60, 0.38, hud_color)

        # El yok uyarısı
        if hand_count == 0:
            msg = "EL ALGILANIYOR..."
            blink = int(t * 2) % 2 == 0
            if blink:
                ts2 = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
                draw_text_hud(frame, msg, (w - ts2[0]) // 2, h // 2, 0.6, (0, 140, 80))

        cv2.imshow('JARVIS | GESTURE CONTROL', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()