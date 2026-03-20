import cv2
import mediapipe as mp
import numpy as np
import os
import time
import math
import random

# --- YAPILANDIRMA VE KURULUM ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode



latest_result = None

def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# --- VOID INK FİZİK SINIFI ---
class VoidInk:
    def __init__(self, w, h):
        self.center = np.array([float(w // 2), float(h // 2)])
        self.num_points = 24
        self.radius = 70
        self.points = []
        for i in range(self.num_points):
            angle = (i / self.num_points) * 2 * math.pi
            px = self.center[0] + math.cos(angle) * self.radius
            py = self.center[1] + math.sin(angle) * self.radius
            self.points.append({'pos': np.array([px, py]), 'vel': np.array([0.0, 0.0])})
        
        self.stiffness = 0.12
        self.friction = 0.88
        self.grasping = False

    def update(self, hand_landmarks, w, h):
        target = None
        self.grasping = False
        
        if hand_landmarks:
            # İşaret parmağı ucu (8) ve Baş parmak ucu (4)
            ix, iy = hand_landmarks[8].x * w, hand_landmarks[8].y * h
            tx, ty = hand_landmarks[4].x * w, hand_landmarks[4].y * h
            
            target = np.array([ix, iy])
            dist_pinch = math.sqrt((ix-tx)**2 + (iy-ty)**2)
            
            # Eğer parmaklar birleşmişse "TUTMA" modu
            if dist_pinch < 45:
                self.grasping = True

        for i, p in enumerate(self.points):
            angle = (i / self.num_points) * 2 * math.pi
            # Merkeze çekilme (Elastikiyet)
            home = self.center + np.array([math.cos(angle)*self.radius, math.sin(angle)*self.radius])
            force = (home - p['pos']) * self.stiffness
            
            if target is not None:
                dist_to_hand = np.linalg.norm(target - p['pos'])
                if self.grasping and dist_to_hand < 120:
                    # Parmaklara yapış
                    force += (target - p['pos']) * 0.6
                elif dist_to_hand < 180:
                    # Hafif etkileşim
                    force += (target - p['pos']) * 0.08

            # Venom dalgalanma efekti
            force += np.array([random.uniform(-0.8, 0.8), random.uniform(-0.8, 0.8)])

            p['vel'] = (p['vel'] + force) * self.friction
            p['pos'] += p['vel']
            
            # Eğer tutuyorsak merkezi de yavaşça kaydır
            if self.grasping:
                self.center = self.center * 0.95 + target * 0.05

    def draw(self, frame):
        pts = np.array([p['pos'] for p in self.points], dtype=np.int32)
        hull = cv2.convexHull(pts)
        
        # Siyah maske üzerine beyaz çizim
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [hull], 255)
        
        # Organik yumuşatma
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        
        # Frame'e "Void" olarak uygula
        frame[mask > 40] = [255, 255, 255]
        
        # Parlayan kenarlar
        cv2.polylines(frame, [hull], True, (200, 200, 200), 2)
        if self.grasping:
             cv2.circle(frame, (int(self.center[0]), int(self.center[1])), 5, (255, 255, 255), -1)

# --- YARDIMCI ÇİZİM FONKSİYONLARI ---
def draw_hand_silhouette(frame, hand_landmarks, w, h, grasping):
    # El noktalarını piksel koordinatlarına çevir (hand_landmarks zaten bir listedir)
    points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks], dtype=np.int32)
    hull = cv2.convexHull(points)
    
    # Elin rengi: Tutuyorsa maviye çalsın, normalde yeşilimsi/beyaz
    color = (255, 100, 0) if grasping else (150, 150, 150)
    
    # Yarı saydam bir katman oluştur
    overlay = frame.copy()
    
    # Elin içini doldur (Silüet)
    cv2.fillPoly(overlay, [hull], color)
    
    # Dış hatları çiz (Hafif parlama)
    cv2.polylines(overlay, [hull], True, color, 2)
    
    # Ana frame ile birleştir (Saydamlık: 0.15 elin görünürlüğü)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    
    # Parmak uçlarına küçük noktalar koy (Referans için)
    for pt in points[[4, 8, 12, 16, 20]]: # Parmak uçları
        cv2.circle(frame, tuple(pt), 3, color, -1)

def draw_hud(frame, w, h, grasping, fps):
    # Üst Bilgi
    color = (0, 255, 150) if not grasping else (0, 150, 255)
    cv2.putText(frame, "VOID SYMBIOSE SYSTEM", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    # Alt Durum
    status = "GRASPING ACTIVE" if grasping else "READY TO CONNECT"
    cv2.putText(frame, status, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.line(frame, (20, h-50), (250, h-50), color, 1)

def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')
    if not os.path.exists(model_path):
        print(f"HATA: {model_path} bulunamadı!")
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
        result_callback=on_result
    )

    cap = cv2.VideoCapture(0)
    
    # Standart genişlik ve yükseklik alma
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Eğer değerler 0 gelirse (bazı kameralarda olabilir), varsayılan ata
    if w <= 0: w, h = 640, 480

    void_ink = VoidInk(w, h)
    timestamp = 0
    start_time = time.time()
    frame_count = 0
    fps = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            timestamp += 1
            frame_count += 1

            # FPS Hesapla
            if time.time() - start_time > 1:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()

            # El Takibi
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, timestamp)

            # EKRANI SİYAH YAP (Void Bölgesi)
            black_screen = np.zeros_like(frame)
            
            # El Verisini Al
            hand_landmarks = None
            if latest_result and latest_result.hand_landmarks:
                hand_landmarks = latest_result.hand_landmarks[0]

            # Fiziği Güncelle ve Çiz
            void_ink.update(hand_landmarks, w, h)
            
            # EL SİLÜETİ (Mürekkebin altında kalsın diye önce bunu çiziyoruz)
            if hand_landmarks:
                draw_hand_silhouette(black_screen, hand_landmarks, w, h, void_ink.grasping)
            
            # MÜREKKEP
            void_ink.draw(black_screen)

            # HUD ve Efektler
            draw_hud(black_screen, w, h, void_ink.grasping, fps)
            
            # Scanline efekti
            for y in range(0, h, 4):
                cv2.line(black_screen, (0, y), (w, y), (20, 20, 20), 1)

            cv2.imshow('VENOM VOID | EXPERIMENT', black_screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
