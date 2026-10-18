import os
import math
import time
import cv2
import mediapipe as mp
import numpy as np
import pygame
from flask import Flask, Response, render_template
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'hand_landmarker.task')
AUDIO_ACERTO = os.path.join(BASE_DIR, 'audio', 'lula.mp3')
AUDIO_ERRO = os.path.join(BASE_DIR, 'audio', 'fogo.mp3')
AUDIO_BG = os.path.join(BASE_DIR, 'audio', 'estrela.mp3')

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 4.0
FONT_THICKNESS = 8

IMAGEM_DIR = os.path.join(BASE_DIR, 'imagem')
IMG_BD1 = cv2.imread(os.path.join(IMAGEM_DIR, 'bd1.png'), cv2.IMREAD_UNCHANGED)
IMG_BD2 = cv2.imread(os.path.join(IMAGEM_DIR, 'bd2.png'), cv2.IMREAD_UNCHANGED)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

THUMB_RATIO = 0.5

pygame.mixer.init()
acerto_som = pygame.mixer.Sound(AUDIO_ACERTO)
erro_som = pygame.mixer.Sound(AUDIO_ERRO)
bg_som = pygame.mixer.Sound(AUDIO_BG)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)
detector = vision.HandLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)

ultima_numero = None
som_atual = None
ultimo_timestamp = 0


def dedos_levantados_ate5(landmarks):
    dedos = 0
    if landmarks[8].y < landmarks[6].y:
        dedos += 1
    if landmarks[12].y < landmarks[10].y:
        dedos += 1
    if landmarks[16].y < landmarks[14].y:
        dedos += 1
    if landmarks[20].y < landmarks[18].y:
        dedos += 1
    palm_size = math.hypot(landmarks[9].x - landmarks[0].x,
                           landmarks[9].y - landmarks[0].y)
    thumb_dist = math.hypot(landmarks[4].x - landmarks[5].x,
                            landmarks[4].y - landmarks[5].y)
    if palm_size > 0 and thumb_dist > THUMB_RATIO * palm_size:
        dedos += 1
    return dedos


def overlay_png(frame, overlay, center_x=None, center_y=None, scale=1.0):
    if overlay is None:
        return
    h, w = overlay.shape[:2]
    if scale != 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        overlay = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = overlay.shape[:2]
    fh, fw = frame.shape[:2]
    if center_x is None:
        center_x = fw // 2
    if center_y is None:
        center_y = fh // 2
    x = center_x - w // 2
    y = center_y - h // 2
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)
    if x1 >= x2 or y1 >= y2:
        return
    ov_x1 = x1 - x
    ov_y1 = y1 - y
    ov_x2 = ov_x1 + (x2 - x1)
    ov_y2 = ov_y1 + (y2 - y1)
    crop = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
    if crop.shape[2] == 4:
        alpha = crop[:, :, 3:4] / 255.0
        rgb = crop[:, :, :3]
    else:
        alpha = np.ones((crop.shape[0], crop.shape[1], 1), dtype=np.float32)
        rgb = crop
    roi = frame[y1:y2, x1:x2].astype(np.float32)
    blended = (1.0 - alpha) * roi + alpha * rgb.astype(np.float32)
    frame[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)


def desenhar_maos(frame, hand_landmarks_list, cor_por_indice=None):
    h, w = frame.shape[:2]
    for i, landmarks in enumerate(hand_landmarks_list):
        cor = (0, 255, 0)
        if cor_por_indice and i in cor_por_indice:
            cor = cor_por_indice[i]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], cor, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, p, 4, cor, 1, cv2.LINE_AA)


def desenhar_numero(frame, numero, cor=(255, 255, 255, 100)):
    if numero == 22:
        cor = (180, 17, 1)
    elif numero == 13:
        cor = (1, 204, 255)
    texto = f'{numero:02d}' if numero is not None else '?'
    (tw, th), _ = cv2.getTextSize(texto, FONT, FONT_SCALE, FONT_THICKNESS)
    x = (frame.shape[1] - tw) // 2
    y = (frame.shape[0] + th) // 2
    cv2.putText(frame, texto, (x, y), FONT, FONT_SCALE, cor, FONT_THICKNESS, cv2.LINE_AA)


def tocar_som(numero):
    global ultima_numero, som_atual
    if numero is None:
        if ultima_numero is None:
            return
        if som_atual:
            som_atual.stop()
            som_atual = None
        ultima_numero = None
        return
    if numero != 13 and numero != 22:
        numero = 10
    if numero == ultima_numero:
        return
    som = None
    if numero == 13:
        som = acerto_som
    elif numero == 22:
        som = erro_som
    elif numero == 10:
        som = bg_som
    if som_atual and som_atual is not som:
        som_atual.stop()
        som_atual = None
    if som:
        som.play(loops=-1)
        som_atual = som
    elif som_atual:
        som_atual.stop()
        som_atual = None
    ultima_numero = numero

def processar_frame(frame):
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame[:, :, 2] = (frame[:, :, 2] * 0.4).astype(np.uint8)
    frame[:, :, 1] = (frame[:, :, 1] * 0.0).astype(np.uint8)
    frame[:, :, 0] = (frame[:, :, 0] * 0.0).astype(np.uint8)

    global ultimo_timestamp
    timestamp_ms = int(time.time() * 1000)
    if timestamp_ms <= ultimo_timestamp:
        timestamp_ms = ultimo_timestamp + 1
    ultimo_timestamp = timestamp_ms
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect_for_video(mp_image, timestamp_ms)

    numero_final = None
    dezena_lm = None
    unidade_lm = None

    if results.hand_landmarks and len(results.hand_landmarks) >= 2:
        hands_info = []
        for i, handedness in enumerate(results.handedness):
            if handedness and handedness[0]:
                hands_info.append((i, handedness[0].category_name))
        if len(hands_info) != 2:
            hands_info = [(0, 'Right'), (1, 'Left')]

        landmarks_por_idx = {i: lm for i, lm in enumerate(results.hand_landmarks)}

        left_idx = None
        right_idx = None
        for lm_idx, cat in hands_info:
            if cat == 'Left':
                left_idx = lm_idx
            elif cat == 'Right':
                right_idx = lm_idx

        def get_x(lm_idx):
            lm = landmarks_por_idx.get(lm_idx)
            if lm is None:
                return 9999
            return lm[0].x * frame.shape[1]

        if left_idx is not None and right_idx is not None:
            if get_x(left_idx) < get_x(right_idx):
                dezena_lm = landmarks_por_idx[left_idx]
                unidade_lm = landmarks_por_idx[right_idx]
            else:
                dezena_lm = landmarks_por_idx[right_idx]
                unidade_lm = landmarks_por_idx[left_idx]
            d = dedos_levantados_ate5(dezena_lm)
            u = dedos_levantados_ate5(unidade_lm)
            numero_final = d * 10 + u
        else:
            lm_list = list(landmarks_por_idx.items())[:2]
            lm_list.sort(key=lambda kv: kv[1][0].x * frame.shape[1])
            dezena_lm = lm_list[0][1]
            unidade_lm = lm_list[1][1]
            d = dedos_levantados_ate5(dezena_lm)
            u = dedos_levantados_ate5(unidade_lm)
            numero_final = d * 10 + u

        if numero_final == 13 and IMG_BD1 is not None:
            overlay_png(frame, IMG_BD1, center_x=frame.shape[1] // 2, center_y=frame.shape[0] // 2, scale=1.0)
        elif numero_final == 22 and IMG_BD2 is not None:
            overlay_png(frame, IMG_BD2, center_x=frame.shape[1] // 2, center_y=frame.shape[0] // 2, scale=1.0)
        desenhar_numero(frame, numero_final, (255, 255, 255) if numero_final is not None else (200, 200, 200))

    if results.hand_landmarks:
        cor_por_indice = {}
        for idx, lm in enumerate(results.hand_landmarks):
            if lm is dezena_lm:
                cor_por_indice[idx] = (0, 255, 0)
            elif lm is unidade_lm:
                cor_por_indice[idx] = (255, 255, 0)
        desenhar_maos(frame, results.hand_landmarks, cor_por_indice)

    tocar_som(numero_final)
    return frame


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = processar_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
