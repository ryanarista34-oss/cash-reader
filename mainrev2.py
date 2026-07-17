import cv2
import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO
import threading
import time
from datetime import datetime
import subprocess
import psutil


# ===== KONFIGURASI =====
MODEL_PATH = "best_ncnn_model"
DB_URL = "https://cashreader2026-default-rtdb.asia-southeast1.firebasedatabase.app"

# ===== COOLDOWN =====
last_nominal = None
last_detection_time = 0
COOLDOWN = 5

# ===== STATUS PERANGKAT =====
camera_status = False
speaker_status = True

# ===== KAMUS NOMINAL =====
kamus = {
    "dua puluh ribu": "Dua Puluh Ribu Rupiah",
    "dua ribu": "Dua Ribu Rupiah",
    "lima puluh ribu": "Lima Puluh Ribu Rupiah",
    "lima ribu": "Lima Ribu Rupiah",
    "sepuluh ribu": "Sepuluh Ribu Rupiah",
    "seratus ribu": "Seratus Ribu Rupiah",
    "seribu": "Seribu Rupiah",
}

# ===== AUDIO =====
def putar_audio(file_audio):

    try:

        subprocess.Popen(

            [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                file_audio
            ],

            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL

        )

    except Exception as e:

        print("Audio Error:", e)
        
# ===== FIREBASE =====
def init_firebase():

    if not firebase_admin._apps:

        cred = credentials.Certificate(
            "serviceAccountKey.json"
        )

        firebase_admin.initialize_app(
            cred,
            {
                "databaseURL": DB_URL
            }
        )

    return db.reference('/')

# ===== KIRIM LOG =====
def kirim_log(firebase_ref, nominal, currency, confidence):

    try:

        firebase_ref.child("detections").push({
            "nominal": nominal,
            "currency": currency,
            "confidence": round(confidence, 4),
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        })

        print("Log berhasil dikirim ke Firebase")

    except Exception as e:

        print("Firebase Error:", e)


# ===== HEARTBEAT =====
def kirim_status():

    while True:

        try:

            db.reference("device_status").set({

                "status": "online",

                "camera": camera_status,

                "speaker": speaker_status,

                "cpu": round(
                    psutil.cpu_percent(interval=0.5),
                    1
                ),

                "ram": round(
                    psutil.virtual_memory().percent,
                    1
                ),

                "last_seen": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            })

        except Exception as e:

            print("Heartbeat Error:", e)

        time.sleep(5)

# ===== MAIN =====
def main():

    global last_nominal
    global last_detection_time
    global camera_status

    print("=" * 40)
    print(" CASH READER REALTIME MODE ")
    print("=" * 40)

    firebase_ref = init_firebase()

    threading.Thread(
        target=kirim_status,
        daemon=True
    ).start()

    print("Loading YOLO model...")

    model = YOLO(
        MODEL_PATH,
        task="detect"
)


    print("Model berhasil dimuat!")

    putar_audio("audio/siap.mp3")

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_V4L2
    )

    if not cap.isOpened():
        print("Webcam tidak terdeteksi!")
        putar_audio("audio/tidak_terdeteksi.mp3")
        return

    audio_map = {
        "Seribu Rupiah": "audio/1000.mp3",
        "Dua Ribu Rupiah": "audio/2000.mp3",
        "Lima Ribu Rupiah": "audio/5000.mp3",
        "Sepuluh Ribu Rupiah": "audio/10000.mp3",
        "Dua Puluh Ribu Rupiah": "audio/20000.mp3",
        "Lima Puluh Ribu Rupiah": "audio/50000.mp3",
        "Seratus Ribu Rupiah": "audio/100000.mp3"
    }

    while True:

        ret, frame = cap.read()

        camera_status = ret

        if not ret:

            camera_status = False

            print("Webcam disconnected. Reconnecting...")

            cap.release()

            time.sleep(2)

            cap = cv2.VideoCapture(
                0,
                cv2.CAP_V4L2
            )
            if not cap.isOpened():
                print("Waiting for webcam...")
                continue

        results = model(
            frame,
            conf=0.7,
            verbose=False
        )

        if len(results[0].boxes) > 0:

            best = max(
                results[0].boxes,
                key=lambda x: float(x.conf)
            )

            kelas = results[0].names[
                int(best.cls)
            ]

            conf = float(best.conf)

            nominal = kamus.get(
                kelas,
                kelas
            )

            confidence_percent = round(
                conf * 100,
                2
            )

            # ===== BOUNDING BOX =====
            x1, y1, x2, y2 = map(
                int,
                best.xyxy[0]
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{nominal} ({confidence_percent:.1f}%)",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # ===== COOLDOWN =====
            current_time = time.time()

            if (
                nominal != last_nominal
                or
                current_time - last_detection_time >= COOLDOWN
            ):

                print(
                    f"[DETEKSI] {nominal} "
                    f"({confidence_percent}%)"
                )

                if nominal in audio_map:
                    putar_audio(
                        audio_map[nominal]
                    )

                threading.Thread(
                    target=kirim_log,
                    args=(
                        firebase_ref,
                        nominal,
                        "IDR",
                        conf
                    ),
                    daemon=True
                ).start()

                last_nominal = nominal
                last_detection_time = current_time

        cv2.imshow(
            "Cash Reader Realtime",
            frame
        )

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    try:
        db.reference("device_status").update({
            "status": "offline",
            "last_seen": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        })

    except Exception as e:
        print(e)

    print("Program selesai.")

if __name__ == "__main__":
    main()