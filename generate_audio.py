import asyncio
import edge_tts
import os

os.makedirs("audio", exist_ok=True)

audio_files = {
    "1000.mp3": "Nominal yang terdeteksi adalah Seribu Rupiah",
    "2000.mp3": "Nominal yang terdeteksi adalah Dua Ribu Rupiah",
    "5000.mp3": "Nominal yang terdeteksi adalah Lima Ribu Rupiah",
    "10000.mp3": "Nominal yang terdeteksi adalah Sepuluh Ribu Rupiah",
    "20000.mp3": "Nominal yang terdeteksi adalah Dua Puluh Ribu Rupiah",
    "50000.mp3": "Nominal yang terdeteksi adalah Lima Puluh Ribu Rupiah",
    "100000.mp3": "Nominal yang terdeteksi adalah Seratus Ribu Rupiah",
    "siap.mp3": "Sistem siap digunakan",
    "tidak_terdeteksi.mp3": "Uang tidak berhasil terdeteksi"
}

async def generate():
    for filename, text in audio_files.items():

        print(f"Membuat {filename}...")

        communicate = edge_tts.Communicate(
            text,
            voice="id-ID-GadisNeural"
        )

        await communicate.save(
            f"audio/{filename}"
        )

    print("Selesai!")

asyncio.run(generate())