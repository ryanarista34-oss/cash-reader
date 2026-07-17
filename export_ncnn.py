from ultralytics import YOLO

print("Loading model...")

model = YOLO("best.pt")

print("Exporting to NCNN...")

model.export(
    format="ncnn"
)

print("Selesai!")