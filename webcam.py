import cv2
from ultralytics import YOLO

print("Carregando YOLOv8m...")
model = YOLO("yolov8m.pt")

print("Abrindo webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: nao foi possivel abrir a webcam.")
    raise SystemExit(1)

print("Webcam ativa! Pressione Q para sair.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar imagem.")
            break

        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.35,
            verbose=False
        )

        annotated = results[0].plot()
        cv2.imshow("YOLOv8m - Webcam", annotated)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
