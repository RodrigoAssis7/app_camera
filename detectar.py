"""Detecta objetos em fotos ou ao vivo na webcam com YOLO e gera relatorio.

Uso:
  python detectar.py fotos/                 # pasta com fotos
  python detectar.py fotos/mesa1.jpg        # uma foto
  python detectar.py --camera               # captura uma foto da webcam e detecta
  python detectar.py --ao-vivo              # webcam em tempo real (tecla q sai)
  python detectar.py fotos/ --conf 0.3 --imgsz 1280 --modelo yolov8s.pt
"""
import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def capturar(destino: Path, indice_camera: int = 0) -> Path:
    cap = cv2.VideoCapture(indice_camera)
    ok, frame = False, None
    for _ in range(10):  # descarta os primeiros frames (exposicao ainda ajustando)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Nao foi possivel capturar a imagem da camera")
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"captura_{int(time.time())}.jpg"
    cv2.imwrite(str(caminho), frame)
    return caminho


def listar_imagens(origem: Path) -> list[Path]:
    if origem.is_file():
        return [origem]
    return sorted(p for p in origem.iterdir() if p.suffix.lower() in EXTENSOES)


def ao_vivo(model: YOLO, conf: float, imgsz: int, indice_camera: int = 0) -> None:
    """Mostra a webcam em tempo real com as caixas desenhadas. Tecla q sai."""
    cap = cv2.VideoCapture(indice_camera)
    if not cap.isOpened():
        raise RuntimeError("Nao foi possivel abrir a camera")

    titulo = "Deteccao ao vivo (tecla q para sair)"
    t_anterior = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            r = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
            imagem = r.plot()  # frame com caixas e rotulos desenhados

            agora = time.perf_counter()
            fps = 1 / (agora - t_anterior)
            t_anterior = agora
            cv2.putText(imagem, f"{fps:.1f} FPS | {len(r.boxes)} objetos", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(titulo, imagem)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("origem", nargs="?", type=Path, help="pasta ou arquivo de imagem")
    ap.add_argument("--camera", action="store_true", help="captura uma foto da webcam")
    ap.add_argument("--ao-vivo", action="store_true", help="webcam em tempo real")
    ap.add_argument("--indice", type=int, default=0, help="indice da camera (0, 1, 2...)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--modelo", default="yolov8n.pt")
    ap.add_argument("--saida", type=Path, default=Path("saida"))
    args = ap.parse_args()

    if args.ao_vivo:
        model = YOLO(args.modelo)  # baixa os pesos na primeira execucao
        ao_vivo(model, args.conf, args.imgsz, args.indice)
        return

    if args.camera:
        imagens = [capturar(Path("capturas"), args.indice)]
    elif args.origem:
        imagens = listar_imagens(args.origem)
    else:
        ap.error("informe uma pasta/arquivo ou use --camera ou --ao-vivo")

    if not imagens:
        raise SystemExit("Nenhuma imagem encontrada")

    args.saida.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.modelo)

    linhas = []
    for img in imagens:
        t0 = time.perf_counter()
        r = model(str(img), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000

        objetos = [model.names[int(c)] for c in r.boxes.cls]
        contagem = Counter(objetos)
        cv2.imwrite(str(args.saida / f"{img.stem}_det.jpg"), r.plot())

        resumo = ", ".join(f"{n}x {nome}" for nome, n in contagem.most_common()) or "nada detectado"
        print(f"{img.name}: {resumo} ({ms:.0f} ms)")
        linhas.append([img.name, len(objetos), resumo, round(ms)])

    caminho_csv = args.saida / f"relatorio_conf{args.conf}_img{args.imgsz}.csv"
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["foto", "total", "objetos", "ms"])
        w.writerows(linhas)
    print(f"\nRelatorio: {caminho_csv}")


if __name__ == "__main__":
    main()