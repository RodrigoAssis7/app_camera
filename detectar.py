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
    for _ in range(10):  # descarta os primeiros frames
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("origem", nargs="?", type=Path)
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--modelo", default="yolov8n.pt")
    ap.add_argument("--saida", type=Path, default=Path("saida"))
    args = ap.parse_args()

    if args.camera:
        imagens = [capturar(Path("capturas"))]
    elif args.origem:
        imagens = listar_imagens(args.origem)
    else:
        ap.error("informe uma pasta/arquivo ou use --camera")

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