"""Servidor FastAPI: o celular abre a pagina, tira a foto e recebe os objetos detectados.

Rodar:
  uvicorn app:app --host 0.0.0.0 --port 8000
No celular (mesmo Wi-Fi): http://IP_DO_PC:8000
Documentacao automatica da API: http://IP_DO_PC:8000/docs
"""
import base64
import os
import threading
import time
from collections import Counter

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

MODELO = os.getenv("MODELO", "yolov8n.pt")
CONF = float(os.getenv("CONF", "0.4"))
IMGSZ = int(os.getenv("IMGSZ", "960"))
MAX_BYTES = 15 * 1024 * 1024
LARGURA_MAX_RESPOSTA = 1280

app = FastAPI(title="Detector de objetos na mesa")
model = YOLO(MODELO)  # carregado uma vez so, ao iniciar o servidor
trava = threading.Lock()  # um modelo, uma inferencia por vez

with trava:  # aquecimento: a primeira inferencia e sempre a mais lenta
    model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)

PAGINA = """<!doctype html>
<html lang="pt-br"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Detector de objetos</title>
<style>
body{font-family:system-ui,sans-serif;margin:0 auto;padding:16px;max-width:640px}
label.btn{display:block;padding:16px;font-size:18px;border-radius:10px;background:#1a73e8;color:#fff;text-align:center}
img{width:100%;border-radius:10px;margin-top:12px}
li{font-size:18px;margin:4px 0}
#msg{margin:12px 0;color:#555}
</style></head><body>
<h2>Detector de objetos</h2>
<label class="btn">Tirar foto da mesa
<input id="foto" type="file" accept="image/*" capture="environment" hidden></label>
<div id="msg"></div><ul id="lista"></ul><img id="resultado" alt="" hidden>
<script>
const fotoEl = document.getElementById('foto');
const msgEl = document.getElementById('msg');
const listaEl = document.getElementById('lista');
const imgEl = document.getElementById('resultado');
fotoEl.addEventListener('change', async () => {
  const arq = fotoEl.files[0];
  if (!arq) return;
  msgEl.textContent = 'Analisando...';
  listaEl.innerHTML = '';
  imgEl.hidden = true;
  const fd = new FormData();
  fd.append('arquivo', arq);
  try {
    const r = await fetch('/detectar', { method: 'POST', body: fd });
    if (!r.ok) throw new Error((await r.json()).detail || r.status);
    const d = await r.json();
    msgEl.textContent = d.total ? d.total + ' objeto(s) em ' + d.ms + ' ms' : 'Nada detectado';
    d.objetos.forEach(o => {
      const li = document.createElement('li');
      li.textContent = o.quantidade + 'x ' + o.nome;
      listaEl.appendChild(li);
    });
    imgEl.src = d.imagem;
    imgEl.hidden = false;
  } catch (e) {
    msgEl.textContent = 'Erro: ' + e.message;
  }
  fotoEl.value = '';
});
</script></body></html>"""


def _jpeg_base64(imagem: np.ndarray) -> str:
    h, w = imagem.shape[:2]
    if w > LARGURA_MAX_RESPOSTA:
        imagem = cv2.resize(imagem, (LARGURA_MAX_RESPOSTA, int(h * LARGURA_MAX_RESPOSTA / w)))
    _, buf = cv2.imencode(".jpg", imagem, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@app.get("/", response_class=HTMLResponse)
def pagina():
    return PAGINA


@app.get("/saude")
def saude():
    return {"ok": True, "modelo": MODELO, "conf": CONF, "imgsz": IMGSZ}


@app.post("/detectar")
def detectar(arquivo: UploadFile = File(...), com_imagem: bool = True):
    dados = arquivo.file.read(MAX_BYTES + 1)
    if len(dados) > MAX_BYTES:
        raise HTTPException(413, "Imagem maior que 15 MB")
    img = cv2.imdecode(np.frombuffer(dados, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Arquivo nao e uma imagem valida")

    t0 = time.perf_counter()
    with trava:
        r = model(img, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    ms = round((time.perf_counter() - t0) * 1000)

    deteccoes = [
        {
            "nome": model.names[int(c.cls[0])],
            "confianca": round(float(c.conf[0]), 3),
            "caixa": [round(float(v)) for v in c.xyxy[0].tolist()],
        }
        for c in r.boxes
    ]
    contagem = Counter(d["nome"] for d in deteccoes)
    resposta = {
        "total": len(deteccoes),
        "objetos": [{"nome": n, "quantidade": q} for n, q in contagem.most_common()],
        "deteccoes": deteccoes,
        "ms": ms,
    }
    if com_imagem:
        resposta["imagem"] = _jpeg_base64(r.plot())
    return resposta