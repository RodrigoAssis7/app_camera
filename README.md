# app_camera

Detector de objetos em imagens com Python, YOLOv8, OpenCV e FastAPI. Tire uma foto da mesa pelo celular, o servidor identifica os objetos e devolve a imagem com as caixas e a contagem. Roda no computador, na rede local e na nuvem, de graça.

[Demonstração: link do Render e prints da página no celular]

## Destaques

- Do protótipo ao app publicado: script de linha de comando, webcam ao vivo, API REST e página web para celular.
- Rodando no plano gratuito do Render (512 MB de RAM) depois de um trabalho real de otimização de memória.
- Comparação medida entre dois modelos (`yolov8n` e `yolov8m`) em fotos reais, com limites documentados.
- Relatório em CSV, imagens anotadas e API com documentação interativa em `/docs`.

## Como funciona

```
Celular (página web) -> reduz a foto -> POST /detectar -> FastAPI + YOLOv8 -> JSON + imagem anotada
```

O mesmo núcleo de detecção atende o script (`detectar.py`), o modo ao vivo e a API (`app.py`).

## Funcionalidades

- Detecção em uma foto ou em uma pasta, com imagens anotadas e CSV.
- Captura pela webcam e detecção em tempo real, com FPS e contagem na tela.
- Endpoint que devolve objetos, quantidades, caixas e a imagem anotada.
- Página para o celular tirar a foto e ver o resultado.
- Escolha do modelo, da confiança mínima e da resolução.

## Tecnologias

Python, YOLOv8 (Ultralytics), OpenCV, FastAPI, Uvicorn e PyTorch (CPU).

## Instalação

```
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Os pesos do modelo são baixados na primeira execução.

## Como usar

```
python detectar.py fotos                  # todas as fotos da pasta
python detectar.py --camera               # uma foto da webcam
python detectar.py --ao-vivo              # webcam em tempo real (q sai)
python detectar.py fotos --modelo yolov8m.pt --conf 0.4 --imgsz 1280
```

API e celular na rede local:

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

Com o celular no mesmo Wi-Fi, abra `http://IP_DO_COMPUTADOR:8000` e toque em "Tirar foto da mesa". O modelo muda pela variável `MODELO` (ex.: `MODELO=yolov8m.pt`). Se a página não abrir, libere a porta 8000 no firewall para a rede local.

Endpoints: `GET /` (página), `POST /detectar` (detecção em JSON), `GET /saude` (modelo e parâmetros em uso).

## Publicação no Render

O app roda no plano gratuito com o `yolov8n`. O servidor chegou a ser encerrado por falta de memória, e o problema foi resolvido assim:

- Inferência em `imgsz=640`.
- Foto reduzida no celular antes do envio e de novo no servidor.
- PyTorch na versão CPU no `requirements.txt`, bem menor que a versão com CUDA.

O diagnóstico partiu dos logs do Render, sem alterar código às cegas. No plano gratuito o serviço entra em repouso sem uso, então a primeira requisição demora mais. Para o `yolov8m`, o ideal é rodar localmente.

O que os testes mostraram:

- O modelo maior detecta melhor os objetos que conhece, principalmente em imagens pequenas.
- Fora das 80 classes do COCO, o modelo não responde "não sei": escolhe a classe mais parecida e costuma errar.
- Baixar a confiança mínima piorou o resultado, com mais classificações erradas.
- Uma mão segurando o objeto pode gerar um falso `person`.
- Em CPU, o `yolov8m` leva cerca de 1 a 2,5 s por foto.

Conclusão: modelos prontos resolvem bem objetos comuns, mas objetos específicos pedem outra abordagem, o que define os próximos passos.

## Limitações

- Apenas as 80 classes do COCO nos modelos padrão.
- A contagem pode errar com objetos próximos, cobertos ou pequenos.
- No Render gratuito só cabe o `yolov8n`.
- A API não tem autenticação, então publicada ela fica aberta a quem tiver o link.
- A Ultralytics usa licença AGPL-3.0, o que exige atenção em uso comercial.

## Próximos passos

- Testar o YOLO-World, que permite definir os objetos procurados.
- Avaliar uma API de IA com visão para objetos específicos.
- Ignorar classes indesejadas, como `person`, e adicionar autenticação.

## Status

Protótipo funcional e publicado: detecção em fotos, webcam e celular, na rede local e na nuvem.
