# app_camera

Detector de objetos em imagens feito em Python com YOLOv8 (Ultralytics), OpenCV e FastAPI. Funciona em fotos, na webcam e pelo celular, por meio de uma página web que tira a foto e mostra os objetos encontrados.

O projeto começou como um protótipo para avaliar se um modelo pronto consegue identificar os objetos sobre uma mesa a partir de uma foto, e onde estão os limites dessa abordagem.

## Funcionalidades

- Detecção em uma foto ou em uma pasta de fotos, com imagens anotadas e relatório em CSV.
- Captura de foto pela webcam e detecção ao vivo, com FPS e contagem de objetos na tela.
- API em FastAPI com endpoint de detecção que devolve JSON (objetos, quantidades, caixas e imagem anotada).
- Página web simples para o celular tirar a foto e ver o resultado, usando a rede local.
- Escolha do modelo (`yolov8n` ou `yolov8m`), da confiança mínima e da resolução.

## Tecnologias

Python, YOLOv8 (Ultralytics), OpenCV, FastAPI e Uvicorn.

## Instalação

```
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Os pesos do modelo são baixados automaticamente na primeira execução.

## Como usar

Fotos e webcam:

```
python detectar.py fotos                  # todas as fotos da pasta
python detectar.py --camera               # captura uma foto da webcam e detecta
python detectar.py --ao-vivo              # webcam em tempo real (tecla q sai)
python detectar.py fotos --modelo yolov8m.pt --conf 0.4 --imgsz 1280
```

Para cada foto, o terminal mostra os objetos e o tempo de processamento, a imagem com as caixas vai para `saida/` e um CSV resume os resultados.

API e celular:

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

Com o celular no mesmo Wi-Fi, abra `http://IP_DO_COMPUTADOR:8000` e toque em "Tirar foto da mesa". A documentação interativa fica em `/docs`. O modelo usado pode ser trocado pela variável `MODELO` (por exemplo `MODELO=yolov8m.pt`). Se a página não abrir no celular, verifique o firewall do computador e a porta 8000.

Endpoints:

- `GET /`: página para o celular.
- `POST /detectar`: recebe a imagem e devolve o resultado em JSON.
- `GET /saude`: modelo e parâmetros em uso.

## Resultados dos testes

Testes com 11 imagens, comparando os dois modelos. A amostra é pequena e mostra tendências, não uma taxa de acerto.

| Objeto | yolov8n | yolov8m |
|---|---|---|
| Caneca | certo | certo |
| Despertador | certo | certo |
| Tesoura | erro | certo |
| Grampeador | nada detectado | erro |
| Chave | erro | erro |
| Clipes de papel | erro | erro |

- O modelo maior melhora a detecção de objetos que ele conhece, principalmente em imagens pequenas.
- Objetos fora das classes do modelo não recebem "não sei" como resposta: ele escolhe a classe mais parecida, e costuma errar.
- Baixar a confiança mínima piorou os resultados, com mais classificações erradas.
- Uma mão segurando um objeto pode gerar um falso `person`.
- Em CPU, o `yolov8m` leva em torno de 1 a 2,5 segundos por foto.

## Limitações

- Os modelos padrão reconhecem apenas as 80 classes do conjunto COCO.
- A contagem pode errar com objetos próximos, parcialmente cobertos ou pequenos.
- A API não tem autenticação e deve ser usada só em rede de confiança.
- A biblioteca Ultralytics usa a licença AGPL-3.0, o que exige atenção em uso comercial.

## Próximos passos

- Testar o YOLO-World, que permite definir os objetos procurados.
- Avaliar uma API de IA com visão para objetos específicos.
- Ignorar classes indesejadas, como `person`, e adicionar autenticação à API.

## Status

Protótipo funcional: detecção em fotos, na webcam e pelo celular na rede local, em avaliação para definir o modelo mais adequado.
