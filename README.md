# app_camera

# Reconhecimento de Objetos por Imagem

Aplicação desenvolvida em Python para reconhecer objetos em imagens utilizando inteligência artificial. O projeto também possui implementação para captura e análise de imagens por webcam.

## Sobre o projeto

O projeto tem como objetivo testar diferentes modelos de reconhecimento de objetos, avaliar seus resultados e explorar possibilidades de aplicação em situações reais.

Durante o desenvolvimento, foram realizados testes com diferentes modelos para comparar a precisão e o desempenho na identificação de objetos.

## Funcionalidades

- Reconhecimento de objetos em imagens.
- Captura de imagens por webcam.
- Testes com diferentes modelos de detecção.
- Geração de relatórios com os resultados das detecções.
- Análise das limitações e dos erros de reconhecimento.

## Tecnologias utilizadas

- Python
- YOLOv8 (Ultralytics)
- OpenCV

## Modelos testados

**YOLOv8n:** modelo mais leve, com foco em velocidade e menor consumo de recursos.

**YOLOv8m:** modelo maior, que apresentou melhorias em alguns testes, mas exige mais processamento.

Os resultados variam de acordo com a qualidade da imagem, o tamanho dos objetos e as classes que cada modelo consegue reconhecer.

## Limitações

A aplicação ainda pode apresentar erros de classificação, deixar de detectar objetos ou realizar contagens incorretas.

Os modelos padrão utilizados reconhecem um conjunto limitado de classes. Objetos que não fazem parte desse conjunto podem não ser identificados corretamente.

## Melhorias futuras

- Testar modelos de vocabulário aberto, como YOLO-World.
- Melhorar a precisão do reconhecimento e da contagem de objetos.
- Avaliar APIs de inteligência artificial para reconhecer objetos específicos.
- Implementar o uso da aplicação por meio de um celular.
- Melhorar o desempenho e a confiabilidade das detecções.

## Status do projeto

Em desenvolvimento. O projeto está na fase de prototipagem e testes, com foco no aprimoramento do reconhecimento de objetos por imagem e webcam.
