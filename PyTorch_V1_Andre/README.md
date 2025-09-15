# 🐟 Classificação de Imagens de Peixes com PyTorch

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

Este projeto implementa uma rede neural totalmente conectada (MLP) em PyTorch para a classificação de imagens de diferentes espécies de peixes. O sistema é capaz de treinar, validar e classificar novas imagens, e inclui um mecanismo de *early stopping* para otimizar o processo de treinamento.

## ✨ Principais Funcionalidades

- **Rede Neural Flexível**: Implementação de uma rede totalmente conectada (fully connected) para classificação.
- **Pré-processamento de Imagens**: Redimensionamento e conversão automática das imagens para tensores PyTorch.
- **Treinamento Otimizado**: Inclui *early stopping* para interromper o treinamento quando a acurácia de validação para de melhorar, economizando tempo e evitando overfitting.
- **Validação e Teste**: Scripts para carregar o dataset, dividir em treino/validação e avaliar a performance do modelo.
- **Classificação de Novas Imagens**: Função para classificar imagens individuais e visualizar os resultados com `matplotlib`.
- **Configuração Centralizada**: Todos os hiperparâmetros (épocas, taxa de aprendizado, etc.) são facilmente ajustáveis em um único arquivo de configuração.

## 📂 Estrutura de Pastas

O dataset deve ser organizado com uma subpasta para cada classe de peixe dentro dos diretórios `train` e `test`.

```
/
├── PyTorchRN.py           # Script principal (treino, validação e classificação)
├── Config_Parametros.py   # Configurações de hiperparâmetros e paths
├── Carregar_Banco.py      # Lógica para carregar os datasets
└── bd_peixes/
    ├── train/
    │   ├── especie_1/
    │   │   ├── img1.jpg
    │   │   └── ...
    │   └── especie_2/
    │       ├── imgA.jpg
    │       └── ...
    └── test/
        ├── especie_1/
        │   ├── img_val_1.jpg
        │   └── ...
        └── especie_2/
            ├── img_val_A.jpg
            └── ...
```

## 🚀 Começando

Siga os passos abaixo para executar o projeto localmente.

### Pré-requisitos

- Python 3.8 ou superior

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **Instale as dependências:**
    ```bash
    pip install torch torchvision matplotlib scikit-learn tensorboard
    ```

### Configuração

Ajuste os hiperparâmetros e as configurações do modelo no arquivo `Config_Parametros.py`:

```python
# Config_Parametros.py

epocas = 100             # Número máximo de épocas
tamanho_lote = 33        # Tamanho do batch (batch size)
taxa_aprendizagem = 0.001 # Taxa de aprendizado (learning rate)
tamanho_imagens = 64     # Redimensionar imagens para 64x64 pixels
perc_val = 0.21          # Percentual de imagens de treino para validação interna
paciencia = 5            # Épocas sem melhora antes de parar (early stopping)
tolerancia = 0.01        # Melhora mínima na acurácia para considerar progresso
nome_rede = "resnet"     # Nome base para salvar o modelo treinado
```

## ▶️ Como Executar

Para treinar o modelo e classificar algumas imagens de teste, execute o script principal:

```bash
python PyTorchRN.py
```

O script irá realizar as seguintes ações:
1.  Carregar e pré-processar as imagens das pastas `train` e `test`.
2.  Iniciar o ciclo de treinamento e validação.
3.  Aplicar o *early stopping* se a acurácia de validação não melhorar.
4.  Salvar o modelo com melhor desempenho como `modelo_treinado.pth`.
5.  Carregar o modelo salvo e classificar algumas imagens aleatórias do conjunto de teste, exibindo as previsões.

## 💡 Observações Importantes

-   O modelo atual é uma rede simples, totalmente conectada. Para datasets mais complexos ou imagens de alta resolução, considere o uso de **Redes Neurais Convolucionais (CNNs)** para obter melhores resultados.
-   Se você adicionar a normalização de imagens (`transforms.Normalize`) ao seu pipeline, lembre-se de **desnormalizar** os tensores antes de exibi-los com `matplotlib` para que as cores sejam exibidas corretamente.
