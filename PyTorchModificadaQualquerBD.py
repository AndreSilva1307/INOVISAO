# Vai baixar o banco de imagens de treino e de teste do exemplo com peixes
!curl -L -o v4_train_test.zip "https://drive.google.com/uc?export=download&id=1aW5so-0XAvXlWzpKkvsergw6907Vb7JE"
!mkdir ./data/
!mv v4*.zip ./data/
%cd ./data/
!unzip v4*.zip
%cd ..

pasta_base = "./"
pasta_data = pasta_base+"data/"
print("Realizando leitura de imagens do repositório: ",pasta_data)
pasta_treino = pasta_data+"train"
pasta_teste  = pasta_data+"test"



import torch   # Biblioteca pytorch principal
from torch import nn  # Módulo para redes neurais (neural networks)
from torch.utils.data import DataLoader # Manipulação de bancos de imagens
from torchvision import datasets, models # Ajuda a importar alguns bancos já prontos e famosos
from torchvision.transforms import ToTensor # Realiza transformações nas imagens
import matplotlib.pyplot as plt # Mostra imagens e gráficos
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter # Salva "log" da aprendizagem

# Definindo alguns hiperparâmetros importantes:
epocas = 15  # Total de passagens durante a aprendizagem pelo conjunto de imagens
tamanho_lote = 5  # Tamanho de cada lote sobre o qual é calculado o gradiente
taxa_aprendizagem = 0.01   # Magnitude das alterações nos pesos

nome_rede = "resnet" # Define uma arquitetura já conhecida que será usada
tamanho_imagens = 100
perc_val = 0.2  # Percentual do treinamento a ser usado para validação

transform = transforms.Compose([transforms.Resize((tamanho_imagens,tamanho_imagens)),
                                transforms.ToTensor(),])
training_val_data = datasets.ImageFolder(root=pasta_treino,transform=transform) # Prepara banco de imagens de treino 
test_data = datasets.ImageFolder(root=pasta_teste,transform=transform) # Prepara banco de imagens de teste

# Aqui vai separar em treinamento e validação
train_idx, val_idx = train_test_split(list(range(len(training_val_data))), test_size=perc_val)
training_data = Subset(training_val_data, train_idx)
val_data = Subset(training_val_data, val_idx)

# Cria os objetos
train_dataloader = DataLoader(training_data, batch_size=tamanho_lote,shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=tamanho_lote,shuffle=True)

for X, y in val_dataloader:
    print(f"Tamanho do lote de imagens: {X.shape[0]}")
    print(f"Quantidade de canais: {X.shape[1]}")
    print(f"Altura de cada imagem: {X.shape[2]}")
    print(f"Largura de cada imagem: {X.shape[3]}")
    print(f"Tamanho do lote de classes (labels): {y.shape[0]}")
    print(f"Tipo de cada classe: {y.dtype}")
    break  # Para depois de mostrar os dados do primeiro lote

print(f"\nTotal de imagens de treinamento: {len(training_data)}")
print(f"Total de imagens de validação: {len(val_data)}")
labels_map = {v: k for k, v in test_data.class_to_idx.items()}
print('\nClasses:',labels_map)


# Verifica se tem GPU, ou usa a CPU mesmo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n\nUsando {device}")

total_classes = len(labels_map)
tamanho_entrada_flatten = tamanho_imagens * tamanho_imagens * 3 # Tamanho de entrada para imagens RGB

class NeuralNetwork(nn.Module):
    def __init__(self):


        super(NeuralNetwork, self).__init__() # Inicializa a classe "pai"
        self.flatten = nn.Flatten() # Transforma a imagem para 1 dimensão

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(tamanho_entrada_flatten, 512), # Corrigido o tamanho de entrada
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, total_classes)) # Alterado o número de neurônios de saída para o número de classes

    # Define como funcionará o forward
    def forward(self, x):
        # Realiza o achatamento do tensor
        # Transforma um matriz 28*28 em um vetor com 784 posições
        x = self.flatten(x)
        # Ativação dos neurônios
        output_values = self.linear_relu_stack(x)
        return output_values

model = NeuralNetwork().to(device) # Prepara a rede para o dispositivo
print(model) # Imprime dados sobre a arquitetura da rede

otimizador = torch.optim.SGD(model.parameters(), lr=taxa_aprendizagem) # Define o otimizador como sendo descida de gradiente estocástica
funcao_perda = nn.CrossEntropyLoss() # Define a função de perda como entropia cruzada

# Define a função para treinar a rede

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    # Pega um lote de imagens de cada vez do conjunto de treinamento
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)  # Prepara os dados para o dispositivo (GPU ou CPU)
        pred = model(X)         # Realiza uma previsão usando os pesos atuais
        loss = loss_fn(pred, y) # Calcula o erro com os pesos atuais

        optimizer.zero_grad()  # Zera os gradientes pois vai acumular para todas
                               # as imagens do lote
        loss.backward()        # Retropropaga o gradiente do erro
        optimizer.step()       # e recalcula todos os pesos da rede

        # Imprime informação a cada 100 lotes processados 
        if batch % 100 == 0:
            # Mostra a perda e o total de imagens já processadas
            loss, current = loss.item(), batch * len(X)
            print(f"Perda: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Define a função de validação
def validation(dataloader, model, loss_fn):
    size = len(dataloader.dataset)  # Total de imagens para validação
    num_batches = len(dataloader)   # Total de lotes
    model.eval()  # Coloca a rede em modo de avaliação (e não de aprendizagem)
    # Vai calcular o erro no conjunto de validação
    val_loss, correct = 0, 0

    # Na validação os pesos não são ajustados e por isso não precisa
    # calcular o gradiente
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    acuracia = correct / size
    print("Informações na Validação:")
    print(f"Total de acertos: {int(correct)}")
    print(f"Total de imagens: {size}")
    print(f"Perda média: {val_loss:>8f}")            
    print(f"Acurácia: {(100*acuracia):>0.1f}%")

# Treinando a rede neural, passando as todas as imagens varias vezes pela quantidade de épocas

for t in range(epocas):
    print(f"-------------------------------")
    print(f"Época {t+1}\n-------------------------------")
    train(train_dataloader, model, funcao_perda, otimizador)
    validation(val_dataloader, model, funcao_perda)

print("Terminou a fase de aprendizagem !")

# # Salva em disco os pesos da rede treinada para ser usada novamente 
torch.save(model.state_dict(), "modelo_treinado.pth")
print("Salvou o modelo treinado em modelo_treinado.pth")

# Carregando a rede neural treinada salva anteriormente
model = NeuralNetwork()
model.load_state_dict(torch.load("modelo_treinado.pth"))

# Classifica uma imagem
def classifica_uma_imagem(model,x,y):
    model.eval()
    with torch.no_grad():
       # Add a batch dimension to the image
       x = x.unsqueeze(0)
       pred = model(x)
       predita, real = labels_map[int(pred[0].argmax(0))], labels_map[y]
       print(f'Predita: "{predita}", Real: "{real}"')
    return(predita)

figure = plt.figure(figsize=(10, 10))
cols, rows = 3, 4
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() # Número randomico menor que a quantidade de imagens
    img, label = training_data[sample_idx] # Pega a imagem e classificação usando o número randomico
    predita = classifica_uma_imagem(model,img,label) # Classifica usando a rede treinada
    figure.add_subplot(rows, cols, i) # Adiciona a imagem na grade que será mostrada
    plt.title(predita) # Usa a classe da imagem como título da imagem
    plt.axis("off") # Não mostra valores nos eixos X e Y
    plt.imshow(img.permute(1,2,0)) # Ajusta a ordem das dimensões do tensor
plt.show() # Este é o comando que vai mostrar as imagens
