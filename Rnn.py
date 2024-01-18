import torch
import torch.nn as nn
import torch.optim as optim

# Definir a arquitetura da rede neural
class RedeNeuralSimples(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RedeNeuralSimples, self).__init__()
        self.camada1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.camada2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.camada1(x)
        x = self.relu(x)
        x = self.camada2(x)
        return x

# Definir os parâmetros da rede
input_size = 10
hidden_size = 5
output_size = 1

# Criar uma instância da rede neural
rede_neural = RedeNeuralSimples(input_size, hidden_size, output_size)

# Definir uma amostra de entrada
entrada = torch.randn(1, input_size)

# Obter a saída da rede neural
saida = rede_neural(entrada)

# Imprimir a arquitetura da rede e a saída
print(rede_neural)
print("Saída da rede:", saida)
