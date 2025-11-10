import torch
from torchviz import make_dot
from models.simple_cnn import Simple_CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Simple_CNN()
model = model.to(device)
model.eval()  # Set to evaluation mode for BatchNorm layers
data = torch.randn(1, 1, 28, 28).to(device)

output = model(data)

image = make_dot(output, params=dict(model.named_parameters()))
image.format = "png"
image.render("Simple_CNN_graph")

print("モデルのグラフ 'Simple_CNN_graph.png' が保存されました。")
