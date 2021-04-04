import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    # def __init__(self, input_size, hidden_size, num_classes):
    #     super(NeuralNet, self).__init__()
    #     self.l1 = nn.Linear(input_size, hidden_size)
    #     self.l2 = nn.Linear(hidden_size, hidden_size)
    #     self.l3 = nn.Linear(hidden_size, num_classes)
    #     self.relu = nn.ReLU()

    # def forward(self, x):
    #     out = self.l1(x)
    #     out = self.relu(out)
    #     out = self.l2(out)
    #     out = self.relu(out)
    #     out = self.l3(out)
    #     # no activation and no softmax at the end
    #     return out

    def __init__(self, input_size, hidden_size, num_classes, hidden_layers):
        super(NeuralNet, self).__init__()

        layer_input_size = hidden_size
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for l in hidden_layers:
            print(l)
            if l['type'] == "ReLU":
                layers.append(nn.ReLU())
            elif l['type'] == "Linear":
                layers.append(nn.Linear(layer_input_size, l['output_size']))
                layer_input_size = l['output_size']

        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1]['output_size'], num_classes))
        self.linears = nn.ModuleList(layers)
        


    def forward(self, x):
        for m in self.linears:
            x = m(x)
        # no activation and no softmax at the end
        return x
