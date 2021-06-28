import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable

np.random.seed(324)
torch.manual_seed(32)

class ManualLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        # To make "a" and "b" real parameters of the model, we need to wrap them with nn.Parameter
        self.a = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        
    def forward(self, x):
        # Computes the outputs / predictions
        return self.a + self.b * x

parser = argparse.ArgumentParser(description='Generate ONNX model and test data')
parser.add_argument('--shape', type=int, nargs='+', default=[5, 3, 6, 8])
args = parser.parse_args()

model = ManualLinearRegression()
inp = Variable(torch.randn(args.shape))
model.eval()

with torch.no_grad():
    torch.onnx.export(model, inp, 'model.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

ref = model(inp)
np.save('inp', inp.detach().numpy())
np.save('ref', ref.detach().numpy())
