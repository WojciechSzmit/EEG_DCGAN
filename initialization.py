import torch.nn as nn

"""
 Inicjalizuje wagi modeli zgodnie z zaleceniami z artykułu DCGAN.
 """
def initialize_weights(model):

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)