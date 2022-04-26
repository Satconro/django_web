import torch
from model import Enhancement_Encoder, Enhancement_Decoder


def load_checkpoint(model, checkpoint_file, device):
    print("=> Loading checkpoint from {}".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint)

# 转换model+optimizer的存储为model
def load_checkpoint(model, checkpoint_file, device):
    print("=> Loading checkpoint from {}".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model'])


def checkpoint_cast():
    encoder = Enhancement_Encoder()
    decoder = Enhancement_Decoder()
    load_checkpoint(model=encoder, checkpoint_file='weights/encoder.ckpt', device='cpu')
    load_checkpoint(model=decoder, checkpoint_file='weights/decoder.ckpt', device='cpu')
    torch.save(encoder.state_dict(), 'weights/encoder.ckpt')
    torch.save(decoder.state_dict(), 'weights/decoder.ckpt')