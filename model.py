import torch
import torch.optim as optim
import torchvision


def build_model(path=None):
    model = torchvision.models.resnet101(num_classes=20)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    if path is not None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Loading {}...'.format(path))

    return model, optimizer