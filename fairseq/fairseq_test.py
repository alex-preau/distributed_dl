from torchvision.models import resnet50, resnet18
from torchvision import models
import torch
import pytorch_lightning as pl
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import time
import argparse
import transformers

parser = argparse.ArgumentParser(description='PyTorch Fairseq Timing')
#default batch size 64
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
#default epochs = 10
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')

parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

parser.add_argument('--fp16', action='store_true', default=False,
                    help='use fp16')

parser.add_argument('--gpus', type=int, default=2, metavar='N',
                    help='number of GPUs to use')

parser.add_argument('--resnet',type=int, default=18, metavar='N',
                    help='resnet version')



class ResNet50(pl.LightningModule):
    def __init__(self, num_classes, resnet_version,
                test_path=None, 
                optimizer='adam', lr=1e-3, 
                transfer=True, tune_fc_only=False):
        super().__init__()

        self.__dict__.update(locals())
        resnets = {
            18: models.resnet18, 34: models.resnet34,
            50: models.resnet50, 101: models.resnet101,
            152: models.resnet152
        }
        optimizers = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
        self.optimizer = optimizers[optimizer]
        #instantiate loss criterion
        self.criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
        # Using a pretrained ResNet backbone
        self.resnet_model = resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        if tune_fc_only: # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):

        return self.optimizer(self.trainer.model.parameters(), lr=self.lr)
    

    
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        
        loss = self.criterion(preds, y)

        acc = (y == torch.argmax(preds,1)) \
                .type(torch.FloatTensor).mean()
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    
    def validation_step(self, batch, batch_idx):

        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()

        loss = self.criterion(preds, y)
        acc = (y == torch.argmax(preds,1)) \
                .type(torch.FloatTensor).mean()
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)




if __name__ == '__main__':
    args = parser.parse_args()
    seed_everything(args.seed)

    model = ResNet50(10,args.resnet)
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.48232,), (0.23051,))
        ])
    img_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True,transform=transform)

    img_val =  torchvision.datasets.CIFAR10(root='./data', train=False,
                download=True, transform=transform)
    train_dl = DataLoader(img_train,num_workers=1,batch_size=args.batch_size,pin_memory=True)
    val_dl = DataLoader(img_val)

    trainer = pl.Trainer(strategy="fsdp_native", accelerator="cuda", devices=args.gpus,max_epochs=args.epochs) #strategy='fsdp_native'



    s = time.time()
    trainer.fit(model,train_dl)
    e = time.time()

    print('Net time:',e-s)
    print("Average samples/sec:",(len(img_train) * args.epochs) / (e-s))

