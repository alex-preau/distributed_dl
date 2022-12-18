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
from datetime import datetime
from typing import Optional
import pandas as pd
import datasets
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import transformers
import logging
logging.getLogger('lightning').setLevel(0)

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

parser.add_argument('--all',type=bool, default=False,
                    help='Run all options (ignore others) and print dataframe')





class GLUEDataModule(LightningDataModule):



    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name = "albert-base-v2"
        self.task_name = 'cola'
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size


        self.text_fields = ['sentence']
        self.num_labels = 2#
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", "cola") #load cola dataset from glue task

        #bro into test, train, val

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)


    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.train_batch_size, shuffle=True,num_workers=1)

    def len(self):
        return self.dataset['train'].__len__()



    def convert_to_features(self, example_batch, indices=None):


        texts_or_text_pairs = example_batch['sentence']


        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features

class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        train_batch_size: int = 32,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = datasets.load_metric(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss


    def configure_optimizers(self):


        optimizer = AdamW(self.trainer.model.parameters(), lr=self.hparams.learning_rate)


        return optimizer


if __name__ == '__main__':
    args = parser.parse_args()

    timing_dict = {'Model':[],'Batch':[],'Precision':[],'#GPUs':[],'Samples/Sec':[]}
    seed_everything(args.seed)
    if args.all:
        for m_type in [50]:
            for bs in [2,4,8,16,32,64]:#2,64,128,256,
                for fp in [32,16]:
                    for gpus in [1,2]:


                        dm = GLUEDataModule( train_batch_size=bs)
                        dm.setup("fit")
                        model = GLUETransformer(
                            model_name_or_path="albert-base-v2",
                            num_labels=dm.num_labels,
                            task_name=dm.task_name,
                        )

                        trainer = Trainer(
                            strategy="fsdp_native", accelerator="cuda", 
                            max_epochs=1,
                            precision=fp,
                            devices=gpus # limiting got iPython runs
                        )



                        s = time.time()
                        trainer.fit(model, datamodule=dm)
                        e = time.time()

                        print('Net time:',e-s)
                        print("Average samples/sec:",(dm.len() * args.epochs) / (e-s))
                        timing_dict['Model'].append('alBERT')
                        timing_dict['Batch'].append(bs)
                        timing_dict['Precision'].append(fp)
                        timing_dict['#GPUs'].append(gpus)
                        timing_dict['Samples/Sec'].append((dm.len() * args.epochs) / (e-s))
                        print("Complete Timing Dict")
                        out_data = pd.DataFrame(data=timing_dict)
                        out_data.to_csv('FSDP_timing_albert.csv')
    else:
        dm = GLUEDataModule( train_batch_size=64)
        dm.setup("fit")
        model = GLUETransformer(
            model_name_or_path="albert-base-v2",
            num_labels=dm.num_labels,
            task_name=dm.task_name,
        )

        trainer = Trainer(
            strategy="fsdp_native", accelerator="cuda", 
            max_epochs=1,

            devices=args.gpus if torch.cuda.is_available() else None,  # limiting got iPython runs
        )
        

        s = time.time()
        trainer.fit(model, datamodule=dm)
        e = time.time()

        print('Batch Size,')
        print('Net time:',e-s)
        print("Average samples/sec:",(dm.len() * args.epochs) / (e-s))

# 64 batch size with fsdp and 2 gpus 0.03136875032860385 samples/ second
# 32 batch size with fsdp and 2 gpus 0.03001464685890624samples/ second 
# 64 batch size with ddp and 2 gpus  0.03020631636905007samples/ second 
# 64 batch size with ddp_sharded and 2 gpus    Average samples/sec: 0.031204948981299965 Average samples/sec: 0.029213370824188692 (diff for each gpu)
# 64 batch size with ddp_sharde_spawn and 2 gpus    Average samples/sec: 0.02956397669445762

#64 batch with fsdp and 1 gpu Average samples/sec: 0.01714749563701319
