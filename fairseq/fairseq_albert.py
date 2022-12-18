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

parser.add_argument('--gpus', type=int, default=1, metavar='N',
                    help='number of GPUs to use')





class GLUEDataModule(LightningDataModule):

    task_text_field_map = {
        "cola": ["sentence"],
        "sst2": ["sentence"],
        "mrpc": ["sentence1", "sentence2"],
        "qqp": ["question1", "question2"],
        "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        "qnli": ["question", "sentence"],
        "rte": ["sentence1", "sentence2"],
        "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        "cola": 2,
        "sst2": 2,
        "mrpc": 2,
        "qqp": 2,
        "stsb": 1,
        "mnli": 3,
        "qnli": 2,
        "rte": 2,
        "wnli": 2,
        "ax": 3,
    }

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
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size


        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True)

    def len(self):
        return self.dataset.__len__()


    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
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


    seed_everything(args.seed)

    dm = GLUEDataModule(model_name_or_path="albert-base-v2", task_name="cola",train_batch_size=64)
    dm.setup("fit")
    model = GLUETransformer(
        model_name_or_path="albert-base-v2",
        num_labels=dm.num_labels,
        task_name=dm.task_name,
    )

    trainer = Trainer(
        strategy="fsdp", accelerator="cuda", 
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
