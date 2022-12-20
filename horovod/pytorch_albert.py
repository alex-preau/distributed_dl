#https://github.com/horovod/horovod/issues/2103

import argparse
import time
import numpy as np
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torch.utils.data.distributed
import horovod.torch as hvd
from torchvision import models
from pytorch_lightning import LightningDataModule, LightningModule #perhaps remove
from datetime import datetime
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import datasets
import pandas as pd

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example')
#default batch size 64
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
#default epochs = 10
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=2000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')

parser.add_argument('--all', action='store_true', default=False,
                    help='Run all configurations and save results')


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
        return DataLoader(self.dataset['train'], batch_size=self.train_batch_size, shuffle=True,num_workers=2,pin_memory=True)

    def len(self):
        return self.dataset['train'].__len__()



    def convert_to_features(self, example_batch, indices=None):


        texts_or_text_pairs = example_batch['sentence']


        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]
       # print(features)
        return features

class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        lr_scalar: float, #horovod specific
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

    def training_step(self, batch):
        outputs = self(**batch)
        loss = outputs[0]
        return loss


    def configure_optimizers(self):


        optimizer = AdamW(self.model.parameters(), lr=self.hparams.learning_rate * self.hparams.lr_scalar)


        return optimizer                    

def train(epoch):
    model.train()
    # Horovod: set epoch to sampler for shuffling.
    train_sampler.set_epoch(epoch)
    for batch_idx, batch in enumerate(train_loader):

        batch['labels'] = batch['labels'].cuda()
        batch['input_ids'] = batch['input_ids'].cuda()
        batch['token_type_ids'] = batch['token_type_ids'].cuda()
        batch['attention_mask'] = batch['attention_mask'].cuda()

        optimizer.zero_grad()

        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # Horovod: use train_sampler to determine the number of examples in
            # this worker's partition.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(batch['labels'] ), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def test():
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for batch in test_loader:
        batch['labels'] = batch['labels'].cuda()
        batch['input_ids'] = batch['input_ids'].cuda()
        batch['token_type_ids'] = batch['token_type_ids'].cuda()
        batch['attention_mask'] = batch['attention_mask'].cuda()
        print(batch)
        test_loss += model.training_step(batch).item()
        # sum up batch loss

        

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)


    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')


    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}\n'.format(
            test_loss))


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    hvd.init()
    # Horovod: initialize library.
    torch.manual_seed(args.seed)

    if args.all:
        timing_dict = {'Model':[],'Batch':[],'Precision':[],'#GPUs':[],'Samples/Sec':[]}
        #we are only able to iterate with one gpu setting bc of how horovod works

        for bs in [2,4,8,16,32,64,96]:#2,64,128,256,
            for fp in [False]:
                if args.cuda:
                    # Horovod: pin GPU to local rank.
                    torch.cuda.set_device(hvd.local_rank())
                    torch.cuda.manual_seed(args.seed)


                # Horovod: limit # of CPU threads to be used per worker.
                torch.set_num_threads(1)

                kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
                # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
                # issues with Infiniband implementations that are not fork-safe
                if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
                        mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
                    kwargs['multiprocessing_context'] = 'forkserver'

                data_module = GLUEDataModule(train_batch_size=bs)
                data_module.setup("fit")
                train_dataset = data_module.dataset['train']



                # Horovod: use DistributedSampler to partition the training data.
                train_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=bs, sampler=train_sampler, **kwargs)

                test_dataset = data_module.dataset['test']
                # Horovod: use DistributedSampler to partition the test data.
                test_sampler = torch.utils.data.distributed.DistributedSampler(
                    test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
                test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs,
                                                            sampler=test_sampler, **kwargs)



                # By default, Adasum doesn't need scaling up learning rate.
                lr_scaler = hvd.size() if not args.use_adasum else 1

                model = GLUETransformer(
                        lr_scalar=lr_scaler,
                        model_name_or_path="albert-base-v2",
                        num_labels=data_module.num_labels,
                        task_name=data_module.task_name,
                    )

                if args.cuda:
                    # Move model to GPU.
                    model.cuda()
                    # If using GPU Adasum allreduce, scale learning rate by local_size.
                    if args.use_adasum and hvd.nccl_built():
                        lr_scaler = hvd.local_size()

                # Horovod: scale learning rate by lr_scaler.
                optimizer = model.configure_optimizers()


                # Horovod: broadcast parameters & optimizer state.
                hvd.broadcast_parameters(model.state_dict(), root_rank=0)
                hvd.broadcast_optimizer_state(optimizer, root_rank=0)

                # Horovod: (optional) compression algorithm.

                compression = hvd.Compression.fp16 if fp else hvd.Compression.none

                # Horovod: wrap optimizer with DistributedOptimizer.
                optimizer = hvd.DistributedOptimizer(optimizer,
                                                        named_parameters=model.named_parameters(),
                                                        compression=compression,
                                                        op=hvd.Adasum if args.use_adasum else hvd.Average)

                img_per_secs = []
                for epoch in range(1, args.epochs + 1):
                    start_time = time.time()
                    train(epoch)
                    epoch_time = time.time() - start_time
                    img_per_secs.append(len(train_dataset)/epoch_time)
                    if hvd.rank() == 0:
                        print('Average samples/sec: {}'.format(img_per_secs[-1]))
                  #  test()
                mean_img_per_secs= np.mean(img_per_secs)
                if hvd.rank() == 0:
                    print('Average samples/sec: {}'.format(mean_img_per_secs))
                    print('Average samples/sec per gpu: {}'.format(mean_img_per_secs/hvd.size()))
                    timing_dict['Model'].append('alBERT')
                    timing_dict['Batch'].append(bs)
                    if fp:
                        timing_dict['Precision'].append(16)
                    else:
                        timing_dict['Precision'].append(32)
                    timing_dict['#GPUs'].append('2')
                    timing_dict['Samples/Sec'].append(mean_img_per_secs)
                    print("Complete Timing Dict")
                    out_data = pd.DataFrame(data=timing_dict)
                    out_data.to_csv('horovod_timing_albert_1gpus.csv')

    

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)


    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    data_module = GLUEDataModule(train_batch_size=args.batch_size)
    data_module.setup("fit")
    train_dataset = data_module.dataset['train']
    
    

    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    test_dataset = data_module.dataset['test']
    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              sampler=test_sampler, **kwargs)



    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    model = GLUETransformer(
            lr_scalar=lr_scaler,
            model_name_or_path="albert-base-v2",
            num_labels=data_module.num_labels,
            task_name=data_module.task_name,
        )

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = model.configure_optimizers()


    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Adasum if args.use_adasum else hvd.Average)

    img_per_secs = []
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(epoch)
        epoch_time = time.time() - start_time
        img_per_secs.append(len(train_dataset)/epoch_time)
        if hvd.rank() == 0:
            print('Average samples/sec: {}'.format(img_per_secs[-1]))
      #  test()
    mean_img_per_secs= np.mean(img_per_secs)
    if hvd.rank() == 0:
        print('Average samples/sec: {}'.format(mean_img_per_secs))
        print('Average samples/sec per gpu: {}'.format(mean_img_per_secs/hvd.size()))