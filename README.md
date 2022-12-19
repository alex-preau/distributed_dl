
<div align="center">
<img src="./img_src/Columbia-Lions-Logo.png" width="200" />
</a>
</div>

# COMS-6998: Distributed Training
#### Alex Preau (awp2128) and Skyler Szot (sls2305)

<div align="center">
<img src="./img_src/resnet18_vid04_tissue_input.gif" width="200"> <img src="./img_src/resnet18_vid04_instrument_cam.gif" width="200"> <img src="./img_src/resnet18_vid04_instrument.gif" width="200">
</div>  
<div align="center">
<img src="./img_src/resnet18_vid04_tissue_input.gif" width="200"> <img src="./img_src/resnet18_vid04_tissue_cam.gif" width="200"> <img src="./img_src/resnet18_vid04_tissue.gif" width="200">
</div>  

## I. Project Description

Training a deep learning model to solve a real world problem requires a lot of data, which can be very time consuming. To ensure better efficiency and less time consumption, researchers can leverage parallel training approaches. We will be comparing various methods of parallel training of deep learning models in PyTorch. There are many proposed methods for this problem, from SIMD-like methods of averaging the gradients of batches computed on each independent GPU, to sharding and computing the parallel components of the computational graph independently. In this project, we will be comparing the advantages and disadvantages of these distributed training approaches.

#### Objective 1: PyTorch Baseline

This project aims to investigate various distributed training techniques for deep learning training. Our first goal is to implement and train various models in PyTorch as a baseline, without any distributed training optimizations. The models investigated in this report are ResNet-18, ResNet-50, and the ALBERT transformer model. For the ResNet-18 and ResNet-50 models we utilized the CIFAR-10 dataset, and for alBERT we utilized the General Language Understanding Evaluation (GLUE) language classification COLA dataset set.

#### Objective 2: Horovod Distributed Training

Our second objective is to implement the same models PyTorch, but now with distributed gradient computation and averaging across multiple GPUs using Horovod [1]. Horovod is a distributed deep learning training framework for PyTorch, which makes distributed deep learning fast and easy to use. Each GPU trains on a subset of data, and gradients are synchronized with an allreduce or allgather step [1]. We will compare the distributed training results with the baseline to investigate which cases benefit from multi-GPU training.

<div align="center">
<img src="./img_src/horovod.PNG" width="800">
</div>

#### Objective 3: FairScale

Our second objective is using Fair Scale to train the same models with fully sharded data parallelism on multiple GPUs. This is an advanced form of Data Parallelism in which data is split between GPUs as in distributred data parallelism, but the model itself is also sharded onto multiple GPUs. The weights are shared among shards via an allgather step and the foreward pass is calcualted. Allgather is performed again to allow the backward pass to take place, and then the gradients are spread to other GPUs and average via a reduce-scatter operation. The sharding of models is particularly optimized for models which are too large to fit in a single GPU memory, such as modern large language models. We will compare the performance of this optimized training versus the baseline and other models. 


<div align="center">
<img src="./img_src/FSDP.png" width="800">
</div>

#### Thrust 4: Pipelining

Our final objective is to utilize a pipeline parallelism approach, using the inbuilt Pipe APIs in PyTorch. Typically for large models which don’t fit on a single GPU, model parallelism is employed where certain parts of the model are placed on different GPUs. Although, if this is done naively for sequential models, the training process suffers from GPU under utilization since only one GPU is active at one time as shown in the figure below. To alleviate this problem, pipeline parallelism splits the input minibatch into multiple microbatches and pipelines the execution of these microbatches across multiple GPUs.

<div align="center">
<img src="./img_src/pipeline.png" width="800">
</div>

## II. Repository Description

The repsitory is a fork of [2] with additions to the data handling module, deep learning module, and main execution module. Moreover, custom scripts were added for class activation mapping and further model characterization. The basis structure of the repository is below. The `__checkpoint__` folder is used to automatically store the `.log` files recording the results and the `.pth` trained model files with examples shown for two separate test runs `100` and `101`. The results from the `.log` file are more conveniently stored in a `seaborn`-friendly csv format in the `pytorch/results/` folder, with examples shown from two tests. 

```
distributed_dl/
├── fairseq
│   ├── fairseq_albert.py
│   ├── fairseq_test.py
├── horovod
│   ├── pytorch_resnet_cifar10.py
├── pipeline
│   ├── pipeline_tutorial.py

```

The repository is organized into folders for each of our model objectives. All experiments were conducted using PyTorch and each requires a separate setup procedure.

## III. Example Commands

### Horovod

#### System Requirements

The Horovod experiments were conducted on a Google Cloud instance with two Tesla T4 GPUs. In order to create an instance with multiple GPUs, you must make a quota increase request. The Google Cloud instance was initialized with boot disk Debian 10 based Deep Learning VM for PyTorch CPU/GPU with CUDA 11.3 M102. This comes with PyTorch 1.12 and CUDA 11.3 pre-installed.

Horovod can be installed with:
```
pip install horovod
```

#### Example run command
Horovod can be run with the following command, specifying the number of GPUs to use.
```
horovodrun -np 2 python pytorch_resnet_cifar10.py
```

### FairScale 
FairScale can be installed with
```
pip install fairscale
```

Pytorch Lightning can be installed with 
```
pip install pytorch-lightning

```
#### Example run command
FairScale Resnet 18 can be run with the following command, specifcying number of GPUs to use, if FP16 should be used
NOTE only ResNet18 and ResNet50 are supported
```
python fairseq_resnet.py --resnet 18 --fp False --gpus 2
```

FairScale alBERT can be run with the followign command, specifying the number of GPUs to use and if FP16 should be used
```
python fairseq_albert.py --fp False --gpus 2
```
### Pipelining

#### System Requirements

The pipelining experiments were conducted on a Google Cloud instance with two Tesla T4 GPUs, and the same boot disk Debian 10 based Deep Learning VM for PyTorch CPU/GPU with CUDA 11.3 M102. You must use the disk image with PyTorch 1.12, and there is no additional installation needed for the pipelining API which is included in PyTorch.

Additional installation steps required to run the pipelining script:
```
pip install torchtext==0.13.0
pip install torchdata==0.4.0
```

#### Example run command
The pipelining script can be run with the following command:
```
python pipeline_tutorial.py
```

## IV. Results

#### Horovod Distributed Training

Horovod accelerates deep learning training by distributing batches across multiple GPUs. Each device gets a separate copy of the model, and the weights are updated with an All-Reduce step. We investigated ResNet-18 (~11 million trainable parameters) and ResNet-50 (~23 million trainable parameters) training using Horovod on the CIFAR-10 dataset. CIFAR-10 is a popular computer vision dataset for object recognition, and contains 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. We investigated 36 combinations of model, batch size, precision at the All-Reduce step, and number of GPUs. The different options used are as follow:

- Model: ResNet-18, ResNet-50
- Batch size: 32, 64, 128, 256, 512, 1024
- Precision: FP32, FP16, N/A
- GPUs: 1, 2

Note that when using a single GPU there is no All-Reduce step so precision is not applicable. We recorded the throughput for each training configuration as samples/second (or images/second). We did not record loss or accuracy which had negligible variation. The full table of results are shown below:

<div align="center">
<img src="./img_src/horovod_table_1.PNG" width="600">
</div>

<div align="center">
<img src="./img_src/horovod_table_2.PNG" width="600">
</div>

From the results, ResNet-18 has higher throughput than ResNet-50 as expected, because a smaller model requires less computations to train. Another interesting observation was the effect of the All-Reduce precision. In some cases like ResNet-18 with batch size 32, halving the precision nearly doubled the throughput. This indicates that the All-Reduce step was the bottleneck. In other cases like ResNet-50 with batch size 512 there is a much smaller improvement. Due to the more complex model and larger batch, there is much more computation to be done on each GPU making the All-Reduce precision less important. Batch size also had interesting behavior, and performance seemed to peak around a batch of 256 or 512. It seems that a larger batch improves performance up until a certain point when the GPU memory is saturated, and then has worsened performance.

The best performing configuration for ResNet-18 actually just used a single GPU with batch size 256. It seems that the model is too small to make batch distribution worthwhile, and it is most efficient to just train on a single GPU without having to share gradients. However, ResNet-50 had the best performance with 2 GPUs, a batch size of 512, and FP16 All-Reduce precision. When using a larger model it is most efficient to distribute the training because much more time is spent computing gradients, making the communication cost of sharing gradients worthwhile.


#### FairScale Distributed Training

FairScale accelerates deep learning training by distributing batches shards of models accross different across multiple GPUs. Each device gets a copy of a porition of the model, a shard. All shard weights are shared during an all-gather step preceding the foreward pass, and then the weights belonging to other shards are discarded. Another all-gather step precedes the backward pass, afterwhich other weights are again discarded and an all-scatter-reduce step distributes and averages weights accross GPUs. These operations mean that a large dataset can be split into N components, one for each back just as standard distributed data training. However, unlike distributed data the model itself is sharded allowing significant memory savings for large models, and an accompanying speedup.

This is unlike pipelining, as we will discuss next. Pipeliing does break the model into components, but they must be run partially sequentially. In fully shareded data parallelism each sharded portion of the model runs a minibatch end-to-end, as the weights are briefly shared in the all-gather step.


We investigated ResNet-18 (~11 million trainable parameters) and ResNet-50 (~23 million trainable parameters) training using FairScale on the CIFAR-10 dataset. We also investigated alBERT, a transfromer-based large language model trained on the GLUE COLA dataset. The COLA dataset is made of 10,657 sentences and is a binary classification task. The sentences are classified based on gramatical acceptability. <cite this https://arxiv.org/pdf/1805.12471.pdf>

 We investigated 72 combinations of model, batch size, precision, and number of GPUs. Precision here is not exclusively at the all-gather step unlike Horovod. FairScale uses pytorch lighning, which has advanced mixed precision features. Parts of the foreward and backward steps themselves are done in reduced precision, allowing speedups even on single GPU models.
 
- Model: ResNet-18, ResNet-50, alBERT
- Batch size: ResNet:[32, 64, 128, 256, 512, 1024], alBERT:[2, 4, 8, 16, 32, 64]
- Precision: FP32, FP16
- GPUs: 1, 2

We do not record loss as we do not train models to convergence and obvserved losses after a single epoch were roughtly equivilent for all training schemes. 

<div align="center">
<img src="./img_src/fairscale_table_1.png" width="600">
</div>

<div align="center">
<img src="./img_src/fairscale_table_2.png" width="600">
</div>

<div align="center">
<img src="./img_src/fairscale_table_3.png" width="600">
</div>


#### Pipelining

Pipeline parallelism is a useful technique for improving training speed in deep learning models. In this report, we split a Transformer model across two GPUs and use pipeline parallelism for training. The Transformer Encoder Layer parameters are split evenly between two GPUs. We train a Transformer model on a language modeling task to assign a probability for the likelihood of a given word to follow a sequence of words.

<div align="center">
<img src="./img_src/transformer_architecture.jpg" width="400">
</div>

We compare a single GPU training with the pipelined multi-GPU training of this transformer model with the following hyperparameters:
- embedding dimension = 200
- dimension of the feedforward network model = 200
- number of Transformer Encoder Layers = 2
- number of heads in Multihead Attention = 2

Below are the results of training time per epoch:
- Single GPU (transformer_tutorial.py): ~48 seconds
- Two GPU pipelined (pipeline_tutorial.py): ~4 seconds

As we can see there is over 10x improvement in training speed with a multi-GPU pipelined approach. We verified that both models contained the exact same number of parameters (12,025,582) so we do not believe there is any mistake in the code. We had expected around a 2x improvement so the pipelining approach far exceeded our expectations.

#### Transfer Learning

A key part of the tripnet model training is the ResNet feature extractor. The feature extraction layer is responsible for extracting high and low level features from each input image from a surgical video. These features are utilized in the tripnet model for instrument, verb, and target classification. One possibility to improve the models accuracy and convergence is transfer learning. Rather than training the ResNet feature extractor from scratch, it is possible to initialize with pretrained weights that share some high-level features with the target dataset (CholecT45). We utilized the ImageNet-1K pretrained weights as the starting point for our transfer learning task with a ResNet-18 feature extractor. ImageNet is a dataset containing more than 14M images and 22K categories, while ImageNet-1K contains the same 14M images, but is reduced to just 1K high-level categories. We hoped that the ImageNet-1K dataset would share many of the same high level features as the CholecT45 dataset, and might provide an improvement in convergence and accuracy.

We investigated two identical training schemes using the default hyperparameters for 15 epochs, changing only the ResNet-18 feature extractor pretraining. We recorded the loss for instrument, verb, and target individually, as well as the IVT triplet loss. The results are shown below:

<div align="center">
<img src="./img_src/instrument.png" width="400"> <img src="./img_src/verb.png" width="400">
</div>

<div align="center">
<img src="./img_src/target.png" width="400"> <img src="./img_src/ivt.png" width="400">
</div>


As we can see in the graphs above, pretraining provided significant improvement for all three classification tasks individually, reaching a lower loss in all cases. This means that the ImageNet-1K pretrained weights provided useful high level features as a basis for fine-tuning on the individual tasks. However, there appears to be little improvement in the loss for IVT triplet classification as a result of pretraining. Correctly identifying a triplet of instrument, verb, and target is a much more complex task, and the ImageNet-1K pretraining provided no improvement.

#### Class Activation Mapping

Each class requires a different feature extractor activation. Intuitively, identifying the surgical instrument will require different features than identifying the target tissue. In order to evaluate the models' ability to make decisions with this intuition, the class activation maps for each class type (instrument, verb, tissue) for the same image are compared. It's clear from the comparison with a ResNet-18 below that the model is learning to extract different features for each class type. 

<div align="center">
<img src="./img_src/class_cam_comparison_vid04_frame300_resnet18.png" width="800">
</div>

This comparison is done for a larger model, a ResNet-50, and a smaller model, a SqueezeNet, to show that the features learned per class type are mostly agnostic to the architecture and more specific to the class type. Specifically in this video frame, all models show instrument class activation as a large central feature over one or both of the surgical tools. The verb class type has a more sporadic activation map across all models which is intuitive because the model is looking at moving components. Lastly, the tissue class activation map is more interleaved between the tools, seemingly focusing on the background tissue. 

<div align="center">
<img src="./img_src/class_cam_comparison_vid04_frame300_resnet50.png" width="800"> 
</div>

<div align="center">
<img src="./img_src/class_cam_comparison_vid04_frame300_squeezenet0.png" width="800">
</div>


# V. References

[1] 

[2] Fully Sharded Data Parallel: faster AI training with fewer GPUs: 
    https://engineering.fb.com/2021/07/15/open-source/fsdp/
