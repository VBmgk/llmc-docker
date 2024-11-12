# Host Configuration (Archlinux)

On Archlinux, you need the @nvidia-container-toolkit@ package to run. You also need
to restart the docker service after that installation.

You can test you installation with:
```
docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 nvidia-smi
```

You can run the image with:
```
docker build .
```

# Train GPT2

To execute the images, you need to run the container with (you need a gpu):
```
docker run --gpus all -it 6da25d955c75 bash
```

You should get similar results to the following:

```
./train_gpt2cu
hwloc/linux: Ignoring PCI device with non-16bit domain.
Pass --enable-32bits-pci-domain to configure to support such devices
(warning: it would break the library ABI, don't enable unless really needed).
hwloc/linux: Ignoring PCI device with non-16bit domain.
Pass --enable-32bits-pci-domain to configure to support such devices
(warning: it would break the library ABI, don't enable unless really needed).
+-----------------------+----------------------------------------------------+
| Parameter             | Value                                              |
+-----------------------+----------------------------------------------------+
| train data pattern    | dev/data/tinyshakespeare/tiny_shakespeare_train.bin |
| val data pattern      | dev/data/tinyshakespeare/tiny_shakespeare_val.bin  |
| output log dir        | NULL                                               |
| checkpoint_every      | 0                                                  |
| resume                | 0                                                  |
| micro batch size B    | 4                                                  |
| sequence length T     | 1024                                               |
| total batch size      | 4096                                               |
| LR scheduler          | cosine                                             |
| learning rate (LR)    | 3.000000e-04                                       |
| warmup iterations     | 0                                                  |
| final LR fraction     | 1.000000e+00                                       |
| weight decay          | 0.000000e+00                                       |
| skip update lossz     | 0.000000                                           |
| skip update gradz     | 0.000000                                           |
| max_steps             | -1                                                 |
| val_loss_every        | 20                                                 |
| val_max_steps         | 20                                                 |
| sample_every          | 20                                                 |
| genT                  | 64                                                 |
| overfit_single_batch  | 0                                                  |
| use_master_weights    | enabled                                            |
| gelu_fusion           | 0                                                  |
| recompute             | 1                                                  |
+-----------------------+----------------------------------------------------+
| device                | NVIDIA GeForce RTX 3060 Laptop GPU                 |
| peak TFlops           | -1.0                                               |
| precision             | BF16                                               |
+-----------------------+----------------------------------------------------+
| weight init method    | gpt2_124M_bf16.bin                                 |
| max_sequence_length T | 1024                                               |
| vocab_size V          | 50257                                              |
| padded_vocab_size Vp  | 50304                                              |
| num_layers L          | 12                                                 |
| num_heads NH          | 12                                                 |
| channels C            | 768                                                |
| num_parameters        | 124475904                                          |
+-----------------------+----------------------------------------------------+
| train_num_batches     | 74                                                 |
| val_num_batches       | 20                                                 |
+-----------------------+----------------------------------------------------+
| run hellaswag         | no                                                 |
+-----------------------+----------------------------------------------------+
| Zero Optimization is disabled                                              |
| num_processes         | 1                                                  |
| zero_stage            | 0                                                  |
+-----------------------+----------------------------------------------------+
num_parameters: 124475904 => bytes: 248951808
allocated 237 MiB for model parameters
batch_size B=4 * seq_len T=1024 * num_processes=1 and total_batch_size=4096
=> setting grad_accum_steps=1
allocating 237 MiB for parameter gradients
allocating 1326 MiB for activations
allocating 474 MiB for AdamW optimizer state m
allocating 474 MiB for AdamW optimizer state v
allocating 474 MiB for master copy of params
device memory usage: 3511 MiB / 5930 MiB
memory per sequence: 331 MiB
 -> estimated maximum batch size: 11
val loss 4.503414
step    1/74 | loss 4.283499 (+nanz)| norm 13.0082 (+nanz)| lr 3.00e-04 | 349.48 ms | -100.0% bf16 MFU | 11720 tok/s
...
step   42/74 | loss 3.364718 (+nanz)| norm 1.5494 (+nanz)| lr 3.00e-04 | 168.70 ms | -100.0% bf16 MFU | 24333 tok/s
step   43/74 | loss 3.361276 (+nanz)| norm 1.5459 (+nanz)| lr 3.00e-04 | 168.71 ms | -100.0% bf16 MFU | 24330 tok/s
...
generating:
---
LITERATURE:
In my leave I take your voices:
If he hear you speak,
Then not a word of mine.

<|endoftext|>JOHN:<|endoftext|>GRUMIO:<|endoftext|>ernel of pedant base,
And spirit of his side;
Sleep no longer;
And therefore never weary you
---
total average iteration time: 168.436902 ms
---------------------------------------------
```

Test Results

```
root@3c5eede03490:~/llm.c# USE_CUDNN=1 make test_gpt2
---------------------------------------------
✓ cuDNN found, will run with flash-attention
✓ OpenMP found
✓ NCCL found, OK to train with multiple GPUs
✓ MPI enabled
✓ nvcc found, including GPU/CUDA support
---------------------------------------------
cc -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes -march=native -fopenmp -DOMP   test_gpt2.c -lm -lgomp -o test_gpt2
root@3c5eede03490:~/llm.c# ./test_gpt2
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
padded_vocab_size: 50304
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124475904

[State]
batch_size: 4
seq_len: 64
num_activations: 73347840
-43.431618, -43.431736
-39.836346, -39.836460
-43.065910, -43.066002
-42.828045, -42.828171
-43.529541, -43.529659
-44.318398, -44.318508
-41.227425, -41.227512
-41.270760, -41.270866
-42.541393, -42.541489
-42.394997, -42.395119
OK (LOGITS), max_diff = 1.251221e-03
LOSS OK: 5.269997 5.270009
dwte
OK -0.002320 -0.002320
OK 0.002072 0.002072
OK 0.003717 0.003717
OK 0.001307 0.001307
OK 0.000632 0.000632
TENSOR OK, maxdiff = 1.502037e-04
dwpe
TENSOR OK, maxdiff = 2.964400e-05                                                                                                                                                                                                                              13:40:54 [47/1643]
dln2w
OK -0.018318 -0.018312
OK 0.004812 0.004813
OK 0.008089 0.008091
OK -0.001469 -0.001470
OK -0.002737 -0.002737
TENSOR OK, maxdiff = 9.164214e-04
dln2b
OK -0.026374 -0.026368
OK -0.016703 -0.016695
OK 0.001072 0.001074
OK 0.034711 0.034711
OK -0.028584 -0.028584
TENSOR OK, maxdiff = 7.778406e-05
dfcw
OK 0.000440 0.000440
OK -0.000000 -0.000000
OK -0.000154 -0.000154
OK -0.000165 -0.000165
OK 0.000405 0.000405
TENSOR OK, maxdiff = 7.795542e-05
dfcb
OK 0.003291 0.003293
OK 0.002043 0.002043
OK -0.001386 -0.001386
OK 0.000386 0.000386
OK 0.001603 0.001604
TENSOR OK, maxdiff = 2.011610e-05
dfcprojw
OK 0.000680 0.000681
OK 0.000073 0.000073
OK -0.000416 -0.000416
OK -0.000060 -0.000061
OK -0.000604 -0.000604
TENSOR OK, maxdiff = 3.764278e-05
dfcprojb
OK 0.003583 0.003584
OK -0.007157 -0.007158
OK -0.001962 -0.001964
OK 0.001462 0.001462
OK 0.001217 0.001217
TENSOR OK, maxdiff = 1.168525e-05
dlnfw
OK -0.000022 -0.000022
OK 0.000810 0.000811
OK 0.001161 0.001161
OK -0.002957 -0.002957
OK 0.001145 0.001145
TENSOR OK, maxdiff = 1.670718e-04
dlnfb
OK -0.011101 -0.011101
OK 0.008009 0.008007
OK -0.004771 -0.004769
OK -0.002112 -0.002113
OK -0.005905 -0.005905
TENSOR OK, maxdiff = 2.920954e-05
step 0: loss 5.269997 (took 1580.996770 ms) OK = 1
step 1: loss 4.059699 (took 1590.308096 ms) OK = 1
step 2: loss 3.375026 (took 1393.871860 ms) OK = 1
step 3: loss 2.800780 (took 1390.227751 ms) OK = 1
step 4: loss 2.315454 (took 1468.947172 ms) OK = 1
step 5: loss 1.849150 (took 1446.225618 ms) OK = 1
step 6: loss 1.394839 (took 1425.464618 ms) OK = 1
step 7: loss 0.999173 (took 1478.736013 ms) OK = 1
step 8: loss 0.624303 (took 1676.499174 ms) OK = 1
step 9: loss 0.376687 (took 1616.597850 ms) OK = 1
overall okay: 1

```
