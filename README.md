### GAN in Pytorch

Pytorch implementations of Generative Adversarial Network algorithms. 

* GAN
* WGAN

&emsp;

#### GAN

Note that we choose the -log(D(G(x))) to update the generator, and the generation effect are as follows.

|             Example1             |             Example2             |
| :------------------------------: | :------------------------------: |
| ![mnist1](gan/images/mnist1.gif) | ![mnist2](gan/images/mnist2.gif) |

##### Run Example

```bash
$ cd gan
$ python train.py
$ python infer.py
```

&emsp;

#### WGAN

Compared with GAN, WGAN has the following modification

* the sigmoid function employed in last layer is removed.
* the loss function of generator and discriminator don't need the log operation.
* clipping the absolute value of  parameters of the discriminator to a constant c after updating the discriminator.
* When choosing optimizer, note that RMSProp, SGD can be better.

|             Epoch(1-100)             |             Epoch(101-200)             |
| :------------------------------: | :------------------------------: |
| ![mnist1](wgan/images/mnist1.gif) | ![mnist2](wgan/images/mnist2.gif) |

##### Run Example

```bash
$ cd wgan
$ python train.py
$ python infer.py
```

&emsp;

#### WGAN-GP

Compared with WGAN, WGAN has the following modification
* Modify the object function and add the second gradient punishment
* Batch Normalization can't be used in the discriminator

To be continued~~~