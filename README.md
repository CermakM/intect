# ponco-ocr

### Text recognition engine

<br>

#### Defining the architecture

The text recognition is done via a convolutional neural network (CNN).
The architecture of the CNN is to a significant extant customizable
by writing a `.yaml` specification.

Examples of such specification can be found in the [model](src/data/model)
directory.


#### Training

In order to train the neural network with the given architecture,
a few things have to be done first.

1) ###### Prepare data sets

The data is expected to be in *Keras*-like format. This means
that the data directory consists of *folders representing classes*
usually we want two to three such directories - *train, validation
and test*.


Example of training data set directory structure could look like this:

```
.
+-- train_data
|   +-- class_A
|   |   +-- sample_A_1.png
|   |   +-- sample_A_2.png
        ...
|   +-- class_B
|   |   +-- sample_B_1.png
|   |   +-- sample_B_2.png
        ...
|   +-- class_C
|   |   +-- sample_B_1.png
|   |   +-- sample_B_2.png
```

By default, the estimator expects images of shape (32, 32, 1), but it should be possible
to specify custom `IMAGE_SHAPE`.

**NOTE:** Take a look at the [char-generator](https://github.com/CermakM/char-generator) GitHub repository.
The generator generates data suitable for this purpose. Feel free to experiment with the fonts
used for data generation.


2) ###### Run the training

In order to train the network, we'll make use of the  [train.py](src/poncoocr/train.py).