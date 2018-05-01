# intect

### Neural network architect

<br>

#### Defining the architecture

The text recognition is done via a convolutional neural network (CNN).
The architecture of the CNN is to a significant extant customizable
by writing a `.yaml` specification.

Examples of such specification can be found in the [model](src/data/architectures)
directory.


#### Deployment

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


2) ###### Deployment

As a `minimum` it is refered to a deployment of the application itself, that is without client.
It is useful for training and evaluation. It is also possible to get a simple prediction, but for that we should deploy
a dedicated server (more on that bellow).

The application is shipped in [docker](https://www.docker.com/).
Make sure you have it docker installed and properly set up.

- the **minimal** image
  - defined by Dockerfile.min
   
  From the main directory, run the following command to build the minimal image.

  ```bash
  docker build -t $IMAGE_NAME -f Dockerfile.min .
  ```
  
  *NOTE: Replace the `$IMAGE_NAME` with your custom value.*
   
- the **complete** image, which contains bundled TensorFlow Serving and sets up the client as well
  - defined by Dockerfile (default)
   
  From the main directory, run the following command to build the minimal image.

  ```bash
  docker build -t $IMAGE_NAME .
  ```
  
  or you can make use of predefined `docker-compose.yml` and run the container immediately by

  ```bash
  docker-compose up
  ```

<br>

In case of building the containers (not using `docker-compose`), you have to run them.

Then execute one of the following commands to run the container.
(see [docker](https://ww.docker.com/) help for more info about running a container)

```bash
docker run -dit -p 6006:6006 --name $CONTAINER_NAME $IMAGE_NAME
```

*NOTE: Replace the `$CONTANER_NAME` with your custom value.*

or you can bind-mount the working directory, for example to provide training data easily

```bash
docker run -dit -p 6006:6006 -v ${PWD}:/code --name $CONTAINER_NAME $IMAGE_NAME
```

3) ###### Usage


```bash
docker attach $CONTAINER_NAME
```

The command above attaches the container development environment.

The setup creates two entry points (executable commands). Feel free to explore the following commands:

- intect

  The base cli for training, evaluation and simple prediction request
  
- intect-client (not usable in the minimal image)

  Client api for more comples prediction requests and serving pre-trained models
  
  
**NOTE:** The project is currently in development phase, and is still, unfortunately, meant for developers mainly.

<br>

---

<br>

###### Known Issues

After the installation, the command `intact --help` or `intact-client --help` is not functionnal.
Use `intact` or `intact-client` only to display help.
