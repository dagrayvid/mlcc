# Training a CNN on Fashion MNIST using Keras and TensorFlow 
In this example we will use MLCC to build a container for running a Jupyter notebook server on a remote host with NVIDIA GPUs. The container will include Keras with a TensorFlow backend, as well as Matplotlib, Seaborn, and scikit-learn for visualizing the result.

## Before you start
This example expects that you have an mlcc executable on a properly configured machine with NVIDIA_GPUs. It also uses the nvidia-container-runtime-hook installed.

## Connect to the remote server
Connect to the remote host, forwarding port 8888 for Jupyter.
```sh
ssh -L 8888:localhost:8888
```

## Build the container
In whatever directory you have the mlcc executable, and MLCC_Frags create the Dockerfile with
```sh
./mlcc -i RHEL7.6,CUDA10.0,Jupyter,TensorFlow,Keras,scikit-learn,Seaborn -o keras_demo_file
```

Then build the container. I always use nohup and save the logs when building a container on a remote machine.
```sh
nohup podman build -t keras_jupyter_demo -f keras_demo_file . > build_log.out 2>&1 &
```

This will take 30 - 60 minutes, so in the meantime you can download the dataset.

## Download Dataset
Run the dataset download script to download the [Fashion MINST](https://github.com/zalandoresearch/fashion-mnist) dataset in this directory: 
```sh
$ ./download_dataset.sh
```

## Run the container
Once the container is successfully built, run the container with.
```sh
podman run -it -p 8888:8888 -v $(pwd)/demo_data:/demo_data:Z  \
--security-opt label=type:nvidia_container_t jupyter_demo_image /bin/bash
```

Once in the container, you can use `nvidia-smi` to verify GPU access.

## Run the Jupyter notebook server
```sh
cd /demo_data
jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```

From the machine that you connected to the server with, you can open the browser and go to the link
`http://localhost:8888/?token=<token>`

Where the <token> is the token printed when the jupyter notebook server starts.

From within Jupyter, navigate to the keras_tensorflow_fashion_mnist.ipynb file, and run the cells.
