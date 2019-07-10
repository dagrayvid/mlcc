In the directory with mlcc, create a Dockerfile using mlcc for running pytorch and jupyter on your local machine.


# Running live object detection on video frames
In this example we will use MLCC to build a container for running the YOLOv3 model in PyTorch to detect objects in video from files or the webcam. OpenCV is used for the video I/O, and for drawing bounding boxes. The program can be run using GPU acceleration with CUDA, or just on CPU (with a limited framerate). 

## Before you start
This example expects that you have an mlcc executable somewhere on your system. This example can be run locally or remotely. Connecting to the display has only been tested locally on Fedora30.

## Build the container
In whatever directory you have the mlcc executable, and MLCC_Frags create the Dockerfile.

For a system without a GPU:
```sh
./mlcc -i Fedora29,CPU,Matplotlib,Jupyter,OpenCV,PyTorch -o demo_02_dockerfile
```

On a GPU system, 
```sh
./mlcc -i Fedora29,CUDA10.0,Matplotlib,Jupyter,OpenCV,PyTorch -o demo_02_dockerfile
```

Then build the container. We suggest using nohup and saving the logs when building a container.
```sh
nohup podman build -f demo_02_dockerfile -t object_detection . 2>&1 &
tail -f nohup.out
```


## Run the container
Once the container is successfully built, run the container.

On a system without a GPU:
```sh
CONTAINER_DISPLAY_ARGS="-v /tmp/.X11-unix/:/tmp/.X11-unix/ -v /dev/dri:/dev/dri --ipc=host -e DISPLAY"
podman run -it -v $(pwd)/object_detection_files:/object_detection_files:Z  -v /dev/video0:/dev/video0 $CONTAINER_DISPLAY_ARGS object_detection /bin/bash
```

On a system with a GPU, you can manually pass in the NVIDIA devices:
```sh
CONTAINER_DISPLAY_ARGS="-v /tmp/.X11-unix/:/tmp/.X11-unix/ -v /dev/dri:/dev/dri --ipc=host -e DISPLAY"
NVIDIA_DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
podman run -it -v $(pwd)/object_detection_files:/object_detection_files:Z  -v /dev/video0:/dev/video0 $CONTAINER_DISPLAY_ARGS $NVIDIA_DEVICES object_detection /bin/bash
```

Or if you have the nvidia-container-runtime-hook installed:
```sh
CONTAINER_DISPLAY_ARGS="-v /tmp/.X11-unix/:/tmp/.X11-unix/ -v /dev/dri:/dev/dri --ipc=host -e DISPLAY"
NVIDIA_HOOK_ARGS="--security-opt=no-new-privileges --cap-drop=ALL --security-opt label=type:nvidia_container_t"
podman run -it -v $(pwd)/object_detection_files:/object_detection_files:Z  -v /dev/video0:/dev/video0 $CONTAINER_DISPLAY_ARGS $NVIDIA_HOOK_ARGS object_detection /bin/bash
```

Of course if you will only be running the script to an output video file, you don't need the $CONTAINER_DISPLAY args. Similarly, if you will not use a webcam you don't need to pass in /dev/video0 as a volume.

## Run the script:
Once inside the container, 
```sh
cd /object_detection_files
```

To see the options the script has:
```
python3 object_detection.py --help
```
To stop the program running, press ESC.

To run with cuda on webcam footage
```
python3 object_detection.py --cuda
```

To run with cuda on video footage
```
python3 object_detection.py --cuda --input-video /path/to/video
```


## References:
This example uses code from Chris Fotache's ["Object detection and tracking in PyTorch" article](https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98) and [GitHub project](https://github.com/cfotache/pytorch_objectdetecttrack).

YOLOv3 was created by Joseph Redmon and Ali Farhadi. More information can be found at the [YOLO website](https://pjreddie.com/darknet/yolo/) and in their paper [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf).

Using OpenCV to capture frames from the webcam was done with inspiration from [this Medium article by Squirrel](https://medium.com/@neotheicebird/webcam-based-image-processing-in-ipython-notebooks-47c75a022514).
