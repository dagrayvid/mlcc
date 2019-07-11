# Using RStudio in a container on a remote host with VNC
In this example we will use MLCC to create a container with RStudio and vncserver installed so that we can run the RStudio GUI in a remote server in a container.

## Connect to the remote server
Connect to the remote host.
```sh
ssh <remote_server_id>
```

## Build the container
In whatever directory you have the mlcc executable, and MLCC_Frags create the Dockerfile with
```sh
./mlcc -i RHEL7.6,CPU,R-Studio,VNC -o example_03_dockerfile
```

Then build the container. (Using nohup and saving the build log)
```sh
nohup podman build -t rstudio_vnc -f example_03_dockerfile . > build_log.out 2>&1 &
```

## Run the container
Once the container is successfully built, run the container, forwarding port 5901 for the vncserver.
```sh
podman run -it -p 5901:5901 -v /etc/machine-id:/etc/machine-id:ro rstudio_vnc /bin/bash
```

## Start the VNC server
```sh
vncpasswd #Enter a password
vncserver :1
```

## Connect to the VNC server from your local machine
Assuming you have some vncviewer installed, such as TigerVNC, connect to the VNC server.

```sh
vncviewer <remote_host_ip>:1

You should be prompted for the vncserver password. After entering the password a window will pop up with a terminal. You can start Rstudio in the window by entering ```rstudio``` into the terminal.

In RStudio try making a graph. ```curve(sin(x), xlim=c(0,10))```
