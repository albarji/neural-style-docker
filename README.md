# neural-style-docker

A dockerized version of the neural style algorithm by jcjohnson, with a simple flask server to produce images. nvidia-docker is used to make use of GPU hardware.

## Install and deploy

The only prerequisites you need are docker and nvidia-docker. To run a local server type

	nvidia-docker build -t neural-style .
	nvidia-docker run -d -p 80:80 neural-style

and you will get a server listening to localhost:80. Docs on API usage are available at http://localhost:80/v1/ui/

