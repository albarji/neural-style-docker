# neural-style-docker

A dockerized version of the [neural style algorithm by jcjohnson](https://github.com/jcjohnson/neural-style), with a simple flask server to produce images. [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) is used to make use of GPU hardware if available.

## Install

The only prerequisites you need are docker and nvidia-docker. To build the image type

	nvidia-docker build -t neural-style .

## Use as command

You can invoke the neural-style command by simply running a container of this image, for example:

	nvidia-docker run --rm neural-style -h

produces the usage help.

To apply the neural-style method on some host images, mount a volume such as

	nvidia-docker run --rm -v $(pwd):/images neural-style -backend cudnn -cudnn_autotune -content_image /images/content.png -style_image /images/style.png -output_image /images/output.png

In order to take full advantage of the cudnn libraries (also included in the image) the options -backend cudnn -cudnn_autotune are always recommended.
	
## Use as server

You can also deploy neural-style as an API REST server, running

	nvidia-docker run -d -p 80:80 --entrypoint "python" neural-style app.py

and you will get a server listening to localhost:80. Docs on API usage are available at http://localhost:80/v1/ui/ . This server only supports basic parameters, so the command method above should be preferred.

