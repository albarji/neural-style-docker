# neural-style-docker

![Stylized Docker](./doc/docker_afremov_sw5000_ss1.png)
![Stylized Docker](./doc/docker_broca_sw5000_ss1.png)
![Stylized Docker](./doc/docker_brownrays_sw375_ss1.png)
![Stylized Docker](./doc/docker_ediaonise_sw1500_ss1.png)
![Stylized Docker](./doc/docker_edimburgGraffit_sw20000.0_ss1.png)
![Stylized Docker](./doc/docker_himesama_sw10000_ss1.png)
![Stylized Docker](./doc/docker_paisaje_urbano-hundertwasser_sw2000_ss1.png)
![Stylized Docker](./doc/docker_potatoes_sw375_ss1.png)
![Stylized Docker](./doc/docker_RenoirDogesPalaceVenice_sw1500_ss1.png)
![Stylized Docker](./doc/docker_revellerAndCourtesan_sw2000_ss1.png)
![Stylized Docker](./doc/docker_seated-nude_sw375_ss1.png)
![Stylized Docker](./doc/docker_starryNight_sw1500_ss1.png)

A dockerized version of the [neural style algorithm by jcjohnson](https://github.com/jcjohnson/neural-style). [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) is used to make use of GPU hardware.

## Install

### Prerequisites

* [docker](https://www.docker.com/)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* Appropriate nvidia drivers for your GPU

### Installation

You can either pull the Docker image from Docker Hub with

	docker pull albarji/neural-style

or build the image locally with

	make

## Simple use

Just run

	bash scripts/fake-it.sh

This applies a blend of content and style with some default parameters. Both content and style images must be present in the "contents" and "styles" folders, respectively.

Example: to draw the Golden Gate bridge the style of Van Gogh's Starry Night, type

	bash scripts/fake-it.sh goldengate.jpg vangogh.jpg

## Advanced use

### Generating variants

Running the command script

	bash scripts/variants.sh

will generate several variants of the same image blends, for different neural-style parameters that work well in general. This is useful for producing several versions of the same blend and afterwards hand-picking the best one. Run this command with the -h option to obtain usage help.

For example, to generate different variants of Docker logo + Starry Night:

	bash scripts/variants.sh --contents contents/docker.png --styles styles/vangogh.jpg

### Use as the neural-style command

You can directly invoke the core neural-style algorithm by simply running a container of this image, for example:

	nvidia-docker run --rm albarji/neural-style -h

produces the usage help.

To apply the neural-style method on some host images, map the host folder with such images to the container /images folder through a volume such as

	nvidia-docker run --rm -v $(pwd):/images albarji/neural-style -backend cudnn -cudnn_autotune -content_image content.png -style_image style.png

The container uses as work directory the /images folder, so the results will be readily available at the mounted host folder.

In order to take full advantage of the cudnn libraries (also included in the image) the options -backend cudnn -cudnn_autotune are always recommended.

As an example, let's redraw Docker's logo in the famous style of Van Gogh's Starry Night:

	nvidia-docker run --rm -v $(pwd):/images albarji/neural-style -backend cudnn -cudnn_autotune -content_image contents/docker.png -style_image styles/vangogh.jpg


