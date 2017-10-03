.PHONY: all build clean build_tests tests clean_tests
IMGNAME=albarji/neural-style
IMGTAG=latest
TESTIMGNAME=$(IMGNAME)-tests

build:
	nvidia-docker build -t $(IMGNAME):$(IMGTAG) .

clean:
	docker rmi $(IMGNAME):$(IMGTAG)

build_tests:
	nvidia-docker build -t $(TESTIMGNAME) tests

tests:
	nvidia-docker run --rm -it $(TESTIMGNAME)

clean_tests:
	docker rmi $(TESTIMGNAME)

all: build build_tests tests
