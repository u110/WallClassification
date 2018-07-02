build:
	docker build . -t justxseraku/workspace

local-dir:=$(shell pwd)/../..
workspace-dir:=/notebooks/WallClassification
run: build
	docker run --rm -it \
		-v $(local-dir):$(workspace-dir) \
		-p 8888:8888 \
		-t justxseraku/workspace

