model-name:=test-model
base-model-name:=vgg16

model:
	python model.py $(model-name) -b $(base-model-name)

base-models:= \
	vgg16 \
	vgg19 \
	resnet50 \
	inception_v3 \
	mobilenet \
	densenet \
	xception \
	nasnet

models-with-base: $(base-models:%=train-with/%)

train-with/%:
	python model.py $(model-name)_$(@F) -b $(@F)
