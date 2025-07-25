# AVNN - Associated Value Neural Network

## What the hell did I stumble on?
This was just an attempt to create a neural network that preserves associations between values and meaning, creating layers that either work as value-activated, meaning carrying, or meaning-activated, value-carrying. This would mean that associations between tokens would be kept as the they travel through the neural net. Feel free to play around and commit.

At first, I tried just carrying over the value through functions (derives) that take the highest activator/mean activator etc's corresponding meaning value, but quickly ran into problems creating descrete, non-differential selection mechanics that mess up backpropagation, making it unlearnable.

# Parts and pieces

## Entrypoint 1: The tokenizer
The `TagValueTokenizer` turns .csv files into an associated tensor, where each value is a pair. As such, the tensors look like this:
`[B, F, 2]` where B is batch, and F is features.

## Entrypoint 2: Packagers

Packagers are mainly there to allow the data from normal neural networks to flow into the AVNN architecture.

## Type1EmptyPackager

This packager fills the given tensor into the value dimension `[..., 0]`. An empty meaning dimension `[..., 1]` gets created and added to the packager.

## Type2EmptyPackager

This packager fills the given tensor into the meaning dimension `[..., 1]`. An empty value dimension `[..., 0]` gets created and added to the packager.

## FuseAsMeaningPackager

This packager takes a tensor on initiation, and will set it as the meaning. During forwarding, another tensor will be taken and used as value.

## FuseAsValuePackager

This packager takes a tensor on initiation, and will set it as the value. During forwarding, another tensor will be taken and used as meaning.

# Layers, bridges and blocks

Layers come as type 1 or type 2, meaning either (type 1) the values will be used as activators and the meaning will be derived from a derive function, or (type 2) the meaning will be used as activators and the value will be derived from a derive function. Blocks are arranged as one Type 1 layer, one Type 2 layer.

 - AVNNType1Linear
 - AVNNType2Linear
 - AVNNLinearBlock

Bridges are learnable layers that can help change data from linear-compatible to Conv2d-compatible, changing the tensor from: `[B, F, 2]` to `[B, C, H, W, 2]`. The bridges are:

 - AVNNLinearToConv2dBridge
 - AVNNConv2dToLinarBridge

There's also implementations for 2D convolutional layers, and a block, named in the same fashion as the linear layers:
 - AVNNType1Conv2d
 - AVNNType2Conv2d
 - AVNNConv2dBlock

# Derive modes

Derive modes define the function that will be applied on the meaning values, calculating. Each of these take in two tensors, one for value, another for meaning, as well as a temperature setting. Here's the current ones available:
 - derived_max
 - derived_min
 - derived_adjustedmin
 - derived_mean
 - derived_adjustedmean

# Condensers

Condensers bridge the way from AVNN back to common neural networks, reducing the dimension of AVNN's  `[B, F, 2]` tensor into a conventional `[B, F]` tensor. Here a few examples:

## MappingCondenser

Collapses one dimension of the tensor by appending the meaning and the value, thus going from `[B, F, 2]` to `[B, F x 2]`

## FusingCondenser

A learned layer is used where both value and meaning is put inside an activation function, relu by default, as: `Sum(value x value_weight + meaning x meaning_value) + bias`

## SeparatingCondenser

Kinda experimental, this condenser forwards an array with entries being a value tensor and a meaning tensor.