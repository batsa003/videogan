PyTorch Implementation of Generating Videos with Scene Dynamics.
=====================================

This repository contains a PyTorch implementation of [Generating Videos with Scene Dynamics](http://web.mit.edu/vondrick/tinyvideo/) by Carl Vondrick, Hamed Pirsiavash, Antonio Torralba, to appear at NIPS 2016. The model learns to generate tiny videos using adversarial networks.

Example Generations
-------------------
Below are some selected videos during the training of the network. Currently, due to hardware limitations because the video data is huge and a lot computing power is needed. The results aren't high-quality as of now. It could also be due to a bug in my implementation of the model.

<table><tr><td>
<img src='https://media.giphy.com/media/l49JN6JV9Evt49NFS/giphy.gif'>
<img src='https://media.giphy.com/media/3oFzmtfseE1LYJSBDq/giphy.gif'>
<img src='https://media.giphy.com/media/3oFzmlwpcN3SGG8TgA/giphy.gif'>
</td><td>
</td></tr></table>

Training
--------
The code requires a pytorch installation. 

To train a generator for video, see main.py. This file will construct the networks, start many threads to load data, and train the networks.

Data
----
The data used in the training is from the golf images from the original paper.
Make sure you modify self.data_root in the data_loader.py.

Reference:
http://carlvondrick.com/tinyvideo/
