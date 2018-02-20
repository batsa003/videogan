PyTorch Implementation of Generating Videos with Scene Dynamics.
=====================================

This repository contains a PyTorch implementation of [Generating Videos with Scene Dynamics](http://web.mit.edu/vondrick/tinyvideo/) by Carl Vondrick, Hamed Pirsiavash, Antonio Torralba, to appear at NIPS 2016. The model learns to generate tiny videos using adversarial networks.

I hope you find this implementation useful.

Example Generations
-------------------
Below are some selected videos during the training of the network. Unnoticed bugs in the implementation could be degrading the results.

<table><tr><td>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_1299_a.jpg'>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_1299_b.gif'>
</td><td>
</td></tr></table>
<table><tr><td>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_1299_a.jpg'>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_1299_b.gif'>
</td><td>
</td></tr></table>

Training
--------
The code requires a pytorch installation. 

To train a generator for video, see main.py. This file will construct the networks, start many threads to load data, and train the networks.

Data
----
The data used in the training is from the golf videos from the original paper.
Make sure you modify self.data_root in the data_loader.py once you download the dataset. The dataset can be downloaded from the videogan official website.

Reference:
---------
http://carlvondrick.com/tinyvideo/
