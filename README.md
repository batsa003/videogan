PyTorch Implementation of Generating Videos with Scene Dynamics.
=====================================

This repository contains a PyTorch implementation of [Generating Videos with Scene Dynamics](http://web.mit.edu/vondrick/tinyvideo/) by Carl Vondrick, Hamed Pirsiavash, Antonio Torralba, appeared at NIPS 2016. The model learns to generate tiny videos using adversarial networks.

I hope you find this implementation useful.

Results
-------------------
The golf dataset contains about 600,000 videos, and we use the batch size of 64. Each epoch is around 8500 iterations. The learning rate is decayed by a factor of 10 every 1000 iterations. A training for one epoch took several days on GPU. The following graphs displays the discriminator training loss, generator training loss, and generator validation loss over time. The model seems to converge around 1000-th iteration in the first epoch, where the first learning rate decay happens.

<img src ="https://u.imageresize.org/12b9deaf-b9fc-448b-b24c-4dd48c19023f.png" width="400" height="400"/>
<img src ="https://u.imageresize.org/5ac7d8c2-1f50-4ac3-995b-631aa93fae18.png" width="400" height="400"/>
<img src ="https://u.imageresize.org/e65ede8d-9cf5-4f2f-99fc-8114ad4f88c9.png" width="400" height="400"/>

Example Generations
-------------------
Below are some selected videos during the training of the network. Unnoticed bugs in the implementation could be degrading the results.

<table><tr><td>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_1299_a.jpg'>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_1299_b.gif'>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_1399_a.jpg'>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_1399_b.gif'>
</td><td>
</td></tr></table>

<table><tr><td>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_2199_a.jpg'>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_2199_b.gif'>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_2299_a.jpg'>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_2299_b.gif'>
</td><td>
</td></tr></table>

<table><tr><td>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_3199_a.jpg'>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_3199_b.gif'>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_4999_a.jpg'>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_1_4999_b.gif'>
</td><td>
</td></tr></table>

<table><tr><td>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_2_1099_a.jpg'>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_2_1099_b.gif'>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_2_1499_a.jpg'>
<img src='https://github.com/batsa003/videogan/blob/master/gen_videos/fake_gifs_2_1499_b.gif'>
</td><td>
</td></tr></table>

Training
--------
The code requires a pytorch installation. 

To train the model, refer to main.py.

Data
----
The data used in the training is from the golf videos from the original paper.
Make sure you modify self.data_root in the data_loader.py once you download the dataset. The dataset can be downloaded from the Author's paper website below.

Reference:
---------
http://carlvondrick.com/tinyvideo/
