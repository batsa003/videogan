import imageio

# Receives [3,32,64,64] tensor, and creates a gif
def make_gif(images, filename):
    x = images.permute(1,2,3,0)
    x = x.numpy()
    frames = []
    for i in range(32):
        frames += [x[i]]
    imageio.mimsave(filename, frames)

#x = torch.rand((3,32,64,64))
#make_gif(x, 'movie.gif')
