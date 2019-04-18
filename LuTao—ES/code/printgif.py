import imageio

img_path = ['image/'+ str(i) + '.png' for i in range(150)]
gif_images = []

for path in img_path:
    gif_images.append(imageio.imread(path))
imageio.mimsave("nes.gif",gif_images,fps=10)