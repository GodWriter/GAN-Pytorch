import os
import imageio


def create_gif(image_path):
    frames = []
    gif_name = os.path.join("images", 'mnist2.gif')

    image_list = os.listdir(image_path)
    sorted(image_list)

    for image_name in image_list:
        frames.append(imageio.imread(os.path.join(image_path, image_name)))

    imageio.mimsave(gif_name, frames, 'GIF', duration=0.1)


if __name__ == "__main__":
    image_path = "images/example2"
    create_gif(image_path)
