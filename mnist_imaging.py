from PIL import Image
import numpy as np


def arr_to_img(letter_array, x_dim, y_dim, rescale=True):
    img = Image.new("RGB", (x_dim, y_dim), "black")
    pixels = img.load()
    for i in range(y_dim):
        for j in range(x_dim):
            if rescale:
                intensity = one_to_256(letter_array[(i * y_dim) + j])
            else:
                intensity = letter_array[(i * y_dim) + j]

            pixels[j, i] = (intensity, intensity, intensity)
    return img


def show_image(img):
    arr_to_img(img[0, :784], 28, 28).show()


def save_image(img, file_name):
    img_to_save = arr_to_img(img[0, :784], 28, 28)
    img_to_save.save(file_name)


def one_to_256(number):
    return int(round(number * 256))


def two56_to_one(number):
    return float(number / 256)


def get_random_image():
    return np.float32(np.random.randint(2, size=(1, 784)))

def get_random_image_3lev():
    rnd_arr = np.float32(np.random.randint(3, size=(1, 784)))
    rnd_arr = np.multiply(rnd_arr, (1.0 / 2.0))
    return rnd_arr

def get_random_image_5lev():
    rnd_arr = np.float32(np.random.randint(5, size=(1, 784)))
    rnd_arr = np.multiply(rnd_arr, (1.0 / 4.0))
    return rnd_arr

def img_to_arr(filename, dimx, dimy):
    img = Image.open(filename)
    pixel_grid = img.load()

    image_array = np.zeros([dimx * dimy], dtype=np.float32)

    for i in range(dimy):
        for j in range(dimx):
            pixel = pixel_grid[j, i]
            pixel_gs = (pixel[0] + pixel[1] + pixel[2]) / 3
            print("pgs: " + str(pixel_gs))
            pgs_one = pixel_gs / 256.0
            print("pgs1: " + str(pgs_one))

            image_array[(i * dimx) + j] = pgs_one

    return image_array


def average_image_arrays(image1, image2, length):

    new_image = np.zeros([length], dtype=np.float32)
    for i in range(length):
        avg_pixel = (image1[i] + image2[i]) / 2.0
        new_image[i] = avg_pixel
    return new_image


def avg_disc_image_arrays(image1, image2, length):
    new_image = average_image_arrays(image1, image2, length)
    for i in range(length):
        if new_image[i] > .5:
            new_image[i] = 1.0
        else:
            new_image[i] = 0.0
