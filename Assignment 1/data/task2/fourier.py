import cv2
import matplotlib.pyplot as plt
import numpy as np


def fm_spectrum(img):
    fft_img = abs(np.fft.fft2(img))
    size = len(fft_img)

    for i in range(size):
        fft_img[i] = np.roll(fft_img[i], size//2)

    fft_img = np.roll(fft_img, size // 2, axis=0)

    return 30 * np.log(fft_img+0.001)


def low_pass_filter(img, th=20):
    fft_img = np.fft.fft2(img)
    size = len(fft_img)

    for i in range(size):
        fft_img[i] = np.roll(fft_img[i], size//2)

    fft_img = np.roll(fft_img, size // 2, axis=0)

    filter = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    center = [(img.shape[0]-1)/2, (img.shape[1]-1)/2]

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            dist = np.sqrt((row-center[0])*(row-center[0]) + (col-center[1])*(col-center[1]))
            if dist < th:
                filter[row][col] = 1.0

    fft_img = filter * fft_img

    for i in range(size):
        fft_img[i] = np.roll(fft_img[i], size // 2)

    fft_img = np.roll(fft_img, size // 2, axis=0)

    ifft_img = np.fft.ifft2(fft_img)
    original_img = np.real(ifft_img)

    return original_img

def high_pass_filter(img, th=30):
    fft_img = np.fft.fft2(img)
    size = len(fft_img)

    for i in range(size):
        fft_img[i] = np.roll(fft_img[i], size // 2)

    fft_img = np.roll(fft_img, size // 2, axis=0)

    filter = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    center = [(img.shape[0] - 1) / 2, (img.shape[1] - 1) / 2]

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            dist = np.sqrt((row - center[0]) * (row - center[0]) + (col - center[1]) * (col - center[1]))
            if dist > th:
                filter[row][col] = 1.0

    fft_img = filter * fft_img

    for i in range(size):
        fft_img[i] = np.roll(fft_img[i], size // 2)

    fft_img = np.roll(fft_img, size // 2, axis=0)

    ifft_img = np.fft.ifft2(fft_img)
    original_img = np.real(ifft_img)

    return original_img

def denoise1(img):
    fft_img = np.fft.fft2(img)
    size = len(fft_img)

    for i in range(size):
        fft_img[i] = np.roll(fft_img[i], size // 2)

    fft_img = np.roll(fft_img, size // 2, axis=0)

    filter = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)

    noises = [
        [145,35],[365,35],
        [90,90],[200,90],[310,90],[420,90],
        [145,145],[365,145],
        [90,200],[200,200],[310,200],[420,200],
        [145,255],[365,255],
        [90,310],[200,310],[310,310],[420,310],
        [145,365],[365,365],
        [90,420],[200,420],[310,420],[420,420],
        [145,475],[365,475]
    ]

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            for point in noises:
                dist = np.sqrt((row - point[0]) * (row - point[0]) + (col - point[1]) * (col - point[1]))
                if 0 <= dist <= 5:
                    filter[row][col] = 0

    fft_img = filter * fft_img

    for i in range(size):
        fft_img[i] = np.roll(fft_img[i], size // 2)

    fft_img = np.roll(fft_img, size // 2, axis=0)

    ifft_img = np.fft.ifft2(fft_img)
    original_img = np.real(ifft_img)

    cv2.imwrite("denoised1.png", original_img)

    return original_img

def denoise2(img):
    fft_img = np.fft.fft2(img)
    size = len(fft_img)

    for i in range(size):
        fft_img[i] = np.roll(fft_img[i], size // 2)

    fft_img = np.roll(fft_img, size // 2, axis=0)

    filter = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
    center = [(img.shape[0] - 1) / 2, (img.shape[1] - 1) / 2]

    outer_radius = 45
    inner_radius = 35

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            dist = np.sqrt((row - center[0]) * (row - center[0]) + (col - center[1]) * (col - center[1]))
            if dist < outer_radius and dist > inner_radius:
                filter[row][col] = 0.0

    fft_img = filter * fft_img

    for i in range(size):
        fft_img[i] = np.roll(fft_img[i], size // 2)

    fft_img = np.roll(fft_img, size // 2, axis=0)

    ifft_img = np.fft.ifft2(fft_img)
    original_img = np.real(ifft_img)

    cv2.imwrite("denoised2.png", original_img)

    return original_img


if __name__ == '__main__':
    img = cv2.imread('task2_sample.png', cv2.IMREAD_GRAYSCALE)
    cor1 = cv2.imread('task2_corrupted_1.png', cv2.IMREAD_GRAYSCALE)
    cor2 = cv2.imread('task2_corrupted_2.png', cv2.IMREAD_GRAYSCALE)

    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])

    drawFigure((2,7,1), img, 'Original')
    drawFigure((2,7,2), low_pass_filter(img), 'Low-pass')
    drawFigure((2,7,3), high_pass_filter(img), 'High-pass')
    drawFigure((2,7,4), cor1, 'Noised')
    drawFigure((2,7,5), denoise1(cor1), 'Denoised')
    drawFigure((2,7,6), cor2, 'Noised')
    drawFigure((2,7,7), denoise2(cor2), 'Denoised')

    drawFigure((2,7,8), fm_spectrum(img), 'Spectrum')
    drawFigure((2,7,9), fm_spectrum(low_pass_filter(img)), 'Spectrum')
    drawFigure((2,7,10), fm_spectrum(high_pass_filter(img)), 'Spectrum')
    drawFigure((2,7,11), fm_spectrum(cor1), 'Spectrum')
    drawFigure((2,7,12), fm_spectrum(denoise1(cor1)), 'Spectrum')
    drawFigure((2,7,13), fm_spectrum(cor2), 'Spectrum')
    drawFigure((2,7,14), fm_spectrum(denoise2(cor2)), 'Spectrum')

    plt.show()