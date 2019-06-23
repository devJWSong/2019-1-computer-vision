import cv2
import statistics
import numpy as np
import math
import utils
import os

"""
This is main function for task 1.
It takes 2 arguments,
'src_img_path' is path for source image.
'dst_img_path' is path for output image, where your result image should be saved.

You should load image in 'src_img_path', and then perform task 1 of your assignment 1,
and then save your result image to 'dst_img_path'.
"""

def task1(src_img_path, clean_img_path, dst_img_path):
    optimal_solution = ""
    optimal_size = 0
    total_rms = []

    src_img = cv2.imread(src_img_path)
    clean_img = cv2.imread(clean_img_path)

    print(src_img_path + "\n")

    original_rms = utils.calculate_rms_cropped(src_img, clean_img)
    print("original rms: " + str(original_rms) + "\n")

    average_rms = []
    average_imgs = []
    for size in range(3, 16, 2):
        src_img_average = cv2.imread(src_img_path)
        dist_img = apply_average_filter(src_img_average, size)
        rms = utils.calculate_rms_cropped(clean_img, dist_img)
        print("average filter" + " kernel size: " + str(size) + " rms: " + str(rms) + "\n")
        average_rms.append(rms)
        average_imgs.append(dist_img)

    total_rms.append(min(average_rms))
    average_index = average_rms.index(min(average_rms))
    average_size = average_index * 2 + 3

    median_rms = []
    median_imgs = []
    for size in range(3, 16, 2):
        src_img_median = cv2.imread(src_img_path)
        dist_img = apply_median_filter(src_img_median, size)
        rms = utils.calculate_rms_cropped(clean_img, dist_img)
        print("median filter" + " kernel size: " + str(size) + " rms: " + str(rms) + "\n")
        median_rms.append(rms)
        median_imgs.append(dist_img)

    total_rms.append(min(median_rms))
    median_index = median_rms.index(min(median_rms))
    median_size = median_index * 2 + 3

    sigmas = [
        [90, 90],
        [75, 75],
        [60, 60],
        [45, 45],
        [30, 30],
        [15, 15]
    ]

    bilateral_rms = []
    bilateral_sigmas = []
    bilateral_imgs = []
    for size in range(3, 16, 2):
        for sigma_pair in sigmas:
            src_img_bilateral = cv2.imread(src_img_path)
            dist_img = apply_bilateral_filter(src_img_bilateral, size, sigma_pair[0], sigma_pair[1])
            rms = utils.calculate_rms_cropped(clean_img, dist_img)
            print("bilateral filter" + " kernel size: " + str(size)
                  + " sigma_s: " + str(sigma_pair[0])
                  + " sigma_r: " + str(sigma_pair[1])
                  + " rms: " + str(rms) + "\n")
            bilateral_rms.append(rms)
            bilateral_sigmas.append(sigma_pair)
            bilateral_imgs.append(dist_img)

    total_rms.append(min(bilateral_rms))
    bilateral_index = bilateral_rms.index(min(bilateral_rms))
    optimal_sigma_s = bilateral_sigmas[bilateral_index][0]
    optimal_sigma_r = bilateral_sigmas[bilateral_index][1]
    bilateral_size = (bilateral_index // len(sigmas)) * 2 + 3

    optimal_rms = min(total_rms)
    if total_rms.index(optimal_rms) == 0:
        optimal_solution = "average"
        optimal_size = average_size
        cv2.imwrite(dst_img_path, average_imgs[average_index])
    elif total_rms.index(optimal_rms) == 1:
        optimal_solution = "median"
        optimal_size = median_size
        cv2.imwrite(dst_img_path, median_imgs[median_index])
    elif total_rms.index(optimal_rms) == 2:
        optimal_solution = "bilateral"
        optimal_size = bilateral_size
        cv2.imwrite(dst_img_path, bilateral_imgs[bilateral_index])

    print("Optimal Solution: " + optimal_solution + "\n"
          + "Optimal Size: " + str(optimal_size) + "\n"
          + "Optimal rms: " + str(optimal_rms) + "\n")

    if optimal_solution == "bilateral":
        print("Optimal sigma_s: " + str(optimal_sigma_s) + " "
              + "Optimal sigma_r: " + str(optimal_sigma_r) + "\n")


"""
You should implement average filter convolution algorithm in this function.
It takes 2 arguments,
'img' is source image, and you should perform convolution with average filter.
'kernel_size' is a int value, which determines kernel size of average filter.

You should return result image.
"""
def apply_average_filter(img, kernel_size):

    mean = 1 / (kernel_size * kernel_size)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            value = 0
            dist = kernel_size//2
            for partial_row in range(row-dist, row+dist+1):
                for partial_col in range(col-dist, col+dist+1):
                    if partial_row < 0 or partial_col < 0 or partial_row >= img.shape[0] or partial_col >= img.shape[1]:
                        pass
                    else:
                        value += img[partial_row, partial_col] * mean

            img[row, col] = value

    return img


"""
You should implement median filter convolution algorithm in this function.
It takes 2 arguments,
'img' is source image, and you should perform convolution with median filter.
'kernel_size' is a int value, which determines kernel size of median filter.

You should return result image.
"""
def apply_median_filter(img, kernel_size):

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            dist = kernel_size//2
            color_b = np.array([], dtype=np.uint8)
            color_g = np.array([], dtype=np.uint8)
            color_r = np.array([], dtype=np.uint8)
            for partial_row in range(row-dist, row+dist+1):
                for partial_col in range(col-dist, col+dist+1):
                    if partial_row < 0 or partial_col < 0 or partial_row >= img.shape[0] or partial_col >= img.shape[1]:
                        color_b = np.append(color_b, 0)
                        color_g = np.append(color_g, 0)
                        color_r = np.append(color_r, 0)
                    else:
                        color_b = np.append(color_b, img[partial_row, partial_col][0])
                        color_g = np.append(color_g, img[partial_row, partial_col][1])
                        color_r = np.append(color_r, img[partial_row, partial_col][2])

            img[row, col] = np.array([np.median(color_b), np.median(color_g), np.median(color_r)], dtype=np.uint8)

    return img


"""
You should implement convolution with additional filter.
You can use any filters for this function, except average, median filter.
It takes at least 2 arguments,
'img' is source image, and you should perform convolution with median filter.
'kernel_size' is a int value, which determines kernel size of average filter.
'sigma_s' is a int value, which is a sigma value for G_s
'sigma_r' is a int value, which is a sigma value for G_r

You can add more arguments for this function if you need.

You should return result image.
"""
def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            color_b = []
            color_g = []
            color_r = []
            sum_weight = 0
            dist = kernel_size//2
            for partial_row in range(row-dist, row+dist+1):
                for partial_col in range(col-dist, col+dist+1):
                    if partial_row < 0 or partial_col < 0 or partial_row >= img.shape[0] or partial_col >= img.shape[1]:
                        color_b.append(0)
                        color_g.append(0)
                        color_r.append(0)
                    else:
                        space_diff = math.sqrt((row-partial_row)*(row-partial_row) + (col-partial_col)*(col-partial_col))
                        range_diff = abs(int(img[row, col][0])-int(img[partial_row, partial_col][0])) \
                                         + abs(int(img[row, col][1])-int(img[partial_row, partial_col][1])) \
                                         + abs(int(img[row, col][2])-int(img[partial_row, partial_col][2]))
                        g_s = math.exp(-1*(1/2)*(space_diff/sigma_s)*(space_diff/sigma_s))
                        g_r = math.exp(-1*(1/2)*(range_diff/sigma_r)*(range_diff/sigma_r))
                        sum_weight += g_s * g_r
                        color_b.append(img[partial_row, partial_col][0] * g_s * g_r)
                        color_g.append(img[partial_row, partial_col][1] * g_s * g_r)
                        color_r.append(img[partial_row, partial_col][2] * g_s * g_r)

            b_value = 0
            g_value = 0
            r_value = 0
            for i in range(0, kernel_size*kernel_size):
                b_value += (color_b[i] / sum_weight)
                g_value += (color_g[i] / sum_weight)
                r_value += (color_r[i] / sum_weight)

            img[row, col][0] = b_value
            img[row, col][1] = g_value
            img[row, col][2] = r_value

    return img


task1("test2_noise.png", "test2_clean.png", "test2_processed.png")

