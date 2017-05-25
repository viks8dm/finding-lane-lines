# Project on finding lane lines in test images and videos
# Term1-project on lanes lines for Udacity's nano-degree
# program on self-driving cars

# import necessary packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# apply grayscale transform
def grayscale(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# apply Gaussian smoothing
def gaussian_blur(gray, kernel_size=5, low_threshold=70, high_threshold=210):
    # values specified above are optimized for this project
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    # read parameters for Canny and apply
    return cv2.Canny(blur_gray, low_threshold, high_threshold)

# define mask (or region of interest) limits for a four sided polygon
def mask_vertices(xsize, ysize):
    left_bottom = [int(0.1 * xsize), ysize]
    left_top = [int(0.42 * xsize), int(0.55 * ysize)]
    right_top = [int(0.58 * xsize), int(0.55 * ysize)]
    right_bottom = [int(0.95 * xsize), ysize]
    return left_bottom, left_top, right_bottom, right_top

# define & create a masked edges for image using cv2.fillPoly()
def region_of_interest(image, edges, left_bottom, left_top, right_bottom, right_top):
    mask = np.zeros_like(edges)
    mask_for_image = np.zeros_like(image)
    ignore_mask_color = 255
    # define a four sided polygon to mask
    vertices = np.array([[left_bottom, left_top, right_top, right_bottom]], dtype=np.int32)
    # extract edges for lane lines
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    # extract masked region
    cv2.fillPoly(mask_for_image, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask_for_image)
    return masked_edges, masked_image

# Define the Hough transform parameters
# Run Hough on edge detected image
def lines_HoughTransform(image, masked_edges, edges):
    # Make a blank the same size as our image to draw on
    rho = 0.5  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 12  # minimum number of pixels making up a line
    max_line_gap = 2  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # Iterate over the output "lines" and draw lines on a blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if 0.3 < abs(slope) and abs(slope) < 0.9:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

    return color_edges, lines, lines_edges, line_image

# add original image and lane lines
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `imgage` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    The result image is computed as follows:
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

######## need to update
def draw_lines(image, lines, left_bottom, left_top, right_bottom, right_top, color=[255, 0, 0], thickness=10):
    """
    use slope to extract lines that might be close to actual lanes. This is
    not going to change significantly over time since camera location is fixed
    slope = ((y2-y1)/(x2-x1)).  Then, average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    positive slope is right lane, negative slope is left lane
    extrapolate from bottom to top of region of interest mask
    """
    right_slope_lines = []
    left_slope_lines = []
    c_right = []
    c_left = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2-y1)/(x2-x1)
            if 0.4<slope and slope<0.7:
                right_slope_lines.append(slope)
                c_right.append(y1-slope*x1)
                c_right.append(y2-slope*x2)
            elif -0.9<slope and slope<-0.5:
                left_slope_lines.append(slope)
                c_left.append(y1 - slope*x1)
                c_left.append(y2 - slope * x2)

    # find slope and constant for equation y=slope*x + constant
    right_poly = [np.mean(right_slope_lines), np.mean(c_right)]
    left_poly = [np.mean(left_slope_lines), np.mean(c_left)]
    # extrapolate data on right and left using y=m*x+c expression
    x_bottom_right = int((right_bottom[1] - right_poly[1])/right_poly[0])
    x_top_right = int((right_top[1]- right_poly[1]) / right_poly[0])
    x_bottom_left = int((left_bottom[1] - left_poly[1]) / left_poly[0])
    x_top_left = int((left_top[1] - left_poly[1]) / left_poly[0])
    # overlap lines on the image
    lane_image = np.copy(image) * 0  # creating a blank to draw lines on
    cv2.line(lane_image, (x_top_right, right_top[1]), (x_bottom_right, right_bottom[1]), color, thickness)
    cv2.line(lane_image, (x_top_left, left_top[1]), (x_bottom_left, left_bottom[1]), color, thickness)

    return lane_image

# main function for lane detection
def process_image(image):

    plt.imshow(image)
    plt.show()

    # read image parameters
    ysize = image.shape[0]
    xsize = image.shape[1]
    # apply grayscale transform
    gray = grayscale(image)
    # smoothen image and extract edges using Canny
    edges = gaussian_blur(gray)

    # mediamBlur for optional challenge video
    # output is open polygons, one can apply slope condition on edges
    # and extract just the edges that have slope along the car
    # blur_gray = cv2.medianBlur(image, 9)
    # edges = cv2.Canny(blur_gray, 20, 200)


    # define region of interest for masking (a four sided polygon)
    left_bottom, left_top, right_bottom, right_top = mask_vertices(xsize, ysize)
    # apply mask to image
    masked_edges, masked_image = region_of_interest(image, edges, left_bottom, left_top, right_bottom, right_top)
    # Run Hough on edge detected image (parameters definied within the function)
    color_edges, lines, lines_edges, line_image = lines_HoughTransform(image, masked_edges, edges)
    # extrapolate lines and draw on entire masked region of interest
    lane_image = draw_lines(image, lines, left_bottom, left_top, right_bottom, right_top)
    # combine extrapolated lane image with original image
    final_image = weighted_img(lane_image, image)

    return final_image



print("\n ========starting function============")
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

white_output = "white.mp4"
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

# challenge_output = 'extra.mp4'
# clip2 = VideoFileClip('challenge.mp4')
# challenge_clip = clip2.fl_image(process_image)
# challenge_clip.write_videofile(challenge_output, audio=False)

print("========end of function============ \n")

