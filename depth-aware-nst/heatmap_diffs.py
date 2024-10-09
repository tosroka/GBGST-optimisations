import cv2
import matplotlib.pyplot as plt
import numpy as np

# img1 = '../../../Unity/my_results_cvpr_2/starry_night/game_4_seed_scene_2/frame_0760.jpg'
# img2 = '../../../Unity/my_results_cvpr_2/starry_night/game_4_seed_scene_2/frame_0759.jpg'

# img1 = '../../my_results_cvpr_image/starry_night/game_4_seed_scene_2/frame_0760.png'
# img2 = '../../my_results_cvpr_image/starry_night/game_4_seed_scene_2/frame_0759.png'

img1 = '../../MCCNet/output_unity/starry_night/game_4_seed_scene_2_14/frame_0760.jpg'
img2 = '../../MCCNet/output_unity/starry_night/game_4_seed_scene_2_14/frame_0759.jpg'

# img1 = '../../unity_games_test/games/game_4_seed_scene_2/frame_0760.jpg'
# img2 = '../../unity_games_test/games/game_4_seed_scene_2/frame_0759.jpg'

# Load the images
image1 = cv2.imread(img1)
image2 = cv2.imread(img2)

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Compute the absolute difference
difference = cv2.absdiff(gray1, gray2)

# Normalize the difference image
norm_diff = cv2.normalize(difference, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Image dimensions
height, width = gray1.shape

# Set DPI for the output image
dpi = 100  # You can adjust this value as needed

# Calculate figure size in inches (Width x Height)
fig_size = width / float(dpi), height / float(dpi)

# Create a figure with the specified size and DPI
plt.figure(figsize=fig_size, dpi=dpi)

# Create a heatmap
plt.imshow(norm_diff, cmap='hot')
# plt.colorbar()
plt.axis('off')  # This disables the axis

# Save the heatmap
plt.savefig('images/heatmaps/fixed/760/MCCNet_game_4_seed_scene_2_frame_0760.png', bbox_inches='tight', pad_inches=0)
plt.close()
# plt.title('Heatmap of Differences')
# plt.show()
