import torch
import matplotlib.pyplot as plt
from bilinear_rescaling import bilinear_rescaling_matrix, apply_transformation_matrix
import random
from typing import Tuple
from time import time

def generate_batch_with_squares(
        batch_size: int,
        image_size: Tuple[int, int] = (128, 320),
        num_squares_range: Tuple[int, int] = (3, 6),
        square_size_range: Tuple[int, int] = (10, 30),
        device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate a batch of grayscale images containing randomly placed white squares.

    Parameters:
    -----------
    batch_size : int
        Number of images to generate.

    image_size : Tuple[int, int]
        Size of each image as (height, width).

    num_squares_range : Tuple[int, int]
        Min and max number of squares to place per image.

    square_size_range : Tuple[int, int]
        Min and max side length of each square.

    device : str
        The device for the result.

    Returns:
    --------
    torch.Tensor
        Tensor of shape (B, H, W) containing the generated images.
    """
    H, W = image_size
    images = torch.zeros((batch_size, H, W), dtype=torch.float32, device=device)

    for b in range(batch_size):
        num_squares = random.randint(*num_squares_range)
        for _ in range(num_squares):
            size = random.randint(*square_size_range)
            top = random.randint(0, H - size)
            left = random.randint(0, W - size)
            images[b, top:top + size, left:left + size] = 1.0  # white square

    return images

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    device = 'cuda:0'
    n_control_points = 180
    n_images = 128
    img_size = (128, 320)
    img_center = (img_size[0] / 2, img_size[1] / 2)

    scale_factors = torch.linspace(0.7, 1.3, n_images)
    leaf_positions = generate_batch_with_squares(n_control_points, image_size=img_size, device=device)


    transformation_matrices = []

    for i in range(n_images):
        matrix = bilinear_rescaling_matrix(img_size, img_center, float(scale_factors[i]), device=device)
        transformation_matrices.append(matrix)
        print(i)


    # Perform transformations
    scale_index = 127
    transformed_leafs = apply_transformation_matrix(leaf_positions, transformation_matrices[scale_index])

    plt.imshow(leaf_positions[0, :, :].cpu().numpy()), plt.show()
    plt.imshow(transformed_leafs[0, :, :].cpu().numpy()), plt.show()

    transformed_leafs = torch.zeros((n_images, n_control_points, *img_size), dtype=torch.float32, device=device)
    torch.cuda.synchronize()
    start = time()

    for idx, transform_matrix in enumerate(transformation_matrices):
        transformed_leafs[idx, ...] = apply_transformation_matrix(leaf_positions, transform_matrix)

    torch.cuda.synchronize()
    end = time()

    print(f'Total time = {end - start} s')




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
