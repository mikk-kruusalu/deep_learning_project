import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataset import load_data


def conv2d(image, kernel):
    fw = kernel.shape[0]
    half_window = fw // 2
    n, m = image.shape

    # zero pad the image so the output is the same shape
    out = torch.zeros(image.shape)
    image = F.pad(image, (half_window, half_window, half_window, half_window))
    for i in range(half_window, n):
        for j in range(half_window, m):
            out[i, j] = torch.sum(
                image[(i - half_window) : (i + half_window + 1), (j - half_window) : (j + half_window + 1)]
                * kernel
            )

    return out


def horiz_edge_kernel(size: int):
    max_value = size // 2
    kernel = torch.zeros((size, size))

    for i in range(kernel.shape[0]):
        kernel[i] = torch.ones(size) * (i - max_value)

    return kernel


def gaussian_blur_kernel(size: int, sigma: float = 1):
    max_dist = size // 2
    sigma = torch.tensor(sigma)

    distances = (
        torch.arange(-max_dist, max_dist + 1).repeat(size, 1) ** 2
        + torch.arange(-max_dist, max_dist + 1).view(size, 1).repeat(1, size) ** 2
    )

    kernel = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(-distances / (2 * sigma**2))
    kernel /= kernel.max()
    return kernel


def sharpening_filter(size: int):
    kc = size // 2

    kernel = -torch.ones((size, size))
    kernel[kc, kc] = (size - 1) ** 2

    return kernel


if __name__ == "__main__":
    data, _ = load_data()

    imgs = []
    for i in torch.randint(0, len(data), (3,)):
        imgs.append(data[i][0][0])

    print(f"Size of images: {imgs[0].shape}")
    print("Horizontal edge detection")
    print(horiz_edge_kernel(5))
    print("Gaussian blur")
    print(gaussian_blur_kernel(5, 1))
    print("Sharpening filter")
    print(sharpening_filter(5))

    fig, ax = plt.subplots(3, 10, constrained_layout=True, figsize=(16, 6))

    for i in range(3):
        ax[i, 0].imshow(imgs[i], cmap="gray")

        ax[i, 1].imshow(conv2d(imgs[i], horiz_edge_kernel(3)), cmap="gray")
        ax[i, 2].imshow(conv2d(imgs[i], horiz_edge_kernel(5)), cmap="gray")
        ax[i, 3].imshow(conv2d(imgs[i], horiz_edge_kernel(7)), cmap="gray")

        ax[i, 4].imshow(conv2d(imgs[i], gaussian_blur_kernel(3)), cmap="gray")
        ax[i, 5].imshow(conv2d(imgs[i], gaussian_blur_kernel(5)), cmap="gray")
        ax[i, 6].imshow(conv2d(imgs[i], gaussian_blur_kernel(7)), cmap="gray")

        ax[i, 7].imshow(conv2d(imgs[i], sharpening_filter(3)), cmap="gray")
        ax[i, 8].imshow(conv2d(imgs[i], sharpening_filter(5)), cmap="gray")
        ax[i, 9].imshow(conv2d(imgs[i], sharpening_filter(7)), cmap="gray")

    for a in ax.ravel():
        # a.set_axis_off()
        a.get_xaxis().set_ticks([])
        a.get_yaxis().set_ticks([])

    ax[0, 0].set_title("Original image")
    ax[0, 2].set_title("Horizontal edge detection")
    ax[0, 5].set_title("Gaussian blur")
    ax[0, 8].set_title("Sharpening filter")
    fig.suptitle("Manual convolution on 3 random images from the dataset")

    for i, k in zip(range(1, 4), [3, 5, 7]):
        ax[2, i].set_xlabel(f"{k}x{k}")
        ax[2, i + 3].set_xlabel(f"{k}x{k}")
        ax[2, i + 6].set_xlabel(f"{k}x{k}")

    plt.savefig("manual_convolution.png")
    plt.show()
