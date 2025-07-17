import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from IPython.display import HTML, display


def tonemap_transient(transient, scaling=1.0):
    """Applies a linear tonemapping to the transient image."""
    channel_top = np.quantile(np.array(transient), 0.99)
    return transient / channel_top * scaling


def show_debug_video(
    weights_jbf,
    membership,
    final_weights,
    estimands,
    patches,
    estimands_variance,
    w_ij,
    axis_video=2,
    figsize=(15, 15),
):
    """
    Shows weights_jbf, membership, final_weights, tile, patches, estimands_variance, and w_ij as a combined video.

    :param weights_jbf: array of shape [H, W, T, 1] - bilateral weights
    :param membership: array of shape [H, W, T, 1] - membership values (0 or 1)
    :param final_weights: array of shape [H, W, T, 1] - final combined weights
    :param estimands: array of shape [H, W, T, C] - estimands
    :param patches: array of shape [H, W, T, C] - patches data
    :param estimands_variance: array of shape [H, W, T, C] - estimands_variance
    :param w_ij: array of shape [H, W, T, 3] - weights between 0.5 and 1 for 3 channels
    :param int axis_video: axis of the array for the temporal dimension (default: 2 for T)
    :param tuple figsize: figure size for the plot
    """

    def generate_index(axis_video, dims, index):
        return tuple(
            [np.s_[:] if dim != axis_video else np.s_[index] for dim in range(dims)]
        )

    # Remove the channel dimension for weight tensors (squeeze the last dimension)
    weights_jbf = weights_jbf.squeeze(-1)  # [H, W, T]
    membership = membership.squeeze(-1)  # [H, W, T]
    final_weights = final_weights.squeeze(-1)  # [H, W, T]

    num_frames = weights_jbf.shape[axis_video]

    # Create figure with 3 rows, 3 columns
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle("Debug Weights and Tiles Visualization", fontsize=16)

    # Initialize first frame for all tensors
    frame_weights = weights_jbf[generate_index(axis_video, len(weights_jbf.shape), 0)]
    frame_membership = membership[generate_index(axis_video, len(membership.shape), 0)]
    frame_final = final_weights[generate_index(axis_video, len(final_weights.shape), 0)]
    frame_estimands = estimands[generate_index(axis_video, len(estimands.shape), 0)]
    frame_patches = patches[generate_index(axis_video, len(patches.shape), 0)]
    frame_estimands_variance = estimands_variance[
        generate_index(axis_video, len(estimands_variance.shape), 0)
    ]
    frame_w_ij = w_ij[generate_index(axis_video, len(w_ij.shape), 0)]

    # Top row: weights
    im1 = axes[0, 0].imshow(frame_weights, cmap="viridis", vmin=0, vmax=1)
    axes[0, 0].set_title("Weights JBF")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 1].imshow(frame_membership, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Membership")
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im3 = axes[0, 2].imshow(frame_final, cmap="viridis", vmin=0, vmax=1)
    axes[0, 2].set_title("Final Weights")
    axes[0, 2].axis("off")
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Middle row: tiles (assuming RGB channels, take first 3 channels)
    # Clamp values to [0, 1] for proper display
    frame_estimands_rgb = np.clip(frame_estimands[..., :3], 0, 1)
    frame_patches_rgb = np.clip(frame_patches[..., :3], 0, 1)
    frame_estimands_variance_rgb = np.clip(frame_estimands_variance[..., :3], 0, 1)

    im4 = axes[1, 0].imshow(frame_estimands_rgb)
    axes[1, 0].set_title("Estimands")
    axes[1, 0].axis("off")

    im5 = axes[1, 1].imshow(frame_patches_rgb)
    axes[1, 1].set_title("Patches")
    axes[1, 1].axis("off")

    im6 = axes[1, 2].imshow(frame_estimands_variance_rgb)
    axes[1, 2].set_title("Estimands variance")
    axes[1, 2].axis("off")

    # Bottom row: w_ij channels (3 channels)
    im7 = axes[2, 0].imshow(frame_w_ij[..., 0], cmap="viridis", vmin=0.5, vmax=1)
    axes[2, 0].set_title("w_ij Channel 0")
    axes[2, 0].axis("off")
    plt.colorbar(im7, ax=axes[2, 0], fraction=0.046, pad=0.04)

    im8 = axes[2, 1].imshow(frame_w_ij[..., 1], cmap="viridis", vmin=0.5, vmax=1)
    axes[2, 1].set_title("w_ij Channel 1")
    axes[2, 1].axis("off")
    plt.colorbar(im8, ax=axes[2, 1], fraction=0.046, pad=0.04)

    im9 = axes[2, 2].imshow(frame_w_ij[..., 2], cmap="viridis", vmin=0.5, vmax=1)
    axes[2, 2].set_title("w_ij Channel 2")
    axes[2, 2].axis("off")
    plt.colorbar(im9, ax=axes[2, 2], fraction=0.046, pad=0.04)

    # Add frame counter
    frame_text = fig.text(
        0.5, 0.02, f"Frame: 0/{num_frames-1}", ha="center", fontsize=12
    )

    def update(i):
        frame_weights = weights_jbf[
            generate_index(axis_video, len(weights_jbf.shape), i)
        ]
        frame_membership = membership[
            generate_index(axis_video, len(membership.shape), i)
        ]
        frame_final = final_weights[
            generate_index(axis_video, len(final_weights.shape), i)
        ]
        frame_tile = estimands[generate_index(axis_video, len(estimands.shape), i)]
        frame_patches = patches[generate_index(axis_video, len(patches.shape), i)]
        frame_denoised = estimands_variance[
            generate_index(axis_video, len(estimands_variance.shape), i)
        ]
        frame_w_ij = w_ij[generate_index(axis_video, len(w_ij.shape), i)]

        # Update weight images
        im1.set_data(frame_weights)
        im2.set_data(frame_membership)
        im3.set_data(frame_final)

        # Update tile images (RGB channels)
        frame_tile_rgb = np.clip(frame_tile[..., :3], 0, 1)
        frame_patches_rgb = np.clip(frame_patches[..., :3], 0, 1)
        frame_denoised_rgb = np.clip(frame_denoised[..., :3], 0, 1)

        im4.set_data(frame_tile_rgb)
        im5.set_data(frame_patches_rgb)
        im6.set_data(frame_denoised_rgb)

        # Update w_ij channels
        im7.set_data(frame_w_ij[..., 0])
        im8.set_data(frame_w_ij[..., 1])
        im9.set_data(frame_w_ij[..., 2])

        frame_text.set_text(f"Frame: {i}/{num_frames-1}")

        return [im1, im2, im3, im4, im5, im6, im7, im8, im9, frame_text]

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames, repeat=True, interval=200
    )
    display(HTML(ani.to_html5_video()))
    plt.close()


def save_video(path, transient, axis_video, fps=24, display_video=False):
    """Saves the transient image in video format (.mp4)."""
    import cv2

    def generate_index(axis_video, dims, index):
        return tuple(
            [np.s_[:] if dim != axis_video else np.s_[index] for dim in range(dims)]
        )

    size = (transient.shape[1], transient.shape[0])
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    for i in range(transient.shape[axis_video]):
        frame = transient[generate_index(axis_video, len(transient.shape), i)]
        bitmap = mi.Bitmap(frame).convert(
            component_format=mi.Struct.Type.UInt8, srgb_gamma=True
        )
        out.write(np.array(bitmap)[:, :, ::-1])

    out.release()

    if display_video:
        from IPython.display import Video, display

        return display(Video(path, embed=True, width=size[0], height=size[1]))


def save_frames(data, axis_video, folder):
    """Saves the transient image in separate frames (.exr format for each frame)."""
    import os

    os.makedirs(folder, exist_ok=True)

    def generate_index(axis_video, dims, index):
        return tuple(
            [np.s_[:] if dim != axis_video else np.s_[index] for dim in range(dims)]
        )

    num_frames = data.shape[axis_video]
    for i in range(num_frames):
        mi.Bitmap(data[generate_index(axis_video, len(data.shape), i)]).write(
            f"{folder}/{i:03d}.exr"
        )


def show_video(input_sample, axis_video, uint8_srgb=True):
    """
    Shows the transient video in a IPython/Jupyter environment.

    :param input_sample: array representing the transient image
    :param int axis_video: axis of the array for the temporal dimension
    :param bool uint8_srgb: precision to use when converting to bitmap each frame of the video
    """
    # if not in_ipython():
    #     print("[show_video()] needs to be executed in a IPython/Jupyter environment")
    #     return

    import matplotlib.animation as animation
    import numpy as np
    from IPython.display import HTML, display
    from matplotlib import pyplot as plt

    def generate_index(axis_video, dims, index):
        return tuple(
            [np.s_[:] if dim != axis_video else np.s_[index] for dim in range(dims)]
        )

    num_frames = input_sample.shape[axis_video]
    fig = plt.figure()

    frame = input_sample[generate_index(axis_video, len(input_sample.shape), 0)]
    im = plt.imshow(mi.util.convert_to_bitmap(frame, uint8_srgb))
    plt.axis("off")

    def update(i):
        frame = input_sample[generate_index(axis_video, len(input_sample.shape), i)]
        img = mi.util.convert_to_bitmap(frame, uint8_srgb)
        im.set_data(img)
        return im

    ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)
    display(HTML(ani.to_html5_video()))
    plt.close()


def rainbow_visualization(
    steady_state,
    data_transient,
    modulo,
    min_modulo,
    max_modulo,
    max_time_bins=None,
    mode="peak_time_fusion",
    scale_fusion=1,
):
    import matplotlib.cm as cm

    # From: http://giga.cps.unizar.es/~ajarabo/pubs/MT/downloads/Jarabo2012_MasterThesis_noannex.pdf
    time_bins = data_transient.shape[2] if max_time_bins is None else max_time_bins

    # Compute the time bin index with the peak radiance for each pixel
    idx = np.argmax(np.max(data_transient, axis=-1), axis=-1)

    valid = (idx % modulo >= min_modulo) & (idx % modulo <= max_modulo)

    # Rainbow colors
    colors = cm.jet(idx / time_bins)[..., :3]

    # Output image: one color per pixel (H, W, channels)
    result = np.zeros_like(steady_state)
    if mode == "sparse_fusion":
        # Select the data at the peak index for each pixel
        result[valid] = steady_state[valid] ** scale_fusion
    elif mode == "rainbow_fusion":
        result[valid] = colors[valid]
    elif mode == "peak_time_fusion":
        result[valid] = colors[valid]
        result[~valid] = steady_state[~valid] ** scale_fusion
    else:
        raise NotImplementedError("Mode not implemented")

    return result
