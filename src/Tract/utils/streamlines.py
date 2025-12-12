import math
import torch
import matplotlib.pyplot as plt

def generate_rotation_matrix_torch(axis: int, angle_deg: float, device: torch.device) -> torch.Tensor:
    """Generates a 4x4 rotation matrix as a PyTorch tensor for a given axis and angle."""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    matrix = torch.eye(4, dtype=torch.float32, device=device)

    if axis == 0:  # X-axis rotation
        matrix[1, 1] = cos_a
        matrix[1, 2] = -sin_a
        matrix[2, 1] = sin_a
        matrix[2, 2] = cos_a
    elif axis == 1:  # Y-axis rotation
        matrix[0, 0] = cos_a
        matrix[0, 2] = sin_a
        matrix[2, 0] = -sin_a
        matrix[2, 2] = cos_a
    elif axis == 2:  # Z-axis rotation
        matrix[0, 0] = cos_a
        matrix[0, 1] = -sin_a
        matrix[1, 0] = sin_a
        matrix[1, 1] = cos_a
    else:
        raise ValueError("Axis must be 0 (x), 1 (y), or 2 (z)")
    return matrix

def apply_affine_transform_torch(coords: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Applies a 4x4 affine transformation matrix to a set of 3D coordinates using PyTorch.
    Expects coords tensor shape (N, P, 3).
    """
    if coords.numel() == 0:
        return coords

    original_shape = coords.shape
    # Reshape for matrix multiplication: (N * P, 3)
    coords_flat = coords.view(-1, 3)

    # Extract rotation part of the 4x4 matrix
    rotation_matrix = matrix[:3, :3]

    # Apply rotation
    transformed_flat = coords_flat @ rotation_matrix

    # Reshape back to the original shape (N, P, 3)
    return transformed_flat.view(original_shape)



def visualize_streamlines(streamlines: torch.Tensor):
    """Helper function to plot the generated streamlines in 3D."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Move tensor to CPU and convert to numpy for plotting
    streamlines_np = streamlines.cpu().numpy()

    for i in range(streamlines_np.shape[0]):
        x = streamlines_np[i, :, 0]
        y = streamlines_np[i, :, 1]
        z = streamlines_np[i, :, 2]
        ax.plot(x, y, z)

    ax.set_title("Generated Dummy Streamlines")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.view_init(elev=20., azim=-35)
    plt.savefig('')
    plt.close()
