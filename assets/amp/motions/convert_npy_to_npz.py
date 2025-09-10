#!/usr/bin/env python3
"""
Convert .npy motion file to .npz format for motion viewer compatibility.
"""

import numpy as np
import os
import sys

def convert_npy_to_npz(npy_file_path, output_npz_path=None):
    """
    Convert .npy motion file to .npz format compatible with motion viewer.
    
    Args:
        npy_file_path: Path to input .npy file
        output_npz_path: Path for output .npz file (optional)
    """
    
    # Load the .npy file
    print(f"Loading .npy file: {npy_file_path}")
    data = np.load(npy_file_path, allow_pickle=True).item()
    
    # Extract data from the OrderedDict structure
    rotation = data['rotation']['arr']  # Shape: (frames, bodies, 4) - quaternions
    root_translation = data['root_translation']['arr']  # Shape: (frames, 3)
    global_velocity = data['global_velocity']['arr']  # Shape: (frames, bodies, 3)
    global_angular_velocity = data['global_angular_velocity']['arr']  # Shape: (frames, bodies, 3)
    skeleton_tree = data['skeleton_tree']
    body_names = skeleton_tree['node_names']  # List of body names
    fps = data['fps']
    
    num_frames, num_bodies = rotation.shape[:2]
    
    print(f"Motion data: {num_frames} frames, {num_bodies} bodies, {fps} FPS")
    print(f"Body names: {body_names}")
    
    # Compute body positions from root translation and local translations
    local_translations = skeleton_tree['local_translation']['arr']  # Shape: (bodies, 3)
    parent_indices = skeleton_tree['parent_indices']['arr']  # Shape: (bodies,)
    
    # Initialize body positions array
    body_positions = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
    
    # Set root body position (pelvis) to root_translation
    body_positions[:, 0, :] = root_translation
    
    # Compute positions for other bodies using forward kinematics
    for frame in range(num_frames):
        for body_idx in range(1, num_bodies):  # Skip root (index 0)
            parent_idx = parent_indices[body_idx]
            if parent_idx >= 0:  # Has parent
                # Get parent's global rotation and position
                parent_rot = rotation[frame, parent_idx]  # Quaternion (w, x, y, z)
                parent_pos = body_positions[frame, parent_idx]
                
                # Convert quaternion to rotation matrix (simplified)
                # This is a basic implementation - for production use a proper quaternion library
                w, x, y, z = parent_rot
                R = np.array([
                    [1-2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                    [2*(x*y + w*z), 1-2*(x*x + z*z), 2*(y*z - w*x)],
                    [2*(x*z - w*y), 2*(y*z + w*x), 1-2*(x*x + y*y)]
                ])
                
                # Transform local translation to global space
                local_trans = local_translations[body_idx]
                global_trans = parent_pos + R @ local_trans
                body_positions[frame, body_idx] = global_trans
    
    # Create dummy DOF data (required by motion viewer but not used for visualization)
    # We'll create minimal DOF data based on the number of bodies
    num_dofs = max(1, num_bodies - 1)  # At least 1 DOF
    dof_names = [f"dof_{i}" for i in range(num_dofs)]
    
    # Create dummy DOF positions and velocities (zeros)
    dof_positions = np.zeros((num_frames, num_dofs), dtype=np.float32)
    dof_velocities = np.zeros((num_frames, num_dofs), dtype=np.float32)
    
    # Prepare the output data
    output_data = {
        'fps': np.array(fps, dtype=np.int64),
        'dof_names': np.array(dof_names, dtype='<U16'),
        'body_names': np.array(body_names, dtype='<U15'),
        'dof_positions': dof_positions,
        'dof_velocities': dof_velocities,
        'body_positions': body_positions.astype(np.float32),
        'body_rotations': rotation.astype(np.float32),
        'body_linear_velocities': global_velocity.astype(np.float32),
        'body_angular_velocities': global_angular_velocity.astype(np.float32)
    }
    
    # Determine output path
    if output_npz_path is None:
        base_name = os.path.splitext(os.path.basename(npy_file_path))[0]
        output_npz_path = f"{base_name}_converted.npz"
    
    # Save as .npz file
    print(f"Saving converted data to: {output_npz_path}")
    np.savez(output_npz_path, **output_data)
    
    print("Conversion completed successfully!")
    print(f"Output file: {output_npz_path}")
    print(f"Data shapes:")
    for key, value in output_data.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    return output_npz_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_npy_to_npz.py <input.npy> [output.npz]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    try:
        convert_npy_to_npz(input_file, output_file)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)
