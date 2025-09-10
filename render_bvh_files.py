#!/usr/bin/env python3
"""
Simple script to render BVH files using the fmbvh library
"""

import sys
import os
sys.path.append('/home/skr/Downloads/AMP_MUL')

from fmbvh.visualization.render_bvh import render_bvh
from fmbvh.bvh.parser import BVH
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for display

def render_bvh_to_image(bvh_file, output_file=None, fps=8.0, start=0, end=10):
    """
    Render BVH file to 2D image using matplotlib
    """
    try:
        print(f"Loading BVH file: {bvh_file}")
        bvh_obj = BVH(bvh_file)
        
        if output_file is None:
            output_file = bvh_file.replace('.bvh', '.png')
        
        print(f"Rendering to: {output_file}")
        print(f"FPS: {fps}, Frames: {start} to {end}")
        
        render_bvh(
            bvh_obj,
            color='midnightblue',
            save_to=output_file,
            fps=fps,
            start=start,
            end=end,
            scale=1.0,
            elev=2.0,
            azim=-90,
            hind_axis=True
        )
        
        print(f"Successfully rendered: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error rendering {bvh_file}: {e}")
        return False

def render_bvh_interactive(bvh_file):
    """
    Render BVH file in interactive 3D viewer
    """
    try:
        from fmbvh.visualization.utils import quick_visualize
        from fmbvh.motion_tensor.bvh_casting import get_quaternion_from_bvh, get_offsets_from_bvh
        from fmbvh.motion_tensor.rotations import quaternion_to_matrix
        from fmbvh.motion_tensor.kinematics import forward_kinematics
        import torch
        
        print(f"Loading BVH file for interactive view: {bvh_file}")
        bvh_obj = BVH(bvh_file)
        
        # Get skeleton structure
        p_index = bvh_obj.dfs_parent()
        
        # Get motion data
        trs, qua = get_quaternion_from_bvh(bvh_obj)
        off = get_offsets_from_bvh(bvh_obj)
        
        # Convert to positions using forward kinematics
        mat = quaternion_to_matrix(qua[None, ...])
        offsets = torch.broadcast_to(off[None, ...], (1, off.shape[0], 3, qua.shape[-1]))
        
        fk_pos = forward_kinematics(p_index, mat, trs[None, ...], offsets)
        
        # Add root translation
        pos = fk_pos[0] + trs[None, :, :]
        
        print("Starting interactive 3D viewer...")
        print("Close the window to exit")
        
        quick_visualize(p_index, pos, scale=200.0)
        
    except Exception as e:
        print(f"Error in interactive rendering: {e}")
        print("Make sure you have all required dependencies installed")

if __name__ == "__main__":
    # Get BVH files from runs directory
    runs_dir = "/home/skr/Downloads/AMP_MUL/runs"
    bvh_files = []
    
    if os.path.exists(runs_dir):
        for file in os.listdir(runs_dir):
            if file.endswith('.bvh'):
                bvh_files.append(os.path.join(runs_dir, file))
    
    if not bvh_files:
        print("No BVH files found in runs directory")
        print("Available files:")
        if os.path.exists(runs_dir):
            for file in os.listdir(runs_dir):
                print(f"  {file}")
        sys.exit(1)
    
    print("Found BVH files:")
    for i, file in enumerate(bvh_files):
        print(f"  {i+1}. {os.path.basename(file)}")
    
    print("\nChoose rendering method:")
    print("1. Render to 2D images (PNG)")
    print("2. Interactive 3D viewer")
    print("3. Both")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice in ['1', '3']:
        print("\n=== Rendering to 2D Images ===")
        for bvh_file in bvh_files:
            render_bvh_to_image(bvh_file, fps=8.0, start=0, end=15)
    
    if choice in ['2', '3']:
        print("\n=== Interactive 3D Viewer ===")
        if len(bvh_files) == 1:
            render_bvh_interactive(bvh_files[0])
        else:
            print("Multiple BVH files found. Choose one for interactive viewing:")
            for i, file in enumerate(bvh_files):
                print(f"  {i+1}. {os.path.basename(file)}")
            
            try:
                file_choice = int(input("Enter file number: ")) - 1
                if 0 <= file_choice < len(bvh_files):
                    render_bvh_interactive(bvh_files[file_choice])
                else:
                    print("Invalid choice")
            except ValueError:
                print("Invalid input")
    
    print("\nDone!")
