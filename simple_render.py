#!/usr/bin/env python3
"""
Simple script to render BVH files - just run this!
"""

import sys
import os
sys.path.append('/home/skr/Downloads/AMP_MUL')

def render_walk():
    """Render the walking animation"""
    try:
        from fmbvh.visualization.render_bvh import render_bvh
        from fmbvh.bvh.parser import BVH
        
        print("Rendering walking animation...")
        bvh_obj = BVH('runs/amp_humanoid_walk.npy.bvh')
        
        render_bvh(
            bvh_obj,
            color='blue',
            save_to='walk_animation.png',
            fps=8.0,
            start=0,
            end=20,
            scale=1.0
        )
        print("Walk animation saved as: walk_animation.png")
        
    except Exception as e:
        print(f"Error rendering walk: {e}")

def render_run():
    """Render the running animation"""
    try:
        from fmbvh.visualization.render_bvh import render_bvh
        from fmbvh.bvh.parser import BVH
        
        print("Rendering running animation...")
        bvh_obj = BVH('runs/amp_humanoid_run.npy.bvh')
        
        render_bvh(
            bvh_obj,
            color='red',
            save_to='run_animation.png',
            fps=12.0,
            start=0,
            end=15,
            scale=1.0
        )
        print("Run animation saved as: run_animation.png")
        
    except Exception as e:
        print(f"Error rendering run: {e}")

if __name__ == "__main__":
    print("=== BVH File Renderer ===")
    print("This will create 2D images of your animations")
    print()
    
    # Check if files exist
    walk_file = 'runs/amp_humanoid_walk.npy.bvh'
    run_file = 'runs/amp_humanoid_run.npy.bvh'
    
    if os.path.exists(walk_file):
        render_walk()
    else:
        print(f"Walk file not found: {walk_file}")
    
    if os.path.exists(run_file):
        render_run()
    else:
        print(f"Run file not found: {run_file}")
    
    print("\nDone! Check for PNG files in the current directory.")
