import open3d as o3d

def merge_ply_files(file1, file2, file3, output_file):
    # Load the first PLY file
    mesh1 = o3d.io.read_triangle_mesh(file1)
    
    # Load the second PLY file
    mesh2 = o3d.io.read_triangle_mesh(file2)
    
    # Load the third PLY file
    mesh3 = o3d.io.read_triangle_mesh(file3)
    
    # Merge the meshes
    combined_mesh = mesh1 + mesh2 + mesh3
    
    # Save the merged mesh to a new PLY file
    o3d.io.write_triangle_mesh(output_file, combined_mesh)
    print(f"Merged mesh saved as '{output_file}'")

# Example usage
# merge_ply_files('/home/wa285/rds/hpc-work/Thesis/inference_analysis/vis/scene0011_00/gt_1.ply', '/home/wa285/rds/hpc-work/Thesis/inference_analysis/vis/scene0011_00/pred_27.ply', '/home/wa285/rds/hpc-work/Thesis/inference_analysis/vis/scene0011_00/mesh.ply', '/home/wa285/rds/hpc-work/Thesis/inference_analysis/vis/scene0011_00/output_merged.ply')
