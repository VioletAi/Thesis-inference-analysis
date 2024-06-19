"""
Snap Script

Author: Zhening Huang (zh340@cam.ac.uk)
"""

import torch
import numpy as np
import torch
from pytorch3d.io import IO
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    AmbientLights,
    HardPhongShader,
    BlendParams,
)
import pytorch3d
from PIL import Image
import numpy as np
import copy
import os


def get3d_box_from_pcs(pc):
    """
    Given point-clouds that represent object or scene return the 3D dimension of the 3D box that contains the PCs.
    """
    w = pc[:, 0].max() - pc[:, 0].min()
    l = pc[:, 1].max() - pc[:, 1].min()
    h = pc[:, 2].max() - pc[:, 2].min()
    return w, l, h


def lookat(center, target, up):
    """
    https://github.com/isl-org/Open3D/issues/2338
    https://stackoverflow.com/questions/54897009/look-at-function-returns-a-view-matrix-with-wrong-forward-position-python-im
    https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function
    https://www.youtube.com/watch?v=G6skrOtJtbM
    f: forward
    s: right
    u: up
    """
    f = target - center
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    u = u / np.linalg.norm(u)

    m = np.zeros((4, 4))
    m[0, :-1] = -s
    m[1, :-1] = u
    m[2, :-1] = f
    m[-1, -1] = 1.0

    t = np.matmul(-m[:3, :3], center)
    m[:3, 3] = t

    return m


def get_rid_of_lip(mesh, scan_pc, remove_lip):
    
    """
    Given a mesh file in pytorch3d, remove the lip with predefined height, return a mesh file in pytorch3d
    """
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    texture_tensor = mesh.textures.verts_features_packed()
    a = verts
    b = faces
    z_max = scan_pc[:, 2].max()
    idx = a[:, 2] <= z_max - remove_lip

    # Crop the vertices and create an index map
    map_idx = torch.zeros_like(a[:, 0], dtype=torch.long).cuda() - 1
    map_idx[idx] = torch.arange(idx.sum()).cuda()
    a = a[idx]
    texture_tensor = texture_tensor[idx]

    # Crop the triangle surface and update the indices
    b = b[(idx[b[:, 0]] & idx[b[:, 1]] & idx[b[:, 2]])]
    final_b = map_idx[b]

    converted_texture = pytorch3d.renderer.mesh.textures.TexturesVertex(
        [texture_tensor]
    )
    cropped_mesh = pytorch3d.structures.Meshes(
        verts=[a], faces=[final_b], textures=converted_texture
    ).cuda()
    return cropped_mesh


def intrinsic_calibration(point_cloud, pose, width, height):
    """
    Calibrate intrinsic matrix 
    """

    depth_intrinsic = np.array(
        [
            [577.590698, 0.000000, 318.905426, 0.000000],
            [0.000000, 578.729797, 242.683609, 0.000000],
            [0.000000, 0.000000, 1.000000, 0.000000],
            [0.000000, 0.000000, 0.000000, 1.000000],
        ]
    )

    fx = depth_intrinsic[0, 0]
    fy = depth_intrinsic[1, 1]
    cx = depth_intrinsic[0, 2]
    cy = depth_intrinsic[1, 2]

    points = np.ones((point_cloud.shape[0], 4))
    points[:, :3] = point_cloud[:, :3]
    inv_pose = np.linalg.inv(np.transpose(pose))
    points_new = np.dot(points, inv_pose)

    point_projected = np.zeros((points_new.shape[0], 2))
    point_projected[:, 0] = points_new[:, 0] * fx / points_new[:, 2] + cx
    point_projected[:, 1] = points_new[:, 1] * fy / points_new[:, 2] + cy
    new_intrinsic = depth_intrinsic.copy()
    cx_new = new_intrinsic[0, 2] - point_projected[:, 0].min()
    cy_new = new_intrinsic[1, 2] - point_projected[:, 1].min()

    point_projected[:, 0] = points_new[:, 0] * fx / points_new[:, 2] + cx_new
    point_projected[:, 1] = points_new[:, 1] * fy / points_new[:, 2] + cy_new

    scale_1 = 1 / point_projected[:, 0].max() * width
    scale_2 = 1 / point_projected[:, 1].max() * height
    scale = scale_1 if scale_1 < scale_2 else scale_2
    fx_new = new_intrinsic[0, 0] * scale
    fy_new = new_intrinsic[1, 1] * scale
    cx_new = cx_new * scale
    cy_new = cy_new * scale

    point_projected_new = np.zeros((points_new.shape[0], 2))
    point_projected_new[:, 0] = points_new[:, 0] * fx_new / points_new[:, 2] + cx_new
    point_projected_new[:, 1] = points_new[:, 1] * fy_new / points_new[:, 2] + cy_new

    assert point_projected_new[:, 0].max() <= width + 1
    assert point_projected_new[:, 0].min() > -0.1
    assert point_projected_new[:, 1].max() <= height + 1
    assert point_projected_new[:, 1].min() > -0.1

    new_intrinsic = depth_intrinsic.copy()

    new_intrinsic[0, 0] = fx_new
    new_intrinsic[1, 1] = fy_new
    new_intrinsic[0, 2] = cx_new
    new_intrinsic[1, 2] = cy_new

    return new_intrinsic


def generate_camera_locations(center, width, length, height, num_split=5):
    """
    Generate the camera position for scene level images
    """
    half_width, half_length, half_height = width / 2, length / 2, height / 2
    top_height = center[0] + half_height
    top_coord = np.linspace(center[0] - half_width, center[0] + half_width, num_split)
    ver_coord = np.linspace(center[1] - half_length, center[1] + half_length, num_split)

    camera_pos_from = []
    for x_coord in top_coord:
        camera_pos_from.append([x_coord, ver_coord[0], top_height])
        camera_pos_from.append([x_coord, ver_coord[-1], top_height])
    for y_coord in ver_coord[1:-1]:
        camera_pos_from.append([top_coord[0], y_coord, top_height])
        camera_pos_from.append([top_coord[-1], y_coord, top_height])
    return camera_pos_from


def render(pose, intrin_path, image_width, image_height, mesh, name):
    device = "cuda"
    background_color = (1.0, 1.0, 1.0)
    intrinsic_matrix = torch.tensor(
        [
            [577.590698, 0.000000, 318.905426, 0.000000],
            [0.000000, 578.729797, 242.683609, 0.000000],
            [0.000000, 0.000000, 1.000000, 0.000000],
            [0.000000, 0.000000, 0.000000, 1.000000],
        ]
    )
    intrinsic_matrix_load = intrin_path
    intrinsic_matrix_load_torch = torch.from_numpy(intrinsic_matrix_load)
    intrinsic_matrix[:3, :3] = intrinsic_matrix_load_torch
    extrinsic_load = pose
    camera_to_world = torch.from_numpy(extrinsic_load)
    world_to_camera = torch.inverse(camera_to_world)
    fx, fy, cx, cy = (
        intrinsic_matrix[0, 0],
        intrinsic_matrix[1, 1],
        intrinsic_matrix[0, 2],
        intrinsic_matrix[1, 2],
    )
    width, height = image_width, image_height
    rotation_matrix = world_to_camera[:3, :3].permute(1, 0).unsqueeze(0)
    translation_vector = world_to_camera[:3, 3].reshape(-1, 1).permute(1, 0)
    focal_length = -torch.tensor([[fx, fy]])
    principal_point = torch.tensor([[cx, cy]])
    camera = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=rotation_matrix,
        T=translation_vector,
        image_size=torch.tensor([[height, width]]),
        in_ndc=False,
        device=device,
    )
    lights = AmbientLights(device=device)
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=HardPhongShader(
            blend_params=BlendParams(background_color=background_color),
            device=lights.device,
            cameras=camera,
            lights=lights,
        ),
    )
    rendered_image = renderer(mesh)
    rendered_image = rendered_image[0].cpu().numpy()
    color = rendered_image[..., :3]
    color_image = Image.fromarray((color * 255).astype(np.uint8))
    color_image.save(name)


def image_generation(mesh_file, adjust_camera, image_width, image_height, folder_saved):
    device = "cuda"
    pt3d_io = IO()
    mesh = pt3d_io.load_mesh(mesh_file, device=device)
    scan_pc = mesh.verts_packed().cpu().numpy()

    lift_cam, zoomout, remove_lip = adjust_camera

    mesh = get_rid_of_lip(mesh, scan_pc, remove_lip)

    w, l, h = get3d_box_from_pcs(scan_pc)
    scene_center = np.array(
        [
            scan_pc[:, 0].max() - w / 2,
            scan_pc[:, 1].max() - l / 2,
            scan_pc[:, 2].max() - h / 2,
        ]
    )
    zoom_factor = 1 + zoomout
    w, l, h = w * zoom_factor, l * zoom_factor, h * zoom_factor
    camera_locations = generate_camera_locations(scene_center, w, l, h, 5)

    for i in range(len(camera_locations)):
        if i==0 or i==5 or i==10 or i==15:
            camera_location = camera_locations[i]
            org_camera_pos = copy.deepcopy(camera_location)
            camera_location[-1] = org_camera_pos[-1] + lift_cam  # lift the camera
            target_location = scene_center
            up_vector = np.array([0, 0, -1])
            pose_matrix = lookat(camera_location, target_location, up_vector)
            pose_matrix_calibrated = np.transpose(np.linalg.inv(np.transpose(pose_matrix)))
            intrinsic_calibrated = intrinsic_calibration(
                scan_pc, pose_matrix_calibrated, image_width, image_height
            )
            intrinsic_folder = f"{folder_saved}/intrinsic"
            pose_folder = f"{folder_saved}/pose"
            image_folder = f"{folder_saved}/image"
            if not os.path.exists(intrinsic_folder):
                os.makedirs(intrinsic_folder)
            if not os.path.exists(pose_folder):
                os.makedirs(pose_folder)
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            np.save(
                f"{intrinsic_folder}/intrinsic_calibrated_angle_{i}.npy",
                intrinsic_calibrated,
            )
            np.save(
                f"{pose_folder}/pose_matrix_calibrated_angle_{i}.npy",
                pose_matrix_calibrated,
            )
            render(
                pose_matrix_calibrated,
                intrinsic_calibrated[:3, :3],
                image_width,
                image_height,
                mesh,
                f"{image_folder}/image_rendered_angle_{i}.png",
            )
    return False

def snapshot_runner(mesh_file,folder_saved):
    image_width = 2000
    image_height = 2000

    # adjust the camera

    lift_cam = 3  # unit m
    zoomout = 0.1
    remove_lip = 0.3  # unit m
    adjust_camera = [lift_cam, zoomout, remove_lip]

    image_generation(mesh_file, adjust_camera, image_width, image_height, folder_saved)



def main():
    image_width = 2000
    image_height = 2000
    mesh_file = ("/home/wa285/rds/hpc-work/Thesis/inference_analysis/vis/scene0011_00/output_merged.ply")
    folder_saved = "./export"
    # adjust the camera

    lift_cam = 3  # unit m
    zoomout = 0.1
    remove_lip = 0.3  # unit m
    adjust_camera = [lift_cam, zoomout, remove_lip]

    image_generation(mesh_file, adjust_camera, image_width, image_height, folder_saved)

if __name__ == "__main__":
    main()
