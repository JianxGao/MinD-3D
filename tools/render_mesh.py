import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import sys
import urllib.request
from urllib.request import urlopen
import ssl
import json
ssl._create_default_https_context = ssl._create_unverified_context
sys.path.append("/home/gjx/anaconda3/envs/mesh/lib/python3.10/site-packages")

import cv2

"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
import uuid
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np
import pdb
import bpy
from mathutils import Vector
# import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    # required=True,
    help="Path to the object file",
)
parser.add_argument("--input_dir", type=str, default="fmri_shapeshape_sub02_30k")
parser.add_argument("--output_dir", type=str, default="fmri_shapeshape_sub02_30k")
parser.add_argument(
    "--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=12)
parser.add_argument("--camera_dist", type=float, default=2)

parser.add_argument('--class_id', type=str, default=0)
parser.add_argument('--fov', type=int, default=0)
parser.add_argument('--video_id', type=int, default=0)
parser.add_argument('--index', type=int, default=0)


argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.data.angle = math.radians(args.fov)

# setup lighting

# bpy.ops.object.select_by_type(type='LIGHT')
# bpy.ops.object.delete()

bpy.ops.object.light_add(type="AREA")
light2 = bpy.data.lights["Area"]
light2.energy = 8000
# light2.size=0.5
# bpy.data.objects["Area"].location[0] = 3
# bpy.data.objects["Area"].location[1] = -4
# bpy.data.objects["Area"].location[2] = 5

bpy.data.objects["Area"].scale[0] = 1
bpy.data.objects["Area"].scale[1] = 1
bpy.data.objects["Area"].scale[2] = 1

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 1024
render.resolution_y = 1024
#render.resolution_x = beta_size#32
#render.resolution_y = render.resolution_x#32
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True
phi_angles = [math.radians(60), math.radians(90), math.radians(120)]
bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = "CUDA" # or "OPENCL"

sphere_sample_list = [[0, -1, 0],  # Front View
                      [-1, 0, 0],  # Side View (left
                      [0, 0, 1],   # Top View
                      [1, 0, 0],   # Back View
                      [0, 1, 0],   # Side View(right
                      [0, 0, -1],  # Bottle View
                      [0, -1 / math.sqrt(2), 1 / math.sqrt(2)],
                      [math.sqrt(3) / 2, 0.5, 1 / math.sqrt(2)],
                      [-math.sqrt(3) / 2, 0.5, 1 / math.sqrt(2)],
                      [0, -1 / math.sqrt(2), -1 / math.sqrt(2)],
                      [math.sqrt(3) / 2, 0.5, -1 / math.sqrt(2)],
                      [-math.sqrt(3) / 2, 0.5, -1 / math.sqrt(2)]
                      ]
distance_coe = 2.0
camera_location_list = [[distance_coe * element for element in row] for row in sphere_sample_list]


def randomize_lighting() -> None:
    light2.energy = 10000 #random.uniform(300, 600)
    # bpy.data.objects["Area"].location[0] = random.uniform(-1., 1.)
    # bpy.data.objects["Area"].location[1] = random.uniform(-1., 1.)
    # bpy.data.objects["Area"].location[2] = random.uniform(0.5, 1.5)

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def recalculate_normals():
    for obj in scene_meshes():
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')

# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    elif object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

    # Get all imported objects (usually the newly imported objects are selected)
    imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not imported_objects:
        raise ValueError("No mesh object was found after import.")
    
    # Loop through all imported objects and normalize each of them
    for imported_object in imported_objects:
        # Calculate the current maximum dimension of the object
        max_dimension = max(imported_object.dimensions)
        
        # Set the target dimension for normalization (e.g., 1 unit)
        target_dimension = 1.0

        # Calculate the scale factor needed to normalize the size
        if max_dimension > 0:
            scale_factor = target_dimension / max_dimension
            imported_object.scale = (scale_factor, scale_factor, scale_factor)

        # Apply the transformations (including scale)
        bpy.context.view_layer.objects.active = imported_object
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        # 重置旋转为 (0, 0, 0)
        imported_object.rotation_euler = (0, 0, 0)

        # 应用旋转
        bpy.context.view_layer.objects.active = imported_object
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Optionally, move the normalized object to the center of the scene
    for imported_object in imported_objects:
        imported_object.location = (0, 0, 0)
    recalculate_normals()


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def save_images(object_file: str) -> None:
    print(object_file)
    """Saves rendered images of the object in the scene."""
    class_name = object_file.split("/")[-2]
    object_name = object_file.split("/")[-1]
    # import pdb;pdb.set_trace()
    output_dir = os.path.join("/ssd/gaojianxiong/evaluation_3d/", args.output_dir, object_name)
    os.makedirs(output_dir, exist_ok=True)

    reset_scene()
    # load the object
    load_object(object_file)
    normalize_scene()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = (0, 0, 0)

    # cam.data.clip_start = 0.1
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    cam_constraint.target = empty

    # light = bpy.data.objects["Area"]
    # track_to_constraint = light.constraints.new(type='TRACK_TO')
    # track_to_constraint.target = empty
    # track_to_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    # track_to_constraint.up_axis = 'UP_Y'

    light = bpy.data.objects["Area"]
    track_to_constraint = light.constraints.new(type='TRACK_TO')
    track_to_constraint.target = empty
    track_to_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    track_to_constraint.up_axis = 'UP_Y'

    azimuths = [30, 90, 150, 210, 270, 330]
    elevations = [20, -10, 20, -10, 20, -10]
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()

    render_layers = tree.nodes.new('CompositorNodeRLayers')

    view_layer = scene.view_layers[0]
    view_layer.use_pass_normal = True
    view_layer.use_pass_z = True


    file_output_alpha = tree.nodes.new(type="CompositorNodeOutputFile")
    file_output_alpha.label = 'Alpha Output'
    file_output_alpha.base_path = output_dir
    file_output_alpha.format.file_format = 'PNG'
    file_output_alpha.format.color_mode = "RGBA"
    tree.links.new(render_layers.outputs['Alpha'], file_output_alpha.inputs[0])

    file_output_depth = tree.nodes.new(type="CompositorNodeOutputFile")
    file_output_depth.label = 'Depth Output'
    file_output_depth.base_path = output_dir
    file_output_depth.format.file_format = 'OPEN_EXR'
    file_output_depth.format.color_mode = "RGBA"
    file_output_depth.format.color_depth = "32"
    # file_output_depth.file_slots[0].use_node_format = True
    tree.links.new(render_layers.outputs['Depth'], file_output_depth.inputs[0]) 

    file_output_normal = tree.nodes.new(type="CompositorNodeOutputFile")
    file_output_normal.label = 'Normal Output'
    file_output_normal.base_path = output_dir
    file_output_normal.format.file_format = 'PNG'
    file_output_normal.format.color_mode = "RGBA"
    tree.links.new(render_layers.outputs['Normal'], file_output_normal.inputs[0])


    bpy.context.scene.render.film_transparent = True

    for i, (azimuth, elevation) in enumerate(zip(azimuths, elevations)):
        azimuth_rad = np.deg2rad(azimuth)
        elevation_rad = np.deg2rad(elevation)
        # azimuth_rad = math.radians(azimuth)
        # elevation_rad = math.radians(elevation)

        x = args.camera_dist * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = args.camera_dist * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = args.camera_dist * np.sin(elevation_rad)
        point = (x,y,z)
        cam.location = point

        # Make the camera look at the origin
        direction = -cam.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam.rotation_euler = rot_quat.to_euler()

        light_point = (
            1.5 * args.camera_dist * np.cos(elevation_rad) * np.cos(azimuth_rad),
            1.5 * args.camera_dist * np.cos(elevation_rad) * np.sin(azimuth_rad),
            1.5 * args.camera_dist * np.sin(elevation_rad)
        )
        light.location = light_point
        light.data.size = 50

        # render the image
        render_path = os.path.join(output_dir, f"{i}.png")
        scene.render.filepath = render_path

        file_output_alpha.base_path = output_dir

        file_output_alpha.file_slots[0].path = f"{i}_alpha_"
        # file_output_alpha.filepath = os.path.join(output_dir, f"alpha_{i}.png")
        # file_output_alpha.format.file_format = 'PNG'

        file_output_depth.base_path = output_dir
        # file_output_depth.filepath = os.path.join(output_dir, f"depth_{i}.png")
        file_output_depth.file_slots[0].path = f"{i}_depth_"
        # file_output_depth.file_slots[0].use_node_format = False
        # file_output_depth.format.file_format = 'PNG'

        file_output_normal.base_path = output_dir
        # file_output_normal.filepath = os.path.join(output_dir, f"_{i}.png")
        file_output_normal.file_slots[0].path = f"{i}_normal_"
        # file_output_normal.file_slots[0].use_node_format = False
        # file_output_normal.format.file_format = 'PNG'

        bpy.ops.render.render(write_still=True)

        # # save camera RT matrix
        # RT = get_3x4_RT_matrix_from_blender(cam)
        # RT_path = os.path.join(output_dir, f"{i}.npy")
        # np.save(RT_path, RT)
        depth_file_path = os.path.join(output_dir, f"{i}_depth_0001.exr")
        depth = cv2.imread(depth_file_path, -1)[..., 0]
        depth_max = depth.max()
        depth_min = depth.min()
        disp = 1 / depth
        disp_range = disp.max() - disp.min()
        cv2.imwrite(os.path.join(output_dir, f"{i}_depth.png"), (disp - disp.min()) / (disp.max() - disp.min()) * 255)
        os.system("rm -rf {}".format(depth_file_path))
        os.system("mv {} {}".format(
            os.path.join(output_dir, f"{i}_alpha_0001.png"),
            os.path.join(output_dir, f"{i}_alpha.png")
        ))
        os.system("mv {} {}".format(
            os.path.join(output_dir, f"{i}_normal_0001.png"),
            os.path.join(output_dir, f"{i}_normal.png")
        ))


def get_3x4_RT_matrix_from_blender(cam):
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1*R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT



if __name__ == "__main__":
    args.input_dir = os.path.join("~/InstantMeshInfer/outputs", args.input_dir, "meshes")
    sid_list = os.listdir(args.input_dir)
    sid_list.sort()
    for sid in sid_list[:]:
        if ".obj" not in sid:
            continue
        save_images(f"{args.input_dir}/{sid}")