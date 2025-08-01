cd InstantMesh
conda activate mind3dpp
CUDA_VISIBLE_DEVICES=0 python infer_fmri_obj.py ./MinD-3D/configs/fmri_shape.yaml \
 --unet_path model_weight \
 --save_name save_dir \
 --input_path ./dataset/fmri_shape/core_test_list.txt \
 --fmri_dir fmri_dir \
 --gt_image_dir gt_image_dir \
 --save_video --export_texmap
