"""
Custom utilities and modular step-by-step pipeline for BrepGen CAD generation.
"""
import os
import copy
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


# Import network architectures from original BrepGen codebase
from eispdiff.cad_gen.brepgen.network import (
    SurfPosNet,
    SurfZNet,
    EdgePosNet,
    EdgeZNet,
    AutoencoderKLFastDecode,
    AutoencoderKL1DFastDecode,
)

# Schedulers
from diffusers import DDPMScheduler, PNDMScheduler

# Import topology & geometry functions from original BrepGen utils.py
from eispdiff.cad_gen.brepgen.utils import (
    randn_tensor,
    compute_bbox_center_and_size,
    generate_random_string,
    detect_shared_vertex,
    detect_shared_edge,
    joint_optimize,
    construct_brep,
    plot_3d_bbox,
)

# Shared visualization utilities
from eispdiff.cad_gen.utils import solid_to_mesh, render_3d_mesh_jupyter
import trimesh
import trimesh.viewer
from IPython.display import display, HTML


# Furniture class label mapping (taken from sample.py)
text2int = {
    'uncond': 0,
    'bathtub': 1,
    'bed': 2,
    'bench': 3,
    'bookshelf': 4,
    'cabinet': 5,
    'chair': 6,
    'couch': 7,
    'lamp': 8,
    'sofa': 9,
    'table': 10,
}
int2text = {v: k for k, v in text2int.items()}


def init_schedulers():
    """
    Initialize PNDM and DDPM noise schedulers for BrepGen inference.
    (taken from sample.py)
    """
    pndm_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
    )
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=True,
        clip_sample_range=3,
    )
    return pndm_scheduler, ddpm_scheduler


def load_brepgen_models(weights_dict, use_cf=False, device=None):
    """
    Instantiate and load weights for the 4 LDMs and 2 VAEs.
    (taken from sample.py)
    
    Args:
        weights_dict (dict): Dictionary with paths for:
            - 'surfpos_weight'
            - 'surfz_weight'
            - 'edgepos_weight'
            - 'edgez_weight'
            - 'surfvae_weight'
            - 'edgevae_weight'
        use_cf (bool): Classifier-free conditioning flag (True for Furniture dataset)
        device: torch.device or None (defaults to cuda if available)
        
    Returns:
        models (dict): Dictionary of loaded PyTorch models in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Verify checkpoint paths
    missing = [k for k, p in weights_dict.items() if not os.path.exists(p)]
    if missing:
        missing_paths = {k: weights_dict[k] for k in missing}
        raise FileNotFoundError(
            f"The following checkpoint files were not found: {missing_paths}\\n"
            f"Please download the weights and set the correct checkpoint paths."
        )

    # Surface Position Model
    surfPos_model = SurfPosNet(use_cf)
    surfPos_model.load_state_dict(torch.load(weights_dict['surfpos_weight'], map_location=device))
    surfPos_model = surfPos_model.to(device).eval()

    # Surface Latent Geometry Model
    surfZ_model = SurfZNet(use_cf)
    surfZ_model.load_state_dict(torch.load(weights_dict['surfz_weight'], map_location=device))
    surfZ_model = surfZ_model.to(device).eval()

    # 3. Edge Position Model
    edgePos_model = EdgePosNet(use_cf)
    edgePos_model.load_state_dict(torch.load(weights_dict['edgepos_weight'], map_location=device))
    edgePos_model = edgePos_model.to(device).eval()

    # 4. Edge Latent Geometry & Vertex Model
    edgeZ_model = EdgeZNet(use_cf)
    edgeZ_model.load_state_dict(torch.load(weights_dict['edgez_weight'], map_location=device))
    edgeZ_model = edgeZ_model.to(device).eval()

    # 5. Surface VAE
    surf_vae = AutoencoderKLFastDecode(
        in_channels=3,
        out_channels=3,
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512,
    )
    surf_vae.load_state_dict(torch.load(weights_dict['surfvae_weight'], map_location=device), strict=False)
    surf_vae = surf_vae.to(device).eval()

    # 6. Edge VAE
    edge_vae = AutoencoderKL1DFastDecode(
        in_channels=3,
        out_channels=3,
        down_block_types=['DownBlock1D', 'DownBlock1D', 'DownBlock1D'],
        up_block_types=['UpBlock1D', 'UpBlock1D', 'UpBlock1D'],
        block_out_channels=[128, 256, 512],
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512,
    )
    edge_vae.load_state_dict(torch.load(weights_dict['edgevae_weight'], map_location=device), strict=False)
    edge_vae = edge_vae.to(device).eval()

    return {
        'surfPos_model': surfPos_model,
        'surfZ_model': surfZ_model,
        'edgePos_model': edgePos_model,
        'edgeZ_model': edgeZ_model,
        'surf_vae': surf_vae,
        'edge_vae': edge_vae,
        'device': device,
    }


def step1_generate_surface_positions(
    models,
    pndm_scheduler,
    ddpm_scheduler,
    batch_size=1,
    num_surfaces=30,
    use_cf=False,
    class_label=None,
    w=0.6,
    bbox_threshold=0.08,
    device=None,
    progress=True,
):
    """
    Step 1-1 & Step 1-2: Generate and deduplicate surface bounding boxes.
    (taken from sample.py)
    
    Returns:
        surfPos (torch.Tensor): Deduplicated surface bounding boxes [B, num_surfaces, 6]
        surfMask (torch.Tensor): Boolean mask where True indicates padded/invalid surfaces [B, num_surfaces]
        num_surfaces (int): Effective surface count after late-increase if unconditional
    """
    if device is None:
        device = models['device']
    surfPos_model = models['surfPos_model']

    # Class conditioning label setup
    if use_cf and class_label is not None:
        if isinstance(class_label, str):
            c_int = text2int[class_label]
        else:
            c_int = int(class_label)
        cl_tensor = torch.LongTensor([c_int] * batch_size + [text2int['uncond']] * batch_size).to(device).reshape(-1, 1)
    else:
        cl_tensor = None

    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.no_grad():
        with torch.autocast(device_type=autocast_device):
            # STEP 1-1: Sample initial Gaussian noise for surface bounding boxes
            surfPos = randn_tensor((batch_size, num_surfaces, 6), device=device)

            # Stage A: Fast PNDM sampling (first 158 steps of 200)
            pndm_scheduler.set_timesteps(200)
            timesteps_pndm = pndm_scheduler.timesteps[:158]
            iterator_pndm = tqdm(timesteps_pndm, desc="Step 1-1: Surface Pos (PNDM)") if progress else timesteps_pndm
            for t in iterator_pndm:
                timesteps = t.reshape(-1).to(device)
                if cl_tensor is not None:
                    _surfPos_ = surfPos.repeat(2, 1, 1)
                    pred = surfPos_model(_surfPos_, timesteps, cl_tensor)
                    pred = pred[:batch_size] * (1 + w) - pred[batch_size:] * w
                else:
                    pred = surfPos_model(surfPos, timesteps, cl_tensor)
                surfPos = pndm_scheduler.step(pred, t, surfPos).prev_sample

            # Late increase for unconditional datasets (ABC / DeepCAD)
            if not use_cf:
                surfPos = surfPos.repeat(1, 2, 1)
                num_surfaces *= 2

            # Stage B: Fine DDPM refinement (last 250 steps of 1000)
            ddpm_scheduler.set_timesteps(1000)
            timesteps_ddpm = ddpm_scheduler.timesteps[-250:]
            iterator_ddpm = tqdm(timesteps_ddpm, desc="Step 1-1: Surface Pos (DDPM)") if progress else timesteps_ddpm
            for t in iterator_ddpm:
                timesteps = t.reshape(-1).to(device)
                if cl_tensor is not None:
                    _surfPos_ = surfPos.repeat(2, 1, 1)
                    pred = surfPos_model(_surfPos_, timesteps, cl_tensor)
                    pred = pred[:batch_size] * (1 + w) - pred[batch_size:] * w
                else:
                    pred = surfPos_model(surfPos, timesteps, cl_tensor)
                surfPos = ddpm_scheduler.step(pred, t, surfPos).prev_sample

            # STEP 1-2: Remove duplicate surface bounding boxes
            surfPos_deduplicate = []
            surfMask_deduplicate = []
            for ii in range(batch_size):
                bboxes = np.round(surfPos[ii].unflatten(-1, torch.Size([2, 3])).detach().cpu().numpy(), 4)
                non_repeat = bboxes[:1]
                for bbox_idx, bbox in enumerate(bboxes):
                    diff = np.max(np.max(np.abs(non_repeat - bbox), -1), -1)
                    same = diff < bbox_threshold
                    bbox_rev = bbox[::-1]
                    diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev), -1), -1)
                    same_rev = diff_rev < bbox_threshold
                    if same.sum() >= 1 or same_rev.sum() >= 1:
                        continue
                    else:
                        non_repeat = np.concatenate([non_repeat, bbox[np.newaxis, :, :]], 0)
                bboxes = non_repeat.reshape(len(non_repeat), -1)

                surf_mask = torch.zeros((1, len(bboxes))) == 1
                bbox_padded = torch.concat([torch.FloatTensor(bboxes), torch.zeros(num_surfaces - len(bboxes), 6)])
                mask_padded = torch.concat([surf_mask, torch.zeros(1, num_surfaces - len(bboxes)) == 0], -1)
                surfPos_deduplicate.append(bbox_padded)
                surfMask_deduplicate.append(mask_padded)

            surfPos = torch.stack(surfPos_deduplicate).to(device)
            surfMask = torch.vstack(surfMask_deduplicate).to(device)

    return surfPos, surfMask, num_surfaces


def step1_generate_surface_latents(
    models,
    pndm_scheduler,
    surfPos,
    surfMask,
    batch_size=1,
    num_surfaces=30,
    use_cf=False,
    class_label=None,
    w=0.6,
    device=None,
    progress=True,
):
    """
    Step 1-3: Generate surface continuous geometry latents (48-D per surface)
    conditioned on predicted surface positions and mask.
    (taken from sample.py)
    
    Returns:
        surfZ (torch.Tensor): Surface latent geometry [B, num_surfaces, 48]
    """
    if device is None:
        device = models['device']
    surfZ_model = models['surfZ_model']

    if use_cf and class_label is not None:
        c_int = text2int[class_label] if isinstance(class_label, str) else int(class_label)
        cl_tensor = torch.LongTensor([c_int] * batch_size + [text2int['uncond']] * batch_size).to(device).reshape(-1, 1)
    else:
        cl_tensor = None

    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.no_grad():
        with torch.autocast(device_type=autocast_device):
            surfZ = randn_tensor((batch_size, num_surfaces, 48), device=device)
            pndm_scheduler.set_timesteps(200)
            iterator = tqdm(pndm_scheduler.timesteps, desc="Step 1-3: Surface Latent z (PNDM)") if progress else pndm_scheduler.timesteps
            for t in iterator:
                timesteps = t.reshape(-1).to(device)
                if cl_tensor is not None:
                    _surfZ_ = surfZ.repeat(2, 1, 1)
                    _surfPos_ = surfPos.repeat(2, 1, 1)
                    _surfMask_ = surfMask.repeat(2, 1)
                    pred = surfZ_model(_surfZ_, timesteps, _surfPos_, _surfMask_, cl_tensor)
                    pred = pred[:batch_size] * (1 + w) - pred[batch_size:] * w
                else:
                    pred = surfZ_model(surfZ, timesteps, surfPos, surfMask, cl_tensor)
                surfZ = pndm_scheduler.step(pred, t, surfZ).prev_sample

    return surfZ


def step2_generate_edge_positions(
    models,
    pndm_scheduler,
    ddpm_scheduler,
    surfPos,
    surfZ,
    surfMask,
    batch_size=1,
    num_surfaces=30,
    num_edges=30,
    use_cf=False,
    class_label=None,
    w=0.6,
    bbox_threshold=0.08,
    device=None,
    progress=True,
):
    """
    Step 2-1 & Step 2-2: Generate and deduplicate edge bounding boxes per face.
    (Original reference: sample.py lines 205-262)
    
    Returns:
        edgePos (torch.Tensor): Edge bounding boxes [B, num_surfaces, num_edges, 6]
        edgeM (torch.Tensor): Edge mask [B, num_surfaces, num_edges]
    """
    if device is None:
        device = models['device']
    edgePos_model = models['edgePos_model']

    if use_cf and class_label is not None:
        c_int = text2int[class_label] if isinstance(class_label, str) else int(class_label)
        cl_tensor = torch.LongTensor([c_int] * batch_size + [text2int['uncond']] * batch_size).to(device).reshape(-1, 1)
    else:
        cl_tensor = None

    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.no_grad():
        with torch.autocast(device_type=autocast_device):
            # Sample initial noise for edge positions
            edgePos = randn_tensor((batch_size, num_surfaces, num_edges, 6), device=device)

            # Stage A: Fast PNDM sampling
            pndm_scheduler.set_timesteps(200)
            timesteps_pndm = pndm_scheduler.timesteps[:158]
            iterator_pndm = tqdm(timesteps_pndm, desc="Step 2-1: Edge Pos (PNDM)") if progress else timesteps_pndm
            for t in iterator_pndm:
                timesteps = t.reshape(-1).to(device)
                if cl_tensor is not None:
                    _surfZ_ = surfZ.repeat(2, 1, 1)
                    _surfPos_ = surfPos.repeat(2, 1, 1)
                    _surfMask_ = surfMask.repeat(2, 1)
                    _edgePos_ = edgePos.repeat(2, 1, 1, 1)
                    noise_pred = edgePos_model(_edgePos_, timesteps, _surfPos_, _surfZ_, _surfMask_, cl_tensor)
                    noise_pred = noise_pred[:batch_size] * (1 + w) - noise_pred[batch_size:] * w
                else:
                    noise_pred = edgePos_model(edgePos, timesteps, surfPos, surfZ, surfMask, cl_tensor)
                edgePos = pndm_scheduler.step(noise_pred, t, edgePos).prev_sample

            # Stage B: Fine DDPM refinement
            ddpm_scheduler.set_timesteps(1000)
            timesteps_ddpm = ddpm_scheduler.timesteps[-250:]
            iterator_ddpm = tqdm(timesteps_ddpm, desc="Step 2-1: Edge Pos (DDPM)") if progress else timesteps_ddpm
            for t in iterator_ddpm:
                timesteps = t.reshape(-1).to(device)
                if cl_tensor is not None:
                    _surfZ_ = surfZ.repeat(2, 1, 1)
                    _surfPos_ = surfPos.repeat(2, 1, 1)
                    _surfMask_ = surfMask.repeat(2, 1)
                    _edgePos_ = edgePos.repeat(2, 1, 1, 1)
                    noise_pred = edgePos_model(_edgePos_, timesteps, _surfPos_, _surfZ_, _surfMask_, cl_tensor)
                    noise_pred = noise_pred[:batch_size] * (1 + w) - noise_pred[batch_size:] * w
                else:
                    noise_pred = edgePos_model(edgePos, timesteps, surfPos, surfZ, surfMask, cl_tensor)
                edgePos = ddpm_scheduler.step(noise_pred, t, edgePos).prev_sample

            # STEP 2-2: Remove duplicate edges per face
            edgeM = surfMask.unsqueeze(-1).repeat(1, 1, num_edges)
            for ii in range(batch_size):
                edge_bboxs = edgePos[ii][~surfMask[ii]].detach().cpu().numpy()
                for surf_idx, bboxes in enumerate(edge_bboxs):
                    bboxes = bboxes.reshape(len(bboxes), 2, 3)
                    valid_bbox = bboxes[0:1]
                    for bbox_idx, bbox in enumerate(bboxes):
                        diff = np.max(np.max(np.abs(valid_bbox - bbox), -1), -1)
                        bbox_rev = bbox[::-1]
                        diff_rev = np.max(np.max(np.abs(valid_bbox - bbox_rev), -1), -1)
                        same = diff < bbox_threshold
                        same_rev = diff_rev < bbox_threshold
                        if same.sum() >= 1 or same_rev.sum() >= 1:
                            edgeM[ii, surf_idx, bbox_idx] = True
                            continue
                        else:
                            valid_bbox = np.concatenate([valid_bbox, bbox[np.newaxis, :, :]], 0)
                    edgeM[ii, surf_idx, 0] = False

    return edgePos, edgeM


def step2_generate_edge_latents(
    models,
    pndm_scheduler,
    surfPos,
    surfZ,
    surfMask,
    edgePos,
    edgeM,
    batch_size=1,
    num_surfaces=30,
    num_edges=30,
    use_cf=False,
    class_label=None,
    w=0.6,
    device=None,
    progress=True,
):
    """
    Step 2-3 & VAE Decoding: Generate edge continuous latents and endpoint vertex offsets (18-D),
    then decode surfaces and edges via their respective VAE decoders.
    (taken from sample.py)
    
    Returns:
        dict containing:
            - 'edge_z': Decoded edge latents [B, S, E, 12]
            - 'edgeV': Predicted endpoint vertex offsets [B, S, E, 6]
            - 'surf_ncs': Decoded surface point clouds (32x32x3) [B, S, 32, 32, 3]
            - 'edge_ncs': Decoded 1D edge curve point clouds (32x3) [B, S, E, 32, 3]
            - 'edge_pos': Scaled edge bounding box coordinates
            - 'surf_pos_scaled': Scaled surface bounding box coordinates
            - 'edge_mask': Numpy edge validity mask
    """
    if device is None:
        device = models['device']
    edgeZ_model = models['edgeZ_model']
    surf_vae = models['surf_vae']
    edge_vae = models['edge_vae']

    if use_cf and class_label is not None:
        c_int = text2int[class_label] if isinstance(class_label, str) else int(class_label)
        cl_tensor = torch.LongTensor([c_int] * batch_size + [text2int['uncond']] * batch_size).to(device).reshape(-1, 1)
    else:
        cl_tensor = None

    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'

    with torch.no_grad():
        with torch.autocast(device_type=autocast_device):
            edgeZV = randn_tensor((batch_size, num_surfaces, num_edges, 18), device=device)
            pndm_scheduler.set_timesteps(200)
            iterator = tqdm(pndm_scheduler.timesteps, desc="Step 2-3: Edge Latent z & Vertices (PNDM)") if progress else pndm_scheduler.timesteps
            for t in iterator:
                timesteps = t.reshape(-1).to(device)
                if cl_tensor is not None:
                    _surfZ_ = surfZ.repeat(2, 1, 1)
                    _surfPos_ = surfPos.repeat(2, 1, 1)
                    _edgePos_ = edgePos.repeat(2, 1, 1, 1)
                    _edgeM_ = edgeM.repeat(2, 1, 1)
                    _edgeZV_ = edgeZV.repeat(2, 1, 1, 1)
                    noise_pred = edgeZ_model(_edgeZV_, timesteps, _edgePos_, _surfPos_, _surfZ_, _edgeM_, cl_tensor)
                    noise_pred = noise_pred[:batch_size] * (1 + w) - noise_pred[batch_size:] * w
                else:
                    noise_pred = edgeZ_model(edgeZV, timesteps, edgePos, surfPos, surfZ, edgeM, cl_tensor)
                edgeZV = pndm_scheduler.step(noise_pred, t, edgeZV).prev_sample

            edgeZV[edgeM] = 0
            edge_z = edgeZV[:, :, :, :12]
            edgeV = edgeZV[:, :, :, 12:].detach().cpu().numpy()

            # Decode the surfaces via Surface VAE (latent: 16x3 -> 4x4 spatial -> 32x32 surface)
            surf_ncs = surf_vae(
                surfZ.unflatten(-1, torch.Size([16, 3])).flatten(0, 1).permute(0, 2, 1).unflatten(-1, torch.Size([4, 4]))
            )
            surf_ncs = surf_ncs.permute(0, 2, 3, 1).unflatten(0, torch.Size([batch_size, num_surfaces])).detach().cpu().numpy()

            # Decode the edges via Edge 1D VAE (latent: 4x3 -> 32x3 1D curve)
            edge_ncs = edge_vae(edge_z.unflatten(-1, torch.Size([4, 3])).reshape(-1, 4, 3).permute(0, 2, 1))
            edge_ncs = edge_ncs.permute(0, 2, 1).reshape(batch_size, num_surfaces, num_edges, 32, 3).detach().cpu().numpy()

            edge_mask = edgeM.detach().cpu().numpy()
            edge_pos = edgePos.detach().cpu().numpy() / 3.0
            surf_pos_scaled = surfPos.detach().cpu().numpy() / 3.0

    return {
        'edge_z': edge_z,
        'edgeV': edgeV,
        'surf_ncs': surf_ncs,
        'edge_ncs': edge_ncs,
        'edge_pos': edge_pos,
        'surf_pos_scaled': surf_pos_scaled,
        'edge_mask': edge_mask,
    }


def step3_reconstruct_cad_solid(
    sample_idx,
    surfPos,
    surfMask,
    surfZ,
    edge_data,
    surf_vae,
    edge_vae,
    z_threshold=0.2,
    use_local_cd=True,
    device=None,
):
    """
    Step 3: Topology Recovery, Joint Optimization, and B-rep Construction for a single sample.
    (taken from sample.py)
    
    Steps:
        3-1: Detect shared vertices
        3-2: Detect shared edges
        3-3: Joint optimization of parametric surfaces and curves
        3-4: Build watertight B-rep CAD Solid via OpenCASCADE
        
    Returns:
        solid (OCC TopoDS_Shape / TopoDS_Solid or None): Reconstructed CAD B-rep body
        debug_info (dict): Diagnostic geometry components (surfaces, curves, graph relations)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Unpack sample tensors
    surfMask_cad = surfMask[sample_idx].detach().cpu().numpy()
    edge_mask_cad = edge_data['edge_mask'][sample_idx][~surfMask_cad]
    edge_pos_cad = edge_data['edge_pos'][sample_idx][~surfMask_cad]
    edge_ncs_cad = edge_data['edge_ncs'][sample_idx][~surfMask_cad]
    edgeV_cad = edge_data['edgeV'][sample_idx][~surfMask_cad]
    edge_z_cad = edge_data['edge_z'][sample_idx][~surfMask[sample_idx]].detach().cpu().numpy()[~edge_mask_cad]
    surf_z_cad = surfZ[sample_idx][~surfMask[sample_idx]].detach().cpu().numpy()
    surf_pos_cad = edge_data['surf_pos_scaled'][sample_idx][~surfMask_cad]

    # Retrieve vertices based on edge start/end coordinates in world space
    edgeV_bbox = []
    for bbox, ncs, mask in zip(edge_pos_cad, edge_ncs_cad, edge_mask_cad):
        epos = bbox[~mask]
        edge = ncs[~mask]
        bbox_startends = []
        for bb, ee in zip(epos, edge):
            bcenter, bsize = compute_bbox_center_and_size(bb[0:3], bb[3:])
            wcs = ee * (bsize / 2) + bcenter
            bbox_start_end = wcs[[0, -1]]
            bbox_start_end = bbox_start_end.reshape(2, 3)
            bbox_startends.append(bbox_start_end.reshape(1, 2, 3))
        bbox_startends = np.vstack(bbox_startends)
        edgeV_bbox.append(bbox_startends)

    # 3-1: Detect shared vertices
    try:
        unique_vertices, new_vertex_dict = detect_shared_vertex(edgeV_cad, edge_mask_cad, edgeV_bbox)
    except Exception as e:
        print(f"Sample #{sample_idx}: Vertex detection failed ({e})")
        return None, {}

    # 3-2: Detect shared edges
    try:
        unique_faces, unique_edges, FaceEdgeAdj, EdgeVertexAdj = detect_shared_edge(
            unique_vertices, new_vertex_dict, edge_z_cad, surf_z_cad, z_threshold, edge_mask_cad
        )
    except Exception as e:
        print(f"Sample #{sample_idx}: Edge detection failed ({e})")
        return None, {}

    # Decode unique faces / edges through VAEs
    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
    with torch.no_grad():
        with torch.autocast(device_type=autocast_device):
            surf_ncs_cad = surf_vae(
                torch.FloatTensor(unique_faces).to(device).unflatten(-1, torch.Size([16, 3])).permute(0, 2, 1).unflatten(-1, torch.Size([4, 4]))
            )
            surf_ncs_cad = surf_ncs_cad.permute(0, 2, 3, 1).detach().cpu().numpy()
            edge_ncs_cad = edge_vae(
                torch.FloatTensor(unique_edges).to(device).unflatten(-1, torch.Size([4, 3])).permute(0, 2, 1)
            )
            edge_ncs_cad = edge_ncs_cad.permute(0, 2, 1).detach().cpu().numpy()

    # 3-3: Joint Optimization
    num_edge = len(edge_ncs_cad)
    num_surf = len(surf_ncs_cad)
    surf_wcs, edge_wcs = joint_optimize(
        surf_ncs_cad, edge_ncs_cad, surf_pos_cad, unique_vertices,
        EdgeVertexAdj, FaceEdgeAdj, num_edge, num_surf, use_local_cd=use_local_cd
    )

    # 3-4: Build B-rep Solid using OpenCASCADE
    try:
        solid = construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj)
    except Exception as e:
        print(f"Sample #{sample_idx}: B-rep rebuild failed ({e})")
        return None, {
            'surf_wcs': surf_wcs,
            'edge_wcs': edge_wcs,
            'FaceEdgeAdj': FaceEdgeAdj,
            'EdgeVertexAdj': EdgeVertexAdj,
            'unique_vertices': unique_vertices,
        }

    debug_info = {
        'surf_wcs': surf_wcs,
        'edge_wcs': edge_wcs,
        'FaceEdgeAdj': FaceEdgeAdj,
        'EdgeVertexAdj': EdgeVertexAdj,
        'unique_vertices': unique_vertices,
    }
    return solid, debug_info


def plot_3d_bounding_boxes(surf_pos, title="Predicted Surface Bounding Boxes", ax=None, color='royalblue', alpha=0.3):
    """
    Visualize 3D bounding boxes for predicted surface patches.
    """
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    bboxes = surf_pos.reshape(-1, 2, 3)
    for bb in bboxes:
        min_pt, max_pt = bb[0], bb[1]
        if np.all(min_pt == 0) and np.all(max_pt == 0):
            continue
        plot_3d_bbox(ax, min_pt, max_pt, color=color)

    ax.set_title(title, fontsize=12, pad=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return fig, ax


def plot_surface_patches_pointcloud(
    surf_wcs,
    edge_wcs=None,
    surf_pos=None,
    title="Decoded Surface Patches & Edge Curves",
    max_patches=15,
    interactive=True,
    height=480,
    point_size=4.0,
):
    """
    Visualize decoded parametric surface point grids and trimming edge curves.
    Supports both an interactive 3D WebGL viewer (Three.js with TrackballControls)
    and a static Matplotlib 3D scatter plot fallback.

    Args:
        surf_wcs: List or array of surface point grids (each shaped (32, 32, 3) or (N, 3))
        edge_wcs: Optional list or array of edge curve points (each shaped (32, 3) or (N, 3))
        surf_pos: Optional bounding boxes to automatically place canonical patches into World Coordinates (WCS)
        title: Header title string
        max_patches: Maximum number of surface patches to display
        interactive: If True, renders interactive Three.js viewer; if False, uses Matplotlib 3D
        height: Height in pixels of the interactive viewer iframe
        point_size: Size in pixels of point cloud dots in the 3D viewer

    Returns:
        If interactive: (scene, embedded_html)
        If not interactive: (fig, ax)
    """
    # If bounding box coordinates (surf_pos) are provided, transform canonical patches (NCS) into world space (WCS)
    if surf_pos is not None:
        transformed_patches = []
        for p, bb in zip(surf_wcs, surf_pos):
            c, s = compute_bbox_center_and_size(bb[0:3], bb[3:])
            transformed_patches.append(np.asarray(p) * (s / 2.0) + c)
        surf_wcs = transformed_patches

    if interactive:
        try:
            cmap = plt.colormaps['tab20']
        except (AttributeError, KeyError):
            cmap = plt.cm.get_cmap('tab20')

        geometries = []
        num_patches = min(len(surf_wcs), max_patches)
        total_pts = 0

        for idx in range(num_patches):
            patch = surf_wcs[idx]
            pts = np.asarray(patch).reshape(-1, 3)
            if len(pts) == 0:
                continue
            total_pts += len(pts)
            # Sample color from tab20 colormap
            rgba = (np.array(cmap(idx % 20)) * 255).astype(np.uint8)
            colors = np.tile(rgba, (len(pts), 1))
            pc = trimesh.points.PointCloud(pts, colors=colors)
            geometries.append(pc)

        num_edges = 0
        if edge_wcs is not None:
            for c_idx, curve in enumerate(edge_wcs):
                c_pts = np.asarray(curve).reshape(-1, 3)
                if len(c_pts) >= 2:
                    num_edges += 1
                    path = trimesh.load_path(c_pts)
                    path.colors = [[15, 15, 15, 255]]
                    geometries.append(path)

        if not geometries:
            print(f"Warning: No valid geometry to display for '{title}'")
            return None, None

        scene = trimesh.Scene(geometries)

        # Generate Three.js HTML with custom point size and trackball controls
        html_base = trimesh.viewer.notebook.scene_to_html(scene, escape_quotes=False)
        # Inject point size and disable attenuation for sharp, clearly visible points
        js_point_config = (
            f"gltf.scene.traverse(function(c){{"
            f"if(c.isPoints){{c.material.size={point_size};c.material.sizeAttenuation=false;}}"
            f"}});"
            f"scene.add(gltf.scene);"
        )
        html_enhanced = html_base.replace("scene.add(gltf.scene);", js_point_config)
        srcdoc = html_enhanced.replace('"', '&quot;')

        edge_info = f", {num_edges} edge curves" if edge_wcs is not None else ""
        header = HTML(
            f'<h4 style="margin: 8px 0 4px 0; font-family: sans-serif; color: #ffffff;">{title} '
            f'<span style="font-size: 0.9em; font-weight: normal; color: #a0aec0;">'
            f'({num_patches} patches, {total_pts} surface points{edge_info})</span></h4>'
        )
        embedded = HTML(
            f'<div><iframe srcdoc="{srcdoc}" width="100%" height="{height}px" '
            f'style="border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.25);"></iframe></div>'
        )

        display(header)
        display(embedded)
        return scene, embedded

    # Matplotlib 3D fallback
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    try:
        cmap = plt.colormaps['tab20']
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('tab20')

    num_patches = min(len(surf_wcs), max_patches)
    for idx in range(num_patches):
        pts = np.asarray(surf_wcs[idx]).reshape(-1, 3)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, color=cmap(idx % 20), alpha=0.6, label=f"Face {idx}")

    if edge_wcs is not None:
        for c_idx, curve in enumerate(edge_wcs):
            c_pts = np.asarray(curve).reshape(-1, 3)
            ax.plot(c_pts[:, 0], c_pts[:, 1], c_pts[:, 2], color='black', linewidth=1.5)

    ax.set_title(title, fontsize=12, pad=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    return fig, ax



def generate_brep_cad_model(
    models,
    schedulers,
    mode="deepcad",
    class_label=None,
    seed=None,
    num_surfaces=None,
    num_edges=None,
    use_cf=None,
    w=0.6,
    bbox_threshold=0.08,
    z_threshold=0.2,
    device=None,
    interactive=True,
):
    """
    Unified end-to-end generator for BrepGen CAD models.
    Supports mode='deepcad', mode='abc', and mode='furniture' (with class conditioning).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if device is None:
        device = models['device']

    pndm_scheduler, ddpm_scheduler = schedulers

    # Default parameters based on dataset
    if mode == "deepcad":
        num_surfaces = num_surfaces or 30
        num_edges = num_edges or 30
        use_cf = False if use_cf is None else use_cf
    elif mode == "abc":
        num_surfaces = num_surfaces or 50
        num_edges = num_edges or 40
        use_cf = False if use_cf is None else use_cf
    elif mode == "furniture":
        num_surfaces = num_surfaces or 60
        num_edges = num_edges or 40
        use_cf = True if use_cf is None else use_cf
        class_label = class_label or "chair"

    # Step 1: Surface Positions & Latents
    surfPos, surfMask, cur_num_surfaces = step1_generate_surface_positions(
        models, 
        pndm_scheduler, 
        ddpm_scheduler, 
        batch_size=1,
        num_surfaces=num_surfaces, 
        use_cf=use_cf, 
        class_label=class_label,
        w=w, 
        bbox_threshold=bbox_threshold, 
        device=device
    )
    surfZ = step1_generate_surface_latents(
        models, 
        pndm_scheduler, 
        surfPos, 
        surfMask, 
        batch_size=1,
        num_surfaces=cur_num_surfaces, 
        use_cf=use_cf, 
        class_label=class_label,
        w=w, 
        device=device
    )

    # Step 2: Edge Positions & Latents
    edgePos, edgeM = step2_generate_edge_positions(
        models, 
        pndm_scheduler, 
        ddpm_scheduler, 
        surfPos, 
        surfZ, 
        surfMask,
        batch_size=1, 
        num_surfaces=cur_num_surfaces, 
        num_edges=num_edges,
        use_cf=use_cf, 
        class_label=class_label, 
        w=w,
        bbox_threshold=bbox_threshold, 
        device=device
    )
    edge_data = step2_generate_edge_latents(
        models, 
        pndm_scheduler, 
        surfPos, 
        surfZ, 
        surfMask, 
        edgePos, 
        edgeM,
        batch_size=1, 
        num_surfaces=cur_num_surfaces, 
        num_edges=num_edges,
        use_cf=use_cf, 
        class_label=class_label, 
        w=w, 
        device=device
    )

    # Step 3: Reconstruction
    solid, debug_info = step3_reconstruct_cad_solid(
        0, 
        surfPos, 
        surfMask, 
        surfZ, 
        edge_data,
        models['surf_vae'], 
        models['edge_vae'],
        z_threshold=z_threshold, 
        device=device
    )

    mesh = None
    if solid is not None:
        mesh, _ = solid_to_mesh(solid)
        title = f"Generated {mode.upper()} CAD Solid"
        if class_label:
            title += f" ({class_label})"
        if seed is not None:
            title += f" [Seed: {seed}]"
        render_3d_mesh_jupyter(mesh, title=title, interactive=interactive)

    return solid, mesh, debug_info

