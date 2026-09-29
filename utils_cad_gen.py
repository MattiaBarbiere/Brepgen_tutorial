"""
Shared CAD generation and visualization utilities for DeepCAD and BrepGen.
"""
import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import trimesh.viewer
from IPython.display import display, HTML
import build123d as bd

# Import OpenCASCADE / OCP STL & STEP writers
try:
    from temp.OCC.Extend.DataExchange import write_stl_file, write_step_file
except ImportError:
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    from OCP.STEPControl import STEPControl_Writer, STEPControl_StepModelType

    def write_stl_file(shape, filename, *args, **kwargs):
        lin_def = kwargs.get("linear_deflection", kwargs.get("lin_deflection", 0.005))
        if len(args) > 0:
            lin_def = args[0]
        BRepMesh_IncrementalMesh(shape, float(lin_def))
        writer = StlAPI_Writer()
        writer.Write(shape, filename)

    def write_step_file(shape, filename):
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_StepModelType.STEPControl_AsIs)
        writer.Write(filename)


def solid_to_mesh(raw_shape, tolerance=0.005, stl_file_name=None, step_file_name=None, folder_path="cad_files"):
    """
    Convert OpenCASCADE TopoDS_Shape into a trimesh.Trimesh and build123d.Shape.
    
    Args:
        raw_shape: TopoDS_Shape from OpenCASCADE (DeepCAD or BrepGen)
        tolerance: Linear deflection for STL tessellation
        stl_file_name: Optional path to save STL file
        step_file_name: Optional path to save STEP file
        folder_path: Directory where CAD export files are stored
        
    Returns:
        mesh (trimesh.Trimesh): Tessellated triangular mesh
        bd_shape (build123d.Shape): build123d wrapper for the CAD solid
    """
    if raw_shape is None:
        return None, None

    # Convert to build123d Shape wrapper
    try:
        bd_shape = bd.Shape(raw_shape)
    except Exception:
        bd_shape = None

    os.makedirs(folder_path, exist_ok=True)
    temp_used = False
    if stl_file_name is None:
        file_name = f"temp_{uuid.uuid4().hex[:8]}.stl"
        temp_used = True
    else:
        file_name = stl_file_name
    file_loc = os.path.join(folder_path, file_name)

    # Tessellate and export STL
    write_stl_file(raw_shape, file_loc, lin_deflection=tolerance)

    # Load tessellated mesh into trimesh
    with open(file_loc, 'rb') as f:
        mesh = trimesh.load(f, file_type='stl')

    # Remove temporary STL file if generated
    if temp_used and os.path.exists(file_loc):
        os.remove(file_loc)

    # Remove export directory if empty
    if os.path.exists(folder_path) and len(os.listdir(folder_path)) == 0:
        os.rmdir(folder_path)

    # Save STEP file if requested
    if step_file_name is not None:
        os.makedirs(folder_path, exist_ok=True)
        step_file_loc = os.path.join(folder_path, step_file_name)
        if bd_shape is not None:
            bd.export_step(bd_shape, step_file_loc)
        else:
            write_step_file(raw_shape, step_file_loc)

    return mesh, bd_shape


def render_3d_mesh_jupyter(mesh, title="3D CAD Solid", interactive=True, height=450):
    """
    Render 3D surface mesh inside Jupyter.
    Uses three.js interactive viewer by default, with matplotlib fallback.
    
    Args:
        mesh (trimesh.Trimesh): 3D triangle mesh
        title (str): Header title
        interactive (bool): Whether to use three.js interactive viewer or matplotlib
        height (int): Height in pixels of interactive viewer
    """
    if mesh is None or len(mesh.vertices) == 0:
        print(f"Warning: Cannot render empty mesh for '{title}'")
        return

    if interactive:
        mesh_display = mesh.copy()
        # Default DeepCAD royal blue color
        mesh_display.visual.face_colors = [74, 144, 226, 255]
        scene = trimesh.Scene(mesh_display)
        
        # Title header with geometry stats
        header = HTML(
            f'<h4 style="margin: 8px 0 4px 0; font-family: sans-serif; color:#ffffff;">{title} '
            f'<span style="font-size: 1em; color: #ffffff; font-weight: normal;">'
            f'({len(mesh.vertices)} vertices, {len(mesh.faces)} faces, watertight: {mesh.is_watertight})</span></h4>'
        )
        viewer_html = trimesh.viewer.notebook.scene_to_notebook(scene, height=height)
        display(header)
        display(viewer_html)
        return

    # Fallback to Matplotlib 3D
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    verts = mesh.vertices
    faces = mesh.faces
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], verts[:, 2],
        triangles=faces,
        color='#4a90e2', edgecolor='#1c3b70', linewidth=0.2, alpha=0.9
    )
    ax.view_init(elev=25, azim=45)
    ax.set_title(title, fontsize=13, pad=15, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = np.array([
        verts[:, 0].max() - verts[:, 0].min(),
        verts[:, 1].max() - verts[:, 1].min(),
        verts[:, 2].max() - verts[:, 2].min()
    ]).max() / 2.0
    mid_x = (verts[:, 0].max() + verts[:, 0].min()) * 0.5
    mid_y = (verts[:, 1].max() + verts[:, 1].min()) * 0.5
    mid_z = (verts[:, 2].max() + verts[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.tight_layout()
    plt.show()
