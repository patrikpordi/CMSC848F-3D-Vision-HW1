# Iporting libraries
import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj
import pytorch3d
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
import mcubes
import pickle


# Defining functions for rendering
def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            
        else:
            device = torch.device("cpu")
            
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

# Defining unprojected depth image
def unproject_depth_image(image, mask, depth, camera):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): A square image to unproject (S, S, 3).
        mask (torch.Tensor): A binary mask for the image (S, S).
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.
    
    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    device = camera.device
    assert image.shape[0] == image.shape[1], "Image must be square."
    image_shape = image.shape[0]
    ndc_pixel_coordinates = torch.linspace(1, -1, image_shape)
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates)
    xy_depth = torch.dstack([X, Y, depth])
    points = camera.unproject_points(
        xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    )
    points = points[mask > 0.5]
    rgb = image[mask > 0.5]
    rgb = rgb.to(device)

    # For some reason, the Pytorch3D compositor does not apply a background color
    # unless the pointcloud is RGBA.
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb

# Definining pointcloud renderer
def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


class MyCowRenderer:
    """
    Renders a cow meshes using Pytorch3D.

    Args:
        path (str): The path to the cow mesh.
        image_size (int): The rendered image size.
        color (list): The color of the rendered mesh.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.

    Parameters:
        renderer (MeshRenderer): The Pytorch3D mesh renderer.
        lights (PointLights): The Pytorch3D lights object.
        mesh (Meshes): The Pytorch3D mesh object.
        textured_mesh (Meshes): The Pytorch3D mesh object with a texture.
        tetra_mesh (Meshes): The Pytorch3D tetrahedron mesh object.
        cube_mesh (Meshes): The Pytorch3D cube mesh object.
        dolly_mesh (Meshes): The Pytorch3D mesh object for the dolly.
        rotate_mesh (Meshes): The Pytorch3D mesh object for the rotating cow.
    """
    
    def __init__(self, path="data/cow.obj",image_size=512, 
                 color=[0.7, 0.7, 1],device=None):
        # Initialize the renderer with the provided or default parameters
        self.path = path
        self.image_size = image_size
        self.color = color
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print("Using CUDA")
            else:
                self.device = torch.device("cpu")
                print("Using CPU")
        # Initialize the mesh renderer
        self.renderer = self.get_mesh_renderer(image_size=self.image_size,device=self.device)
        self.lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=self.device)
        self.cameras = None
        self.mesh = None
        self.textured_mesh=None
        self.tetra_mesh=None
        self.cube_mesh=None

        # Load mesh and generate other meshes
        self.load_mesh()
        self.generate_tetre()
        self.generate_cube()
        self.dolly_mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_on_plane.obj"])
        self.rotate_mesh=pytorch3d.io.load_objs_as_meshes(["data/cow_with_axis.obj"])      
        
        

        

    def get_mesh_renderer(self, image_size=512, lights=None, device=None):
        """
        Returns a Pytorch3D Mesh Renderer.

        Args:
            image_size (int): The rendered image size.
            lights: A default Pytorch3D lights object.
            device (torch.device): The torch device to use (CPU or GPU). If not specified,
                will automatically use GPU if available, otherwise CPU.
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                
            else:
                device = torch.device("cpu")
                
        raster_settings = RasterizationSettings(
            image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=HardPhongShader(device=device, lights=lights),
        )
        return renderer
    
    def load_mesh(self):
        # Function to load and prepare the mesh
        vertices, faces, _ = load_obj(self.path)
        faces = faces.verts_idx
        vertices = vertices.unsqueeze(0)
        faces = faces.unsqueeze(0)
        textures = torch.ones_like(vertices)
        textures = textures * torch.tensor(self.color)
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures)
        )
        self.mesh = mesh

        ## Color the mesh
        color_1 = [0.2, 0.8, 0.2]  # Light Green
        color_2 = [0.0, 0.0, 1]  # Blue

        
        z_coordinates = vertices[:,:, 2]

        # Find the largest and smallest z-coordinates
        largest_z = torch.max(z_coordinates)
        smallest_z = torch.min(z_coordinates)

        alpha = (z_coordinates - smallest_z) / (largest_z - smallest_z)
        textures = torch.ones_like(vertices)  # (1, N_v, 3)

        color_11=textures * torch.tensor(color_1)
        color_22=textures * torch.tensor(color_2)
        
        alpha=alpha.unsqueeze(2).expand(-1, -1, 3)

        colorr = alpha * color_22 + (1 - alpha) * color_11
        
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(colorr),
        )
        self.textured_mesh = mesh
    
    def generate_tetre(self):
        # Define vertices and faces for a tetrahedron
        vertices = torch.tensor([
            [0.0, 0.0, 1.0],  # Vertex 0 (top)
            [1.0, 0.0, -1.0],  # Vertex 1 (bottom front)
            [-1.0, 0.0, -1.0],  # Vertex 2 (bottom left)
            [0.0, 1.0, 0.0]  # Vertex 3 (bottom right)
        ], dtype=torch.float32)

        faces = torch.tensor([
            [0, 1, 2],  # Face 0
            [0, 1, 3],  # Face 1
            [0, 2, 3],  # Face 2
            [1, 2, 3]   # Face 3
        ], dtype=torch.int64)
        # Define a single-color texture (e.g., blue)
        
        vertices = vertices.unsqueeze(0)
        faces = faces.unsqueeze(0)
        textures = torch.ones_like(vertices)
        textures = textures * torch.tensor(self.color)
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures)
        )
        self.tetra_mesh = mesh

    def generate_cube(self):
        # Define vertices and faces for a tetrahedron
        vertices = torch.tensor([
        [-0.5, -0.5, -0.5],  # Vertex 0
        [-0.5, -0.5, 0.5],   # Vertex 1
        [-0.5, 0.5, -0.5],   # Vertex 2
        [-0.5, 0.5, 0.5],    # Vertex 3
        [0.5, -0.5, -0.5],   # Vertex 4
        [0.5, -0.5, 0.5],    # Vertex 5
        [0.5, 0.5, -0.5],    # Vertex 6
        [0.5, 0.5, 0.5]      # Vertex 7
        ], dtype=torch.float32)

        faces = torch.tensor([
            [0, 1, 3],  # Face 0
            [0, 3, 2],  # Face 1
            [1, 5, 7],  # Face 2
            [1, 7, 3],  # Face 3
            [5, 4, 6],  # Face 4
            [5, 6, 7],  # Face 5
            [4, 0, 2],  # Face 6
            [4, 2, 6],  # Face 7
            [2, 3, 7],  # Face 8
            [2, 7, 6],  # Face 9
            [0, 1, 5],  # Face 10
            [0, 5, 4]   # Face 11
        ], dtype=torch.int64)
        # Define a single-color texture (e.g., blue)
        
        vertices = vertices.unsqueeze(0)
        faces = faces.unsqueeze(0)
        textures = torch.ones_like(vertices)
        textures = textures * torch.tensor(self.color)
        mesh = pytorch3d.structures.Meshes(
            verts=vertices,
            faces=faces,
            textures=pytorch3d.renderer.TexturesVertex(textures)
        )
        self.cube_mesh = mesh

    def generate_mesh(self,types="cow"):
        # Function to generate and save a mesh image
        if types=="cow":
            mesh=self.mesh.to(self.device)
            output_file="results/my_cow_mesh.png"
        elif types=="colored":
            mesh=self.textured_mesh.to(self.device)
            output_file="results/textured_mesh.png"
        elif types=="tetra":
            mesh=self.tetra_mesh.to(self.device)
            output_file="results/my_tetra_mesh.png"
        elif types=="cube":
            mesh=self.cube_mesh.to(self.device)
            output_file="results/my_cube_mesh.png"
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0),
        T=torch.tensor([[0, 0, 3]]),
        fov=60,device=self.device)
        rend = self.renderer(mesh, device=self.device, cameras=cameras, lights=self.lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        plt.imsave(output_file,rend)
        
    def generate_turntable_views(self, num_views=36,types="cow"):
        # Generate evenly spaced azimuth angles
        azimuths = torch.linspace(0, 360, num_views)
        if types=="cow":
            mesh=self.mesh.to(self.device)
            output_file="results/my_cow_turntable.gif"
        elif types=="dolly":
            mesh  = self.dolly_mesh.to(self.device)
            output_file="results/my_dolly_turntable.gif"
        elif types=="colored":
            mesh=self.textured_mesh.to(self.device)
            output_file="results/colored_turntable.gif"
        elif types=="tetra":
            mesh=self.tetra_mesh.to(self.device)
            output_file="results/my_tetra_turntable.gif"
        elif types=="cube":
            mesh=self.cube_mesh.to(self.device)
            output_file="results/my_cube_turntable.gif"
        
        # Fixed elevation and distance
        elevations = torch.tensor([30.0] * num_views)  # Fixed elevation
        distances = torch.tensor([3.0] * num_views)    # Fixed distance
        
        # Generate views using look_at_view_transform
        R, T = pytorch3d.renderer.look_at_view_transform(distances, elevations, azimuths)
        # print(R.unsqueeze(0))
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        fov=60,
        device=self.device)
        #Render images for each view
        images=[]
        for i in cameras:
            img=(self.renderer(mesh,device=self.device, cameras=i, lights=self.lights))
            img=img.cpu().numpy()[0, ..., :3]
            images.append(img)
        
        images_pil = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
        imageio.mimsave(output_file, images_pil, duration=30, loop=0)
    
    def generate_dolly(self,image_size=256,
    num_frames=20,
    duration=3,
    device=None,
    output_file="results/my_dolly.gif",
    ):
    # Function to generate and save a dolly gif
        if device is None:
            device = torch.device("cuda:0")

        mesh  = self.dolly_mesh.to(device)
        
        renderer = get_mesh_renderer(image_size=image_size, device=device)
        lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

        fovs = torch.linspace(5, 120, num_frames)

        renders = []
        for fov in tqdm(fovs):
            distance = 5/(2*np.tan(np.radians(fov/2)))  # TODO: change this.
            # distance = 5  # TODO: change this.
            # T = [[0, 0, 2]]  # TODO: Change this.
            T = [[0, 0, distance]]
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
            rend = renderer(mesh, cameras=cameras, lights=lights)
            rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
            renders.append(rend)

        images = []
        for i, r in enumerate(renders):
            image = Image.fromarray((r * 255).astype(np.uint8))
            draw = ImageDraw.Draw(image)
            draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
            images.append(np.array(image))
        imageio.mimsave(output_file, images, duration=200, loop=0)
    
    def rotate_cow(self):
    # Function to rotate the cow mesh and save images

        mesh  = self.rotate_mesh.to(self.device)
        relative_transforms = [
        {"R_relative": [[0, -1, 0], [1, 0, 0], [0, 0, 1]], "T_relative": [0, 0, 3]},
        {"R_relative": [[0, 0, -1], [0, 1, 0], [1, 0, 0]], "T_relative": [0, 0, 3]},
        {"R_relative": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "T_relative":  [0, 0, 6]},
        {"R_relative": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "T_relative": [0.5, -0.5, 3]}
    ]
        

        # Print the list of dictionaries
        for i, transform in enumerate(relative_transforms, start=1):
            
            R_relative=transform["R_relative"]
            T_relative=transform["T_relative"]
            

            R_relative = torch.tensor(R_relative).float()
            T_relative = torch.tensor(T_relative).float()
            R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
            T = R_relative @ torch.tensor([0.0, 0, 0]) + T_relative

            cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R.t().unsqueeze(0),
            T=T.unsqueeze(0),
            fov=60,device=self.device)
            rend = self.renderer(mesh, device=self.device, cameras=cameras, lights=self.lights)
            rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
            plt.imsave("results/rotate_{}.png".format(i),rend)
            
                

cow_renderer = MyCowRenderer()

# Generate and save images
cow_renderer.generate_mesh()
cow_renderer.generate_mesh(types="colored")
cow_renderer.generate_mesh(types="tetra")
cow_renderer.generate_mesh(types="cube")
# Generate and save gifs
cow_renderer.generate_turntable_views(types="colored")
cow_renderer.generate_turntable_views(types="cow")
cow_renderer.generate_turntable_views(types="dolly")
cow_renderer.generate_turntable_views(types="tetra")
cow_renderer.generate_turntable_views(types="cube")


# Generate and save dolly gif
cow_renderer.generate_dolly()


# Generate and save rotating cow images
cow_renderer.rotate_cow()

class MyPlantRenderer:
    """
    Renders plant & geometric meshes using Pytorch3D.

    Args:
        path (str): The path to the cow mesh.
        image_size (int): The rendered image size.
        color (list): The color of the rendered mesh.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    Parameters:
        renderer (MeshRenderer): The Pytorch3D mesh renderer.
        lights (PointLights): The Pytorch3D lights object.
        mesh (Meshes): The Pytorch3D mesh object.
        textured_mesh (Meshes): The Pytorch3D mesh object with a texture.
        tetra_mesh (Meshes): The Pytorch3D tetrahedron mesh object.
        cube_mesh (Meshes): The Pytorch3D cube mesh object.
        dolly_mesh (Meshes): The Pytorch3D mesh object for the dolly.
        rotate_mesh (Meshes): The Pytorch3D mesh object for the rotating cow.
    """
    def __init__(self, path="data/cow.obj",image_size=512, 
                 color=[0.7, 0.7, 1],device=None):
        # Constructor to initialize the class
        # Determine the device to use (GPU or CPU)
        # Initialize variables for point clouds and RGBA values
        # Generate the point cloud and RGBA values
        # Call functions to render the plant, torus, and torus mesh

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print("Using CUDA")
            else:
                self.device = torch.device("cpu")
                print("Using CPU")
        self.points_1=None
        self.rgba_1=None
        self.points_2=None
        self.rgba_2=None
        self.points_3=None
        self.rgba_3=None
        self.generate_point_cloud()
        self.points=[self.points_1,self.points_2,self.points_3]
        self.rgba=[self.rgba_1,self.rgba_2,self.rgba_3]
        self.render_plant(self.points,self.rgba)
        self.render_torus()
        self.render_torus_mesh()
    def load_rgbd_data(self):
        # Function to load RGBD data from a file
        with open("data/rgbd_data.pkl", "rb") as f:
            data = pickle.load(f)
        return data
    def generate_point_cloud(self):
        # Function to generate a point cloud from RGBD data
        data_dict=self.load_rgbd_data()
        rgb1 = data_dict['rgb1']
        mask1 = data_dict['mask1']
        depth1 = data_dict['depth1']
        rgb2 = data_dict['rgb2']
        mask2 = data_dict['mask2']
        depth2 = data_dict['depth2']
        cameras1 = data_dict['cameras1']
        cameras2 = data_dict['cameras2']

        image_1 = torch.tensor(rgb1, dtype=torch.float32)
        mask_1 = torch.tensor(mask1, dtype=torch.float32)
        depth_1 = torch.tensor(depth1, dtype=torch.float32)
        cameras_1 = cameras1
        image_2 = torch.tensor(rgb2, dtype=torch.float32)
        mask_2 = torch.tensor(mask2, dtype=torch.float32)
        depth_2 = torch.tensor(depth2, dtype=torch.float32)
        cameras_2 = cameras2

        R_np = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        R_torch = torch.tensor(R_np, dtype=torch.float32)
        points_1, rgba_1 = unproject_depth_image(image_1, mask_1, depth_1, cameras_1)
        points_2, rgba_2 = unproject_depth_image(image_2, mask_2, depth_2, cameras_2)
        points_1=torch.matmul(points_1,R_torch)
        points_2=torch.matmul(points_2,R_torch)
        points_3=torch.concat([points_1,points_2],dim=0)
        rgba_3=torch.concat([rgba_1,rgba_2],dim=0)
        self.points_1=points_1
        self.rgba_1=rgba_1
        self.points_2=points_2
        self.rgba_2=rgba_2
        self.points_3=points_3
        self.rgba_3=rgba_3

    def render_plant(self,
    points,
    rgba,
    image_size=800,
    background_color=(1, 1, 1),
    device=None, num_views=36
    ):
        # Function to render the plant using point cloud data
        for i in range(0,3):
            """
            Renders a point cloud.
            """
            if device is None:
                if torch.cuda.is_available():
                    device = torch.device("cuda:0")
                    
                else:
                    device = torch.device("cpu")
            renderer = get_points_renderer(
                image_size=image_size, background_color=background_color
            )
            
            verts = points[i].to(device).unsqueeze(0)
            rgb = rgba[i].to(device).unsqueeze(0)
            point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
            R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0)
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
            rend = renderer(point_cloud, cameras=cameras)
            rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
            plt.imsave("results/plant_{}.png".format(i),rend)


            # Generate evenly spaced azimuth angles
            azimuths = torch.linspace(0, 360, num_views)
    
            
            # Fixed elevation and distance
            elevations = torch.tensor([30.0] * num_views)  # Fixed elevation
            distances = torch.tensor([5.0] * num_views)    # Fixed distance
            
            # Generate views using look_at_view_transform
            R, T = pytorch3d.renderer.look_at_view_transform(distances, elevations, azimuths)
            # print(R.unsqueeze(0))
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=R,
            T=T,
            fov=60,
            device=self.device)
            #Render images for each view
            images=[]
            for k in cameras:
                img=(renderer(point_cloud, cameras=k))
                img=img.cpu().numpy()[0, ..., :3]
                images.append(img)
            
            images_pil = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
            imageio.mimsave("results/plant_rot_{}.gif".format(i), images_pil, duration=30, loop=0)

    def render_torus(self, image_size=256, num_samples=200, device=None,num_views=36):
        """
        Renders a torus using parametric sampling. Samples num_samples ** 2 points.
        """

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")

        phi = torch.linspace(0, 2 * np.pi, num_samples)
        theta = torch.linspace(0, 2 * np.pi, num_samples)
        # Densely sample phi and theta on a grid
        Phi, Theta = torch.meshgrid(phi, theta)
        r=0.2
        R=1

        x = (R+r*torch.cos(Theta)) * torch.cos(Phi)
        y = (R+r*torch.cos(Theta)) * torch.sin(Phi)
        z = r*torch.sin(Theta) 

        points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
        color = (points - points.min()) / (points.max() - points.min())

        sphere_point_cloud = pytorch3d.structures.Pointclouds(
            points=[points], features=[color],
        ).to(device)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
        renderer = get_points_renderer(image_size=image_size, device=device)
        rend = renderer(sphere_point_cloud, cameras=cameras)
        rend=rend[0, ..., :3].cpu().numpy()
        plt.imsave("results/torus_para.png",rend)

        # Generate evenly spaced azimuth angles
        azimuths = torch.linspace(0, 360, num_views)

        
        # Fixed elevation and distance
        elevations = torch.tensor([30.0] * num_views)  # Fixed elevation
        distances = torch.tensor([5.0] * num_views)    # Fixed distance
        
        # Generate views using look_at_view_transform
        R, T = pytorch3d.renderer.look_at_view_transform(distances, elevations, azimuths)
        # print(R.unsqueeze(0))
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        fov=60,
        device=self.device)
        #Render images for each view
        images=[]
        for k in cameras:
            img=(renderer(sphere_point_cloud, cameras=k))
            img=img.cpu().numpy()[0, ..., :3]
            images.append(img)
        
        images_pil = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
        imageio.mimsave("results/torus_para_rot.gif", images_pil, duration=30, loop=0)


    def render_torus_mesh(self, image_size=256, voxel_size=64, device=None, num_views=36):
        if device is None:
            # Function to render a torus mesh using voxelization
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")

        r=0.2
        R=0.7
        min_value = -1.5
        max_value = 1.5
        X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
        voxels = (np.sqrt(X ** 2 + Y ** 2) - R) ** 2 + Z ** 2 - r**2
        vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
        vertices = torch.tensor(vertices).float()
        faces = torch.tensor(faces.astype(int))
        # Vertex coordinates are indexed by array position, so we need to
        # renormalize the coordinate system.
        vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
        textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
        textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

        mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
            device
        )
        lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
        renderer = get_mesh_renderer(image_size=image_size, device=device)
        R, T = pytorch3d.renderer.look_at_view_transform(dist=2, elev=0, azim=180)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend=rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
        plt.imsave("results/torus_implicit.png",rend)

        # Generate evenly spaced azimuth angles
        azimuths = torch.linspace(0, 360, num_views)

        
        # Fixed elevation and distance
        elevations = torch.tensor([30.0] * num_views)  # Fixed elevation
        distances = torch.tensor([5.0] * num_views)    # Fixed distance
        
        # Generate views using look_at_view_transform
        R, T = pytorch3d.renderer.look_at_view_transform(distances, elevations, azimuths)
        # print(R.unsqueeze(0))
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R,
        T=T,
        fov=60,
        device=self.device)
        #Render images for each view
        images=[]
        for k in cameras:
            img=(renderer(mesh, cameras=k))
            img=img.cpu().numpy()[0, ..., :3]
            images.append(img)
        
        images_pil = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
        imageio.mimsave("results/torus_implicit_rot.gif", images_pil, duration=30, loop=0)


# Render the plant and the torus
plant=MyPlantRenderer()

        