"""
This script creates either an window or HTML to visualize the point clouds and gripper state.
"""

import os
import time
import json
import argparse

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R


class PointCloudVisualizer:
    def __init__(self, pcl_dir, update_interval=0.04):
        self.pcl_dir = pcl_dir
        self.update_interval = update_interval
        self.idx = 0
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.pcd = o3d.geometry.PointCloud()

        self.curr_gripper_vis = None
        self.curr_gripper_transform = None

    def get_transform_from_pos_quat(self, pos, quat):
        pose_transform = np.eye(4)
        rotation_matrix = R.from_quat(quat).as_matrix()
        pose_transform[:3, :3] = rotation_matrix
        pose_transform[:, 3][:-1] = pos
        return pose_transform

    def get_latest_point_cloud(self):
        if self.idx >= len(os.listdir(self.pcl_dir)):
            return
        pcl_data = np.load(
            "%s/%05d.npy" % (self.pcl_dir, self.idx), allow_pickle=True
        ).item()
        points = pcl_data["points"]
        colors = pcl_data["colors"]
        curr_ee_pos = pcl_data["curr_ee_pos"]
        curr_ee_quat = pcl_data["curr_quat"]
        curr_gripper_state = pcl_data["curr_gripper_state"]

        pose_transform = self.get_transform_from_pos_quat(curr_ee_pos, curr_ee_quat)

        if self.curr_gripper_vis is None:
            self.curr_gripper_vis = o3d.io.read_triangle_mesh("vis_assets/robotiq.obj")
        else:
            # because the curr_gripper_transform is absolute instead of relative, we need to "undo" the prev one before applying the latest when visualizing
            self.curr_gripper_vis.transform(np.linalg.inv(self.curr_gripper_transform))

        if curr_gripper_state:
            self.curr_gripper_vis.paint_uniform_color([0.7, 0.3, 0.1])
        else:
            self.curr_gripper_vis.paint_uniform_color([0.0, 0.4, 0.3])

        self.curr_gripper_vis.transform(pose_transform)
        self.curr_gripper_transform = pose_transform

        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        self.idx += 1

    def update_point_cloud(self, vis):
        self.get_latest_point_cloud()
        vis.update_geometry(self.curr_gripper_vis)
        vis.update_geometry(self.pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(self.update_interval)  # Adjust the sleep time as needed
        return False

    def run(self):
        self.vis.create_window()

        self.get_latest_point_cloud()
        self.vis.add_geometry(self.curr_gripper_vis)
        self.vis.add_geometry(self.pcd)

        self.vis.register_animation_callback(self.update_point_cloud)
        self.vis.run()
        self.vis.destroy_window()


class PointCloudVisualizerHTML:
    def __init__(
        self,
        pcl_dir,
        horizon,
        output_file="point_cloud_visualization.html",
        sample_rate=0.1,
        initial_camera_pos=[0, 0, 2],
        initial_camera_target=[0, 0, 0],
        initial_camera_up=[0, 1, 0],
    ):
        self.pcl_dir = pcl_dir
        self.output_file = output_file
        self.sample_rate = sample_rate
        self.initial_camera_pos = initial_camera_pos
        self.initial_camera_target = initial_camera_target
        self.initial_camera_up = initial_camera_up
        self.data = []
        self.horizon = horizon

    def process_point_clouds(self):
        files = sorted(os.listdir(self.pcl_dir))
        partition_to_idx = {
            "train": 0,
            "val": 0,
        }
        for idx, filename in enumerate(files):
            partition = filename.split("_")[0]
            if filename.endswith(".npy"):
                pcl_data = np.load(
                    os.path.join(self.pcl_dir, filename), allow_pickle=True
                ).item()
                points = pcl_data["points"]
                colors = pcl_data["colors"]
                curr_ee_pos = pcl_data["curr_ee_pos"]
                curr_gripper_state = pcl_data["curr_gripper_state"]
                episode_end_idx = pcl_data["episode_end_idx"]

                if not isinstance(curr_ee_pos[0], (list, np.ndarray)):
                    curr_ee_pos = [curr_ee_pos]

                num_points = len(points)
                sample_size = int(num_points * self.sample_rate)
                indices = np.random.choice(num_points, sample_size, replace=False)
                sampled_points = points[indices]
                sampled_colors = colors[indices]

                frame_data = {
                    "points": sampled_points.tolist(),
                    "colors": sampled_colors.tolist(),
                    "ee_pos": [pos.tolist() for pos in curr_ee_pos],
                    "gripper_state": bool(curr_gripper_state),
                    "future_ee_pos": [],
                }

                end = min(self.horizon, episode_end_idx - partition_to_idx[partition])
                for future_idx in range(1, end):
                    if idx + future_idx < len(files):
                        future_file = files[idx + future_idx]
                        future_data = np.load(
                            os.path.join(self.pcl_dir, future_file), allow_pickle=True
                        ).item()
                        future_ee_pos = future_data["curr_ee_pos"]

                        if not isinstance(future_ee_pos[0], (list, np.ndarray)):
                            future_ee_pos = [future_ee_pos]

                        frame_data["future_ee_pos"].append(
                            [pos.tolist() for pos in future_ee_pos]
                        )

            partition_to_idx[partition] += 1
            self.data.append(frame_data)

    def create_visualization(self):
        self.process_point_clouds()

        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Point Cloud Visualization</title>
            <style>
                body { margin: 0; }
                canvas { display: block; }
                #controls {
                    position: absolute;
                    bottom: 10px;
                    left: 10px;
                    right: 10px;
                    background-color: rgba(0,0,0,0.7);
                    color: white;
                    padding: 10px;
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                #gifButton { margin-left: 10px; }
                #cameraInfo { margin-right: 10px; }
                #instructions {
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    background-color: rgba(0,0,0,0.7);
                    color: white;
                    padding: 10px;
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div id="instructions">
                <h3>Controls:</h3>
                <p>Rotate: Left-click and drag</p>
                <p>Pan: Right-click and drag, or Ctrl+left-click and drag</p>
                <p>Zoom: Scroll, or Shift+left-click and drag up/down</p>
            </div>
            <div id="controls">
                <div id="frameControl"></div>
                <button id="saveVideoButton">Save Video</button>
                <div id="cameraInfo"></div>
            </div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.7/dat.gui.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/gif.js.optimized/dist/gif.js"></script>
            <script>
                const data = POINT_CLOUD_DATA;
                let scene, camera, renderer, points, controls;
                let currentFrame = 0;
                let eeMarkers; 

                function init() {
                    scene = new THREE.Scene();
                    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
                    renderer = new THREE.WebGLRenderer();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                    document.body.appendChild(renderer.domElement);

                    controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.25;
                    controls.screenSpacePanning = true;
                    controls.minDistance = 0.1;
                    controls.maxDistance = 10;
                    controls.enableRotate = true;
                    controls.enableZoom = true;
                    controls.enablePan = true;

                    const geometry = new THREE.BufferGeometry();
                    const material = new THREE.PointsMaterial({ size: 0.01, vertexColors: true });

                    points = new THREE.Points(geometry, material);
                    scene.add(points);

                    eeMarkers = new THREE.Group();
                    scene.add(eeMarkers); // Add the group to the scene

                    camera.position.set(0, 0, 2);
                    controls.update();

                    const gui = new dat.GUI({ autoPlace: false });
                    const frameControl = gui.add({ frame: 0 }, 'frame', 0, data.length - 1, 1).onChange(updateFrame);
                    document.getElementById('frameControl').appendChild(gui.domElement);

                    window.addEventListener('resize', onWindowResize, false);

                    document.getElementById('saveVideoButton').addEventListener('click', saveVideo);

                    updateFrame(0);
                    animate();
                }

                function updateFrame(frame) {
                    currentFrame = Math.floor(frame);
                    const frameData = data[currentFrame];

                    const positions = new Float32Array(frameData.points.flat());
                    const colors = new Float32Array(frameData.colors.flat());

                    points.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                    points.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                    points.geometry.attributes.position.needsUpdate = true;
                    points.geometry.attributes.color.needsUpdate = true;

                    // Remove all existing markers
                    while(eeMarkers.children.length > 0){ 
                        eeMarkers.remove(eeMarkers.children[0]); 
                    }

                    const totalPositions = 1 + frameData.future_ee_pos.length; // Current + future positions
                    
                    // Function to interpolate between red and blue
                    function interpolateColor(factor) {
                        const r = Math.floor(255 * (1 - factor));
                        const b = Math.floor(255 * factor);
                        return new THREE.Color(r / 255, 0, b / 255);
                    }

                    // Add current end-effector markers
                    frameData.ee_pos.forEach(pos => {
                        const marker = new THREE.Mesh(
                            new THREE.SphereGeometry(0.02, 32, 32),
                            new THREE.MeshBasicMaterial({ color: interpolateColor(0) })
                        );
                        marker.position.set(...pos);
                        eeMarkers.add(marker);
                    });

                    // Add future end-effector markers
                    frameData.future_ee_pos.forEach((futurePositions, index) => {
                        const colorFactor = (index + 1) / totalPositions;
                        futurePositions.forEach(pos => {
                            const marker = new THREE.Mesh(
                                new THREE.SphereGeometry(0.015, 32, 32),
                                new THREE.MeshBasicMaterial({ 
                                    color: interpolateColor(colorFactor),
                                    transparent: true,
                                    opacity: 1 - colorFactor * 0.5 // Slight fade-out effect
                                })
                            );
                            marker.position.set(...pos);
                            eeMarkers.add(marker);
                        });
                    });

                    const boundingBox = new THREE.Box3().setFromObject(points);
                    const center = boundingBox.getCenter(new THREE.Vector3());
                    controls.target.copy(center);
                    controls.update();

                    updateCameraInfo();
                }

                function updateCameraInfo() {
                    const pos = camera.position.toArray().map(x => x.toFixed(2));
                    const target = controls.target.toArray().map(x => x.toFixed(2));
                    const up = camera.up.toArray().map(x => x.toFixed(2));
                    document.getElementById('cameraInfo').innerHTML = `
                        Camera: pos=[${pos}], target=[${target}], up=[${up}]
                    `;
                }

                function onWindowResize() {
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                }

                function animate() {
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                    updateCameraInfo();
                }

                function saveVideo() {
                    const videoButton = document.getElementById('saveVideoButton');
                    videoButton.disabled = true;
                    videoButton.textContent = 'Saving Video...';

                    const stream = renderer.domElement.captureStream(25); // 25 FPS
                    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm; codecs=vp9' });
                    const chunks = [];

                    recorder.ondataavailable = function(event) {
                        if (event.data.size > 0) {
                            chunks.push(event.data);
                        }
                    };

                    recorder.onstop = function() {
                        const blob = new Blob(chunks, { type: 'video/webm' });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `point_cloud_animation_${Date.now()}.webm`;
                        document.body.appendChild(a);
                        a.click();
                        setTimeout(() => {
                            document.body.removeChild(a);
                            window.URL.revokeObjectURL(url);
                            videoButton.disabled = false;
                            videoButton.textContent = 'Save Video';
                        }, 100);
                    };

                    recorder.start();

                    let frame = 0;
                    const totalFrames = data.length;

                    function addFrame() {
                        if (frame < totalFrames) {
                            updateFrame(frame);
                            renderer.render(scene, camera);
                            frame++;
                            if (frame === totalFrames) {
                                recorder.stop();
                            } else {
                                setTimeout(addFrame, 100); // Adjust frame capture rate if necessary
                            }
                        }
                    }

                    addFrame();
                }
                init();
            </script>
        </body>
        </html>
        """

        html_content = html_content.replace("POINT_CLOUD_DATA", json.dumps(self.data))

        with open(self.output_file, "w") as f:
            f.write(html_content)


if __name__ == "__main__":
    pcl_dir = "data/pc_tracks/vis"
    fns = list(sorted([fn for fn in os.listdir(pcl_dir) if "npz" in fn]))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--html",
        action="store_true",
        help="Whether to generate an HTML visualization instead of a live visualization",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    if args.html:
        # Workaround since MacM1 can't open visualizer
        # Use the camera parameters you found from the interactive view
        initial_camera_pos = [0, 0, 2]
        initial_camera_target = [0, 0, 0]
        initial_camera_up = [0, 1, 0]
        visualizer = PointCloudVisualizerHTML(
            pcl_dir,
            args.horizon,
            output_file="data/pc_tracks/vis.html",
            initial_camera_pos=initial_camera_pos,
            initial_camera_target=initial_camera_target,
            initial_camera_up=initial_camera_up,
        )
        visualizer.create_visualization()
    else:
        visualizer = PointCloudVisualizer(pcl_dir)
        visualizer.run()
