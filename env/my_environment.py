import time
import glob
import numpy as np
import random
import os
import pybullet as pb
import pybullet_data    
import env.cameras as cameras
from scipy.spatial.transform import Rotation as R
from env.constants import PIXEL_SIZE, WORKSPACE_LIMITS
import math
class Environment:
    def __init__(self, gui=True, time_step=1 / 240):
        """Creates environment with PyBullet.
        Args:
        gui: show environment with PyBullet's built-in display viewer
        time_step: PyBullet physics simulation step speed. Default is 1 / 240.
        """
        self.time_step = time_step
        self.gui = gui
        self.pixel_size = PIXEL_SIZE
        self.obj_ids = {"fixed": [], "rigid": []}
        self.agent_cams = cameras.RealSenseD435.CONFIG
        self.oracle_cams = cameras.Oracle.CONFIG
        self.bounds = WORKSPACE_LIMITS
        self.home_joints = np.array([0, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.ik_rest_joints = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.drop_joints0 = np.array([0.5, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.drop_joints1 = np.array([1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi

        # Start PyBullet.
        self._client_id = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setTimeStep(time_step)
        pb.setGravity(0, 0, -9.8)
        if gui:
            target = pb.getDebugVisualizerCamera()[11]
            pb.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=90, cameraPitch=-25, cameraTargetPosition=target,
            )
    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [
            np.linalg.norm(pb.getBaseVelocity(i, physicsClientId=self._client_id)[0])
            for i in self.obj_ids["rigid"]
        ]
        return all(np.array(v) < 5e-3)
    
    @property
    def is_gripper_closed(self):
        gripper_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]
        return gripper_angle < self.gripper_angle_close_threshold
    
    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        info = {}  # object id : (position, rotation, dimensions)
        for obj_id in self.object_ids:
            
            pos, rot = pb.getBasePositionAndOrientation(
                obj_id, physicsClientId=self._client_id
            )
            dim = pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
            info[obj_id] = (pos, rot, dim)
        return info

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def obj_info(self, obj_id):
        """Environment info variable with object poses, dimensions, and colors."""

        pos, rot = pb.getBasePositionAndOrientation(
            obj_id, physicsClientId=self._client_id
        )
        dim = pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
        fixed_rot = pb.getQuaternionFromEuler([np.pi, 0, 0])
        rot = fixed_rot
        info = (pos, rot, dim)
        return info


    def get_link_pose(self,body, link):
        result = pb.getLinkState(body, link)
        return result[4], result[5]
    
    def go_home(self):
        return self.move_joints(self.home_joints)
    
    def close_gripper(self, is_slow=True):
        self._move_gripper(self.gripper_angle_close, is_slow=is_slow)

    def open_gripper(self, is_slow=False):
        self._move_gripper(self.gripper_angle_open, is_slow=is_slow)
    
    def wait_static(self, timeout=3):
        """Step simulator asynchronously until objects settle."""
        pb.stepSimulation()
        t0 = time.time()
        while (time.time() - t0) < timeout:
            if self.is_static:
                return True
            pb.stepSimulation()
        print(f"Warning: Wait static exceeded {timeout} second timeout. Skipping.")
        return False

    def solve_ik(self, pose):
            """Calculate joint configuration with inverse kinematics."""
            joints = pb.calculateInverseKinematics(
                bodyUniqueId=self.ur5e,
                endEffectorLinkIndex=self.ur5e_ee_id,
                targetPosition=pose[0],
                targetOrientation=pose[1],
                lowerLimits=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283],
                upperLimits=[6.283, 6.283, 3.141, 6.283, 6.283, 6.283],
                jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566],
                restPoses=np.float32(self.ik_rest_joints).tolist(),
                # maxNumIterations=100,
                # residualThreshold=1e-5,
            )
            joints = np.array(joints, dtype=np.float32)
            # joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
            return joints
    
    def get_true_object_pose(self, obj_id):
        pos, ort = pb.getBasePositionAndOrientation(obj_id)
        position = np.array(pos).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(ort)
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        return transform
    
    def get_true_object_poses(self):
        transforms = dict()
        for obj_id in self.obj_ids["rigid"]:
            transform = self.get_true_object_pose(obj_id)
            transforms[obj_id] = transform
        return transforms

    def add_objects(self, num_obj, workspace_limits):
        """Randomly dropped objects to the workspace"""
        self.object_ids = [] 
        self.obj_folder = "assets/obj"
        # self.obj_folder = "assets/ycb"
        self.obj_list = ['0.obj','1.obj','2.obj','3.obj','4.obj','6.obj']
        self.filtered_obj_list = ['0.obj','1.obj','4.obj','6.obj']
        pb.setGravity(0, 0, -10)
        for i in range(num_obj):
            x = np.random.uniform(workspace_limits[0][0]+0.1, workspace_limits[0][1]-0.1)
            y = np.random.uniform(workspace_limits[1][0]+0.1, workspace_limits[1][1]-0.1)
            z = workspace_limits[2][1]-0.1   
            angle = random.uniform(-np.pi, np.pi)
            orientation = pb.getQuaternionFromEuler([0, 0, angle])
            obj_path = os.path.join(self.obj_folder, random.choice(self.filtered_obj_list)) 
            if i == 0:
                color = [1, 0, 0, 1]
                x = (workspace_limits[0][0] + workspace_limits[0][1]) / 2
                y = np.random.uniform(workspace_limits[1][0]+0.3, workspace_limits[1][1]-0.3)
                # y = (workspace_limits[1][0] + workspace_limits[1][1]) / 2
            else:
                color = [0.5, random.random(), random.random(), 1]

            vis_id = pb.createVisualShape(
                shapeType=pb.GEOM_MESH,
                fileName=obj_path,
                meshScale=[1, 1, 1], 
                rgbaColor=color
            )
            max_retry = 3
            for attempt in range(max_retry):
                try:
                    collision_id = pb.createCollisionShape(
                        shapeType=pb.GEOM_MESH,
                        fileName=obj_path,
                        meshScale=[1, 1, 1]
                    )
                    break  
                except Exception as e:
                    print(f"[Attempt {attempt+1}] Failed to create collision shape: {e}")
                    time.sleep(0.1)  
            else:
                raise RuntimeError(f"Failed to create collision shape for {obj_path} after {max_retry} attempts.")

            body_id = pb.createMultiBody(
                baseMass=0.1,  
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[x, y, z],
                baseOrientation=orientation
            )
            self.object_ids.append(body_id)
            for _ in range(240):  
                pb.stepSimulation()
                time.sleep(1/500)
            self.wait_static()
        return self.object_ids

    def add_one_objects(self, num_obj, workspace_limits):
        """Randomly dropped objects to the workspace"""
        self.obj_ids = {"fixed": [], "rigid": []}
        pb.resetSimulation()

        # Temporarily disable rendering to load scene faster.
        if self.gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        if self.gui:
            pb.configureDebugVisualizer(
            pb.COV_ENABLE_RENDERING, 1, physicsClientId=self._client_id
            )   
        self.object_ids = []  
        self.obj_folder = "assets/obj"
        self.obj_list = ['0.obj']
        # self.filtered_obj_list = ['0.obj','1.obj','4.obj','6.obj']
        # pb.setGravity(0, 0, -10)
        for i in range(num_obj):

            x = (workspace_limits[0][0] + workspace_limits[0][1]) / 2
            y = (workspace_limits[1][0] + workspace_limits[1][1]) / 2
            z = 0  

            # angle = random.uniform(-np.pi, np.pi)
            orientation = pb.getQuaternionFromEuler([0, 0, np.pi /2 ])
            # if i == 0:
            #     color = [1, 0, 0, 1]
            #     obj_path = os.path.join(self.obj_folder, random.choice(self.filtered_obj_list)) 
            #     x = (workspace_limits[0][0] + workspace_limits[0][1]) / 2
            #     y = np.random.uniform(workspace_limits[1][0]+0.3, workspace_limits[1][1]-0.3)
            # else:
            color = [0.5, random.random(), random.random(), 1]
            obj_path = os.path.join(self.obj_folder, self.obj_list[0])  
            vis_id = pb.createVisualShape(
                shapeType=pb.GEOM_MESH,
                fileName=obj_path,
                meshScale=[1, 1, 1],  
                rgbaColor=color
            )

            max_retry = 3
            for attempt in range(max_retry):
                try:
                    collision_id = pb.createCollisionShape(
                        shapeType=pb.GEOM_MESH,
                        fileName=obj_path,
                        meshScale=[1, 1, 1]
                    )
                    break  
                except Exception as e:
                    print(f"[Attempt {attempt+1}] Failed to create collision shape: {e}")
                    time.sleep(0.1) 
            else:
                raise RuntimeError(f"Failed to create collision shape for {obj_path} after {max_retry} attempts.")

            body_id = pb.createMultiBody(
                baseMass=0.1, 
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[x, y, z],
                baseOrientation=orientation
            )
            self.object_ids.append(body_id)
            for _ in range(240):  
                pb.stepSimulation()
                time.sleep(1/500)
            self.wait_static()
        return self.object_ids

    def load_objects_from_txt(self, txt_path):

        self.obj_folder = "assets/obj"
        self.object_ids = []
        with open(txt_path, "r") as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                obj_file = parts[0]
                try:
                    nums = list(map(float, parts[1:]))   
                except Exception as e:
                    raise ValueError(f"[line {ln}] number parsing failed:{parts[1:]} ({e})")
                """add challenge objects to the workspace"""
                pb.setGravity(0, 0, -10)
                xyz = nums[0:3]
                theta = nums[5]
                
                euler_angles = nums[3:]
                initial_orientation = pb.getQuaternionFromEuler(euler_angles)
                # orientation = pb.getQuaternionFromEuler(nums[3:])
                color = [random.random(), random.random(), random.random(), 1]
                obj_path = os.path.join(self.obj_folder, obj_file)  
                vis_id = pb.createVisualShape(
                    shapeType=pb.GEOM_MESH,
                    fileName=obj_path,
                    meshScale=[1, 1, 1],  
                    rgbaColor=color
                )
                max_retry = 3
                for attempt in range(max_retry):
                    try:
                        collision_id = pb.createCollisionShape(
                            shapeType=pb.GEOM_MESH,
                            fileName=obj_path,
                            meshScale=[1, 1, 1],
                        )
                        break  
                    except Exception as e:
                        print(f"[Attempt {attempt+1}] Failed to create collision shape: {e}")
                        time.sleep(0.1)  
                else:
                    raise RuntimeError(f"Failed to create collision shape for {obj_path} after {max_retry} attempts.")

                body_id = pb.createMultiBody(
                    baseMass=0.1,  
                    baseCollisionShapeIndex=collision_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=xyz,
                    baseOrientation=initial_orientation
                )
                self.object_ids.append(body_id)
                for _ in range(240):  
                    pb.stepSimulation()
                    time.sleep(1/500)
                self.wait_static()
        return self.object_ids

    def render_camera(self, config):
        """Render RGB-D image with specified camera configuration."""

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(config["rotation"])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0, 0]
        znear, zfar = config["zrange"]
        viewm = pb.computeViewMatrix(config["position"], lookat, updir)
        fovh = (config["image_size"][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config["image_size"][1] / config["image_size"][0]
        projm = pb.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = pb.getCameraImage(
            width=config["image_size"][1],
            height=config["image_size"][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=0,
            flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        )

        # Get color image.
        color_image_size = (config["image_size"][0], config["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config["image_size"][0], config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)


        return color, depth, segm

    def reset(self):  
        self.obj_ids = {"fixed": [], "rigid": []}
        pb.resetSimulation()
        pb.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        if self.gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        # Load workspace
        self.plane = pb.loadURDF(
            "plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True,
        )
        self.workspace = pb.loadURDF(
            "assets/workspace/workspace.urdf", basePosition=(0.5, 0, 0), useFixedBase=True,
        )
        pb.changeDynamics(
            self.plane,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )
        pb.changeDynamics(
            self.workspace,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )

        # Load UR5e
        self.ur5e = pb.loadURDF(
            "assets/ur5e/ur5e.urdf",
            basePosition=(0, 0, 0),
            useFixedBase=True,
        )
        self.ur5e_joints = []
        for i in range(pb.getNumJoints(self.ur5e)):
            info = pb.getJointInfo(self.ur5e, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "ee_fixed_joint":
                self.ur5e_ee_id = joint_id
            if joint_type == pb.JOINT_REVOLUTE:
                self.ur5e_joints.append(joint_id)
        pb.enableJointForceTorqueSensor(self.ur5e, self.ur5e_ee_id, 1)

        self.setup_gripper()

        # Move robot to home joint configuration.
        success = self.go_home()
        self.close_gripper()
        self.open_gripper()

        if not success:
            print("Simulation is wrong!")
            exit()

        # Re-enable rendering.
        if self.gui:
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_RENDERING, 1, physicsClientId=self._client_id
            )
        return self.ur5e
    
    def setup_gripper(self): 
        """Load end-effector: gripper"""
        ee_position, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        self.ee = pb.loadURDF(
            "assets/ur5e/gripper/robotiq_2f_85.urdf",
            ee_position,
            pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
        )
        self.ee_tip_z_offset = 0.1625
        self.gripper_angle_open = 0.03
        self.gripper_angle_close = 0.8
        self.gripper_angle_close_threshold = 0.73
        self.gripper_mimic_joints = {
            "left_inner_finger_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_outer_knuckle_joint": -1,
            "right_inner_finger_joint": -1,
            "right_inner_knuckle_joint": -1,
        }
        for i in range(pb.getNumJoints(self.ee)):
            info = pb.getJointInfo(self.ee, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "finger_joint":
                self.gripper_main_joint = joint_id
            elif joint_name == "dummy_center_fixed_joint":
                self.ee_tip_id = joint_id
            elif "finger_pad_joint" in joint_name:
                pb.changeDynamics(
                    self.ee, joint_id, lateralFriction=1.8 # intial 0.9 change to 1.0
                )
                self.ee_finger_pad_id = joint_id
            elif joint_type == pb.JOINT_REVOLUTE:
                self.gripper_mimic_joints[joint_name] = joint_id
                # Keep the joints static
                pb.setJointMotorControl2(
                    self.ee, joint_id, pb.VELOCITY_CONTROL, targetVelocity=0, force=0,
                )
                
        self.ee_constraint = pb.createConstraint(
            parentBodyUniqueId=self.ur5e,
            parentLinkIndex=self.ur5e_ee_id,
            childBodyUniqueId=self.ee,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.02),
            childFrameOrientation=pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(self.ee_constraint, maxForce=10000)
        pb.enableJointForceTorqueSensor(self.ee, self.gripper_main_joint, 1)

        # Set up mimic joints in robotiq gripper: left
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: right
        c = pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: connect left and right
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=1000)

    def move_joints(self, target_joints, speed=0.01, timeout=10):
        """Move UR5e to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            current_joints = np.array(
                [
                    pb.getJointState(self.ur5e, i, physicsClientId=self._client_id)[0]
                    for i in self.ur5e_joints
                ]
            )
            pos, _ = self.get_link_pose(self.ee, self.ee_tip_id)
            
            if pos[2] < 0.005:
                print(f"Warning: move_joints tip height is {pos[2]}. Skipping.")
                return False
            diff_joints = target_joints - current_joints
            if all(np.abs(diff_joints) < 0.05):
                # give time to stop
                for _ in range(5):
                    pb.stepSimulation()
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diff_joints)
            v = diff_joints / norm if norm > 0 else 0
            step_joints = current_joints + v * speed
            pb.setJointMotorControlArray(
                bodyIndex=self.ur5e,
                jointIndices=self.ur5e_joints,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=step_joints,
                positionGains=np.ones(len(self.ur5e_joints)),
            )
            pb.stepSimulation()
        print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
        return False
    
    def move_ee_pose(self, pose, speed=0.01):
        """Move UR5e to target end effector pose."""
        target_joints = self.solve_ik(pose)
        return self.move_joints(target_joints, speed)
    
    def straight_move(self, pose0, pose1, rot, speed=0.01, max_force=300, detect_force=False, is_push=False):
        """Move every 1 cm, keep the move in a straight line instead of a curve. Keep level with rot"""
        step_distance = 0.01  # every 1 cm
        vec = np.float32(pose1) - np.float32(pose0)
        length = np.linalg.norm(vec)
        vec = vec / length
        n_push = np.int32(np.floor(length / step_distance))  # every 1 cm
        success = True
        for n in range(n_push):
            target = pose0 + vec * n * step_distance
            success &= self.move_ee_pose((target, rot), speed)
            if detect_force:
                force = np.sum(
                    np.abs(np.array(pb.getJointState(self.ur5e, self.ur5e_ee_id)[2]))
                )
                if force > max_force:
                    target = target - vec * 2 * step_distance
                    self.move_ee_pose((target, rot), speed)
                    print(f"Force is {force}, exceed the max force {max_force}")
                    return False    
        if is_push:
            speed /= 5
        success &= self.move_ee_pose((pose1, rot), speed)
        return success

    def _move_gripper(self, target_angle, timeout=3, is_slow=False):
        t0 = time.time()
        prev_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]

        if is_slow:
            pb.setJointMotorControl2(
                self.ee,
                self.gripper_main_joint,
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            pb.setJointMotorControl2(
                self.ee,
                self.gripper_mimic_joints["right_outer_knuckle_joint"],
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            for _ in range(10):
                pb.stepSimulation()
            while (time.time() - t0) < timeout:
                current_angle = pb.getJointState(self.ee, self.gripper_main_joint)[0]
                diff_angle = abs(current_angle - prev_angle)
                if diff_angle < 1e-4:
                    break
                prev_angle = current_angle
                for _ in range(10):
                    pb.stepSimulation()
        # maintain the angles
        pb.setJointMotorControl2(
            self.ee,
            self.gripper_main_joint,
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        pb.setJointMotorControl2(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        for _ in range(10):
            pb.stepSimulation()

    def step(self, pose=None):
        """Execute action with specified primitive.

        Args:
            action: action to execute.

        Returns:
            obs, done
        """
        done = False
        if pose is not None:
            success, grasped_obj_id = self.grasp(pose)
            # Grasping fails
        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            pb.stepSimulation()

        return success, grasped_obj_id, done
    
    def grasp(self, pose, speed=0.002):
        """Execute grasping primitive.

        Args:
            pose: SE(3) grasping pose.

        Returns:
            success: robot movement success if True.
        """
        # Handle unexpected behavior
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )
        transform = pose
        # ee link in tip
        ee_tip_transform = np.array([[0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, -self.ee_tip_z_offset],
                                    [0, 0, 0, 1]])
        # transform from tip to ee link
        ee_transform = transform @ ee_tip_transform
        pos = (ee_transform[:3, 3]).T
        pos[2] = max(pos[2] - 0.02, self.bounds[2][0])
        over = np.array((pos[0], pos[1], pos[2] + 0.2))
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()
        # Execute 6-dof grasping.
        grasped_obj_id = None
        # min_pos_dist = None  
        self.open_gripper()
        success = self.move_joints(self.ik_rest_joints)
        if success:
            success = self.move_ee_pose((over, rot), speed)
            for _ in range(5):
                pb.stepSimulation()
        if success:
            success = self.straight_move(over, pos, rot, speed, detect_force=True)
            for _ in range(10):
                pb.stepSimulation()
        if success:
            self.close_gripper()
            success = self.straight_move(pos, over, rot, speed)
            for _ in range(5):
                pb.stepSimulation()
            success &= self.is_gripper_closed
            if success: # get grasped object id
                max_height = 0.06
                grasped_obj_id = []
                for i in self.object_ids:
                    height = self.info[i][0][2]
                    if height >= max_height:
                        grasped_obj_id.append(i)
                        # break

        if success:
            success = self.move_joints(self.drop_joints1)
            # success &= self.is_gripper_closed
            self.open_gripper(is_slow=True)
        self.go_home()

        # print(f"Grasp at {pose}, the grasp {success}")

        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )

        return success, grasped_obj_id
    
    def is_in_workplace(self,pos):
        is_in_workplace = True
        if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
            or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
            is_in_workplace = False 
        return is_in_workplace

    def push(self, push_action, target_obj, speed=0.005,push_distance=0.125):
        """
        push_action = pose 4x4
        push_distance is fixed.
        """
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )
        transform = push_action
        # move_distance = 0.0
        # current_poses = []
        # move_poses = []
        # for id in obj_list:
        #     current_pose = self.obj_info(id)
        #     current_poses.append(current_pose)
        obj_is_moved = True
        current_poses = self.obj_info(target_obj)
        # ee link in tip
        ee_tip_transform = np.array([[0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, -self.ee_tip_z_offset],
                                    [0, 0, 0, 1]])

        # transform from tip to ee link
        ee_transform = transform @ ee_tip_transform
        pos = (ee_transform[:3, 3]).T
        pos[2] = max(pos[2] - 0.001, self.bounds[2][0])
        over = np.array((pos[0], pos[1], pos[2] + 0.15))
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()

        #execute push action
        self.close_gripper()
        Rotation_grasp = ee_transform[:3,:3]
        grasp_x = -Rotation_grasp[:3, 2]
        end_pos = pos + push_distance * grasp_x.T 
        end_pos[2] = ee_transform[2, 3]
        success = self.move_joints(self.ik_rest_joints)
        if success:
            success = self.move_ee_pose((over, rot), speed)
            for _ in range(5):
                pb.stepSimulation()
        if success:
            success = self.straight_move(over, pos, rot, speed, detect_force=True)
            for _ in range(10):
                pb.stepSimulation()
        if success:
            success = self.straight_move(pos, end_pos, rot, speed=0.003, detect_force=False)
            for _ in range(10):
                pb.stepSimulation()
        move_pose = self.obj_info(target_obj)
        success =True
        move_position = np.linalg.norm(np.array(move_pose[0]) - np.array(current_poses[0]))
        q1 = current_poses[1] / np.linalg.norm(current_poses[1])
        q2 = move_pose[1] / np.linalg.norm(move_pose[1])
        dot_product = np.dot(q1, q2)
        dot_product = np.clip(np.abs(dot_product), 0.0, 1.0)
        move_angle = 2 * np.arccos(dot_product)
        if (move_position < 0.005) and (move_angle < 0.01):
            obj_is_moved = False
        self.go_home()
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )
        return success, obj_is_moved
    
    def save_state(self):
        self.state_id = pb.saveState()
        return self.state_id
    
    def load_state(self, state_id):
        pb.restoreState(stateId=state_id)