from os import getcwd
from os.path import abspath, dirname, basename, splitext

import urdf_parser_py.urdf as URDF

import pybullet as p
import numpy as np
import pickle

class PyBulletRecorder:
    class LinkTracker:
        def __init__(self, name, body_id, link_id, xyz, rpy, mesh_path, mesh_scale):
            self.body_id = body_id
            self.link_id = link_id
            self.mesh_path = mesh_path
            self.mesh_scale = mesh_scale

            translation = np.array(xyz)
            orientation = p.getQuaternionFromEuler(rpy)

            self.link_pose = [translation, orientation]
            self.name = name

        def transform(self, position, orientation):
            return p.multiplyTransforms(
                position, orientation,
                self.link_pose[0], self.link_pose[1],
            )

        def get_keyframe(self):
            if self.link_id == -1:
                position, orientation = p.getBasePositionAndOrientation(self.body_id)
                position, orientation = self.transform(position=position, orientation=orientation)
            else:
                link_state = p.getLinkState(self.body_id, self.link_id, computeForwardKinematics=True)
                position, orientation = self.transform(position=link_state[4], orientation=link_state[5])

            return {
                'position': list(position),
                'orientation': list(orientation)
            }

    def __init__(self):
        self.states = []
        self.links = []

    def register_object(self, body_id, urdf_path, global_scaling=1):
        link_id_map = dict()
        n_joints = p.getNumJoints(body_id)

        # Crate link name -> id mapping
        baselink_name = p.getBodyInfo(body_id)[0].decode('gb2312')
        link_id_map[baselink_name] = -1
        for link_id in range(0, n_joints):
            link_name = p.getJointInfo(body_id, link_id)[12].decode('gb2312')
            link_id_map[link_name] = link_id

        # get abs path of the urdf file
        dir_path = dirname(abspath(urdf_path))
        file_name = splitext(basename(urdf_path))[0]

        # Read .urdf xml file and build URDF DOM
        robot: URDF.Robot = URDF.Robot.from_xml_file(urdf_path)

        # We go under all links and record only links that have visual mesh
        for link in robot.links:
            link_id = link_id_map[link.name]
            if len(link.visuals) > 0:
                for i, link_visual in enumerate(link.visuals):
                    # TODO: check that scaling is correct if there is scale property
                    ext_scale = 1.0
                    mesh_scale = [global_scaling*ext_scale, global_scaling*ext_scale, global_scaling*ext_scale] if link_visual.geometry.scale is None  else [s * global_scaling * ext_scale for s in link_visual.geometry.scale] 

                    # Get transform TODO: Add normal checking that link does not have origin specification
                    # try:
                    try:
                        rpy = link_visual.origin.rpy
                    except:
                        rpy = [0,0,0]
                    try:
                        xyz = link_visual.origin.xyz
                    except:
                        xyz = [0,0,0]

                    # transform to global abspath
                    mesh_abs_filepath = dir_path + '/' + link_visual.geometry.filename

                    tracker = PyBulletRecorder.LinkTracker(
                        name = file_name + f'_{body_id}_{link.name}_{i}',
                        body_id = body_id,
                        link_id = link_id,
                        xyz = xyz,
                        rpy = rpy,
                        mesh_path = mesh_abs_filepath,
                        mesh_scale=mesh_scale
                    )
                    self.links.append(tracker)


    def add_keyframe(self):
        # Ideally, call every p.stepSimulation()
        current_state = {}
        for link in self.links:
            current_state[link.name] = link.get_keyframe()
        self.states.append(current_state)

    def reset(self):
        self.states = []

    def get_formatted_output(self):
        retval = {}
        for link in self.links:
            out_frames = []
            for state in self.states:
                if link.name in state:
                    out_frames.append(state[link.name] )
            retval[link.name] = {
                'type': 'mesh',
                'mesh_path': link.mesh_path,
                'mesh_scale': link.mesh_scale,
                'frames': out_frames
            }
        return retval

    def save(self, path):
        if path is None:
            print("[Recorder] Path is None.. not saving")
        else:
            print("[Recorder] Saving state to {}".format(path))
            # print(self.get_formatted_output())
            pickle.dump(self.get_formatted_output(), open(path, 'wb'))

