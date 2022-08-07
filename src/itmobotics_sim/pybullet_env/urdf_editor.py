from __future__ import annotations
from typing import Tuple
import xml.etree.ElementTree
import numpy as np

from scipy.spatial.transform import Rotation as R


class URDFEditor:
    def __init__(self, urdf_filename: str):
        self.__urdf_filename = urdf_filename
        # Open original file
        self.__et = xml.etree.ElementTree.parse(self.__urdf_filename)

        self.__xmlJointTemplate = """<?xml version="1.0"?>
        <joint name="%(name)s" type="%(type)s">
          <parent link="%(parent_link)s"/>
          <child link="%(child_link)s"/>
          <origin rpy="%(r)s %(p)s %(yy)s" xyz="%(x)s %(y)s %(z)s"/>
        </joint>
        """
        self.__connections, self.__root_link = self.__parse()
    
    def __parse(self)-> Tuple[dict, str]:
        link_list = []
        root = self.__et.getroot()
        for link in root.iter('link'):
            link_list.append(link.attrib['name'])

        joint_connections = {}
        link_connections = {}
        for joint in root.findall('joint'):
            joint_connection = {}
            parent_link = joint.find('parent').attrib['link']
            child_links = joint.findall('child')
            joint_connection[parent_link] = []
            link_connections[parent_link] = []
            for cl in child_links:
                joint_connection[parent_link].append(cl.attrib['link'])
                link_connections[parent_link].append(cl.attrib['link'])
            joint_connections[joint.attrib['name']] = joint_connection
        # print(link_connections)

        if len(link_connections) == 0:
            last_link = link_list[0]
        else:
            start_link = next(iter(link_connections))
            last_link = start_link
            while True:
                for lc in link_connections.keys():
                    if start_link in link_connections[lc]:
                        start_link = lc
                        break
                if last_link == start_link:
                    break
                last_link = start_link
        return link_connections, last_link

        # get_childs = lambda cl: [get_childs(c) for c in cl.keys()]
    
    @property
    def element_tree(self):
        return self.__et

    @property
    def root_link(self):
        return self.__root_link

    def joinURDF(self, joined_urdf: URDFEditor, connect_link: str, transform: np.array):
        join_et = joined_urdf.element_tree
        join_root_link = joined_urdf.root_link
        
        join_root_xml = join_et.getroot()
        for link in join_root_xml.findall('link'):
            self.__et.getroot().append(link)
        
        for link in join_root_xml.findall('joint'):
            self.__et.getroot().append(link)

        # import xml.etree.ElementTree as ET
        # tree = ET.parse('country_data.xml')
        # root = tree.getroot()
        rot  = R.from_matrix(transform[:3,:3])
        xyz = transform[:3,3]
        rpy = rot.as_euler('xyz')

        data = {'name':join_root_link+'_joint',
            'type': 'fixed',
            'parent_link': connect_link,
            'child_link': join_root_link,
            'r': rpy[0], 'p': rpy[1], 'yy': rpy[2],
            'x': xyz[0], 'y': xyz[1], 'z': xyz[2]
        }

        joint_xml_string = self.__xmlJointTemplate%data
        # print(joint_xml_string)
        new_joint_xml = xml.etree.ElementTree.fromstring(joint_xml_string)
        self.__et.getroot().append(new_joint_xml)
    
    def save(self, filename: str):
        with open(filename, 'wb') as f:
            self.__et.write(f, encoding='utf-8')
        