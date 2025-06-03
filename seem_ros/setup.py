from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'seem_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['seem_ros', 'seem_ros.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'configs', 'seem'),
         glob('seem_ros/configs/seem/*.yaml')),
        (os.path.join('share', package_name), glob('seem_ros/*.pt')),
        (os.path.join('share', package_name, 'srv'), glob('seem_ros/srv/*.srv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='kevin-eppacher@hotmail.de',
    description='TODO: Package description', 
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'seem_lifecycle_node = seem_ros.seem_lifecycle_node:main',
            'seem_web_browser_node = seem_ros.demo.seem.app:main',
            'test_service = seem_ros.test.test_service:main',
        ],
    },
)
