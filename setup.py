from setuptools import setup
import os
from glob import glob

package_name = 'f1tenth_gym_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.xacro')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Billy Zheng',
    maintainer_email='billyzheng.bz@gmail.com',
    description='Bridge for using f1tenth_gym in ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gym_bridge = f1tenth_gym_ros.gym_bridge:main',
            'const_vel = f1tenth_gym_ros.const_vel:main',
            'path_publish = f1tenth_gym_ros.path_publish:main',      
            'stan_code3 = f1tenth_gym_ros.stan_code3:main',
            'stan_code_sigma_kappa = f1tenth_gym_ros.stan_code_sigma_kappa:main',
            'stan_code_sigma_steering = f1tenth_gym_ros.stan_code_sigma_steering:main',
            'stan_code4 = f1tenth_gym_ros.stan_code4:main',

        ],
    },
)
