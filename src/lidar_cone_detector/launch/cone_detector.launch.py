from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lidar_cone_detector',
            executable='cone_detector',
            name='cone_detector',
            parameters=[
                {'dbscan_eps': 0.25},
                {'min_samples': 5}
            ],
            output='screen'
        )
    ])
