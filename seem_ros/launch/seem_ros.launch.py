from launch import LaunchDescription
from launch_ros.actions import LifecycleNode, Node

def generate_launch_description():
 
    seem_name = "seem_lifecycle_node"
    seem_namespace = 'seem_ros'
    seem_node = Node(
        package='seem_ros',
        executable='seem_lifecycle_node',
        name=seem_name,
        namespace=seem_namespace,
        output='screen',
        emulate_tty=True
    )


    return LaunchDescription([
        seem_node
    ])
