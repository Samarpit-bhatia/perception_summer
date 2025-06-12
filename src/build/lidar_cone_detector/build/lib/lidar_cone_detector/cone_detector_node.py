# File: lidar_cone_detector/src/cone_detector_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from sklearn.cluster import DBSCAN

class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector_node')

        self.subscription = self.create_subscription(
            PointCloud2,
            '/carmaker/points',
            self.pointcloud_callback,
            10
        )

        self.marker_pub = self.create_publisher(MarkerArray, '/detected_cones', 10)
        self.declare_parameter('dbscan_eps', 0.25)
        self.declare_parameter('min_samples', 5)

        self.dbscan_eps = self.get_parameter('dbscan_eps').value
        self.min_samples = self.get_parameter('min_samples').value

    def pointcloud_callback(self, msg):
        points = np.array([
            [p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        ])

        # Filter out ground points
        points = points[points[:, 2] > 0.05]  # LiDAR is mounted at 0.1629 m

        if points.shape[0] == 0:
            return

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.min_samples).fit(points[:, :2])
        labels = clustering.labels_
        unique_labels = set(labels)

        marker_array = MarkerArray()
        marker_id = 0
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = points[labels == label]
            centroid = np.mean(cluster_points, axis=0)

            # Classify based on y-coordinate
            color = 'blue' if centroid[1] > 0 else 'yellow'

            # Transform to Fr1A frame
            centroid[0] -= 2.921  # LiDAR offset from rear
            centroid[2] -= 0.1629  # LiDAR height

            # Create marker
            marker = Marker()
            marker.header.frame_id = 'Fr1A'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(centroid[0])
            marker.pose.position.y = float(centroid[1])
            marker.pose.position.z = float(centroid[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.31

            if color == 'blue':
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime.sec = 0
            marker_array.markers.append(marker)
            marker_id += 1

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = ConeDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
