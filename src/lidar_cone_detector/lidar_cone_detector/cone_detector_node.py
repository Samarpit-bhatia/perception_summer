import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker, MarkerArray

from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
import numpy as np


class LidarConeClassifier(Node):
    """
    Node to detect and categorize cones in LiDAR point cloud using DBSCAN and intensity profiles.
    """

    def __init__(self):
        super().__init__('lidar_cone_classifier')

        # Setup subscriber and publisher
        self.create_subscription(PointCloud, '/carmaker/pointcloud', self.handle_pointcloud, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/cones', 10)

        # Internal state
        self.current_marker_id = 0
        self.get_logger().info("LiDAR Cone Classifier Node is active.")

    def handle_pointcloud(self, msg: PointCloud):
        """
        Triggered on reception of new point cloud data.
        """
        if not msg.points:
            self.get_logger().warn("Received empty point cloud.")
            return

        try:
            # Prepare point and intensity arrays
            points = np.array([[p.x, p.y, p.z] for p in msg.points])
            intensities = np.array(msg.channels[0].values)
            if np.max(intensities) > 0:
                intensities /= np.max(intensities)
            enriched_points = np.column_stack((points, intensities))

            # Eliminate ground surface
            filtered_points = self.remove_ground(enriched_points)
            if filtered_points.size == 0:
                return

            # Group points using DBSCAN
            db = DBSCAN(eps=1.0, min_samples=2)
            labels = db.fit_predict(filtered_points[:, :2])

            # Prepare marker array and delete old ones
            markers = MarkerArray()
            delete_all = Marker(action=Marker.DELETEALL)
            markers.markers.append(delete_all)
            self.current_marker_id = 0

            for label in set(labels):
                if label == -1:
                    continue

                cluster = filtered_points[labels == label]
                if cluster.shape[0] < 3:
                    continue

                color_tag = self.detect_color(cluster)
                if color_tag == 'white':
                    continue  # skip uncertain cones

                pos = np.mean(cluster[:, :2], axis=0)
                marker = self.generate_marker(pos, color_tag)
                markers.markers.append(marker)
                self.current_marker_id += 1

            self.marker_pub.publish(markers)

        except Exception as e:
            self.get_logger().error(f"Point cloud processing failed: {e}")

    def remove_ground(self, data: np.ndarray) -> np.ndarray:
        """
        Attempt to filter out ground points using a plane model with RANSAC.
        """
        if data.shape[0] < 10:
            return data

        try:
            xy = data[:, :2]
            z = data[:, 2]
            ransac = RANSACRegressor(residual_threshold=0.02)
            ransac.fit(xy, z)
            return data[~ransac.inlier_mask_]
        except Exception as err:
            self.get_logger().error(f"RANSAC exception: {err}")
            return data

    def detect_color(self, cluster: np.ndarray) -> str:
        """
        Estimate the cone's color by analyzing the vertical distribution of intensity.
        """
        z = cluster[:, 2]
        intensity = cluster[:, 3]

        top = (z >= 0.08)
        mid = (z >= -0.02) & (z < 0.01)
        bottom = (z < -0.09)

        top_mean = np.mean(intensity[top]) if np.any(top) else 0.0
        mid_mean = np.mean(intensity[mid]) if np.any(mid) else 0.0
        bottom_mean = np.mean(intensity[bottom]) if np.any(bottom) else 0.0

        # Classification heuristic
        if 0.0 in (top_mean, mid_mean, bottom_mean):
            return 'white'
        elif (top_mean - mid_mean > 1e-12) or (bottom_mean - mid_mean > 1e-12):
            return 'yellow'
        elif (top_mean - mid_mean < -1e-3) or (bottom_mean - mid_mean < -1e-3):
            return 'blue'
        else:
            return 'white'

    def generate_marker(self, position, color_name) -> Marker:
        """
        Construct a cylinder marker representing a detected cone.
        """
        color_map = {
            'yellow': (1.0, 1.0, 0.0),
            'blue': (0.0, 0.0, 1.0),
            'white': (1.0, 1.0, 1.0)
        }

        marker = Marker()
        marker.header.frame_id = 'Lidar_F'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'cone_markers'
        marker.id = self.current_marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = 0.155
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.31

        r, g, b = color_map[color_name]
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0

        marker.lifetime = Duration(seconds=1.0).to_msg()

        return marker


def main(args=None):
    rclpy.init(args=args)
    node = LidarConeClassifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
