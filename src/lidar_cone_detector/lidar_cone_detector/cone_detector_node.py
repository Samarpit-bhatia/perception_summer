
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from sklearn.cluster import DBSCAN
import numpy as np
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker, MarkerArray


class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector')

        # Parameters (can also be declared and set via ROS2 command-line or launch file)
        self.declare_parameter('dbscan_eps', 2.0)
        self.declare_parameter('dbscan_min_samples', 2)
        self.declare_parameter('z_filter_min', -0.16)
        self.declare_parameter('top_intensity_range', [0.1, 0.14])
        self.declare_parameter('mid_intensity_range', [-0.01, 0.0])
        self.declare_parameter('intensity_diff_thresh', 1e-10)
        self.declare_parameter('marker_lifetime', 0.1)

        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        self.subscription = self.create_subscription(
            PointCloud,
            '/carmaker/pointcloud',
            self.pointcloud_callback,
            qos
        )
        self.publisher = self.create_publisher(MarkerArray, '/detected_cones', qos)
        self.marker_id = 0

        self.get_logger().info('ConeDetector node initialized.')

    def pointcloud_callback(self, msg: PointCloud) -> None:
        """Callback: cluster the incoming point cloud and publish cone markers."""
        if not msg.points or not msg.channels:
            self.get_logger().warning('Empty point cloud or missing channels.')
            return

        # Extract parameters
        eps = self.get_parameter('dbscan_eps').value
        min_samples = self.get_parameter('dbscan_min_samples').value
        z_min = self.get_parameter('z_filter_min').value
        top_range = tuple(self.get_parameter('top_intensity_range').value)
        mid_range = tuple(self.get_parameter('mid_intensity_range').value)
        diff_thresh = self.get_parameter('intensity_diff_thresh').value
        lifetime = self.get_parameter('marker_lifetime').value

        # Build Nx4 array: x, y, z, intensity
        points = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=float)
        intensities = np.array(msg.channels[0].values, dtype=float)
        if intensities.size != points.shape[0]:
            self.get_logger().error('Intensity array length mismatch.')
            return
        pc = np.hstack((points, intensities.reshape(-1, 1)))
        
        # Filter by height
        pc = pc[pc[:, 2] > z_min]
        if pc.size == 0:
            return

        # Cluster in XY plane
        xy = pc[:, :2]
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xy)
        labels = clustering.labels_

        # Prepare MarkerArray
        markers = MarkerArray()
        # Clear previous markers
        del_all = Marker()
        del_all.action = Marker.DELETEALL
        markers.markers.append(del_all)
        self.marker_id = 0

        # Process each cluster
        for lbl in set(labels):
            if lbl == -1:
                continue   # noise

            cluster_pts = pc[labels == lbl]
            centroid = np.mean(cluster_pts[:, :2], axis=0)

            # Separate by z-level intensity windows
            top_int = cluster_pts[
                (cluster_pts[:, 2] >= top_range[0]) & (cluster_pts[:, 2] <= top_range[1])
            ][:, 3]
            mid_int = cluster_pts[
                (cluster_pts[:, 2] >= mid_range[0]) & (cluster_pts[:, 2] <= mid_range[1])
            ][:, 3]

            if top_int.size == 0 or mid_int.size == 0:
                continue

            top_val = top_int.max()
            mid_val = mid_int.mean()

            color = 'yellow' if (top_val - mid_val) > diff_thresh else 'blue'
            marker = self._make_marker(centroid, color, lifetime)
            markers.markers.append(marker)
            self.marker_id += 1

        # Publish
        self.publisher.publish(markers)

    def _make_marker(self, centroid: np.ndarray, color: str, lifetime: float) -> Marker:
        """Helper to create a cylinder marker at given XY, with specified color."""
        r, g, b = (1.0, 1.0, 0.0) if color == 'yellow' else (0.0, 0.0, 1.0)
        marker = Marker()
        marker.header.frame_id = 'Lidar_F'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"{color}_cones"
        marker.id = self.marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(centroid[0])
        marker.pose.position.y = float(centroid[1])
        marker.pose.position.z = 0.155
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.31
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        marker.lifetime = Duration(seconds=lifetime).to_msg()
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = ConeDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
