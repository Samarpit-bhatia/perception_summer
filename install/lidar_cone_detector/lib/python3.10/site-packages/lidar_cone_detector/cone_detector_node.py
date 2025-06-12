# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import PointCloud
# from visualization_msgs.msg import Marker, MarkerArray
# from std_msgs.msg import ColorRGBA
# from geometry_msgs.msg import Point

# import numpy as np
# from sklearn.cluster import DBSCAN

# class ConeClassifierNode(Node):
#     def __init__(self):
#         super().__init__('cone_classifier_node')

#         # Parameters
#         self.declare_parameter('input_lidar_topic', '/carmaker/pointcloud')
#         input_topic = self.get_parameter('input_lidar_topic').get_parameter_value().string_value

#         # Subscriptions and publishers
#         self.pointcloud_subscriber = self.create_subscription(PointCloud, input_topic, self.process_lidar_data, 10)
#         self.marker_publisher = self.create_publisher(MarkerArray, 'classified_cones', 10)

#         # Reference frame
#         self.visualization_frame = 'Fr1A'

#         # Configurable parameters
#         self.dbscan_eps = 0.5
#         self.dbscan_min_points = 70
#         self.cluster_min_size = 10
#         self.cluster_max_size = 300
#         self.intensity_jump_threshold = 2.0
#         self.window_fraction = 0.2
#         self.max_lidar_range = 15.0
#         self.cone_height_bounds = (0.0, 0.4)
#         self.marker_ttl_sec = 1

#     def process_lidar_data(self, msg):
#         try:
#             points_xyz = np.array([[pt.x, pt.y, pt.z] for pt in msg.points])

#             # Extract intensity channel
#             lidar_intensity = None
#             for ch in msg.channels:
#                 if ch.name == "intensity":
#                     lidar_intensity = np.array(ch.values)
#                     break

#             if lidar_intensity is None or len(lidar_intensity) != len(points_xyz):
#                 self.get_logger().warn("Intensity channel missing or mismatched.", throttle_duration_sec=1.0)
#                 return

#             # Transform to Fr1A frame
#             points_xyz[:, 0] += 2.921   # x forward shift
#             points_xyz[:, 2] += 0.1629  # z upward shift

#             # Point filtering by height and distance
#             height_mask = (points_xyz[:, 2] > self.cone_height_bounds[0]) & (points_xyz[:, 2] < self.cone_height_bounds[1])
#             distance_mask = (np.linalg.norm(points_xyz[:, :2], axis=1) < self.max_lidar_range)
#             valid_mask = height_mask & distance_mask

#             filtered_points = points_xyz[valid_mask]
#             filtered_intensity = lidar_intensity[valid_mask]

#             if len(filtered_points) == 0:
#                 return

#             # DBSCAN clustering
#             dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_points).fit(filtered_points)
#             labels = dbscan.labels_
#             unique_labels = set(labels)

#             marker_array = MarkerArray()
#             marker_id = 0

#             for cluster_id in unique_labels:
#                 if cluster_id == -1:
#                     continue  # Noise

#                 mask = (labels == cluster_id)
#                 cluster_pts = filtered_points[mask]
#                 cluster_intensities = filtered_intensity[mask]

#                 if len(cluster_pts) < self.cluster_min_size or len(cluster_pts) > self.cluster_max_size:
#                     continue

#                 # Cluster position
#                 cone_center_x = np.mean(cluster_pts[:, 0])
#                 cone_center_y = np.mean(cluster_pts[:, 1])
#                 cone_base_z = np.min(cluster_pts[:, 2])

#                 # Sort by height (z) to analyze vertical profile
#                 height_sorted_indices = np.argsort(cluster_pts[:, 2])
#                 vertical_intensity = cluster_intensities[height_sorted_indices]

#                 num_points = len(vertical_intensity)
#                 window_size = max(3, int(num_points * self.window_fraction))

#                 max_intensity_jump = 0
#                 intensity_jump_pos = num_points // 2
#                 found_jump = False

#                 for i in range(window_size, num_points - window_size):
#                     lower_avg = self.safe_mean(vertical_intensity[i - window_size:i])
#                     upper_avg = self.safe_mean(vertical_intensity[i:i + window_size])
#                     if np.isnan(lower_avg) or np.isnan(upper_avg):
#                         continue

#                     diff = abs(upper_avg - lower_avg)
#                     if diff > max_intensity_jump:
#                         max_intensity_jump = diff
#                         intensity_jump_pos = i
#                         found_jump = True

#                 if not found_jump or max_intensity_jump < self.intensity_jump_threshold:
#                     continue

#                 # Stripe and background analysis
#                 stripe_width = max(1, window_size // 2)
#                 stripe_start = max(0, intensity_jump_pos - stripe_width)
#                 stripe_end = min(num_points, intensity_jump_pos + stripe_width)
#                 stripe_avg = self.safe_mean(vertical_intensity[stripe_start:stripe_end])

#                 bottom_avg = self.safe_mean(vertical_intensity[:intensity_jump_pos - window_size])
#                 top_avg = self.safe_mean(vertical_intensity[intensity_jump_pos + window_size:])

#                 if np.isnan(stripe_avg) or np.isnan(bottom_avg) or np.isnan(top_avg):
#                     continue

#                 background_avg = (bottom_avg + top_avg) / 2
#                 cone_on_left = cone_center_y > 0

#                 # Classify cone
#                 is_blue_cone = (stripe_avg > background_avg + self.intensity_jump_threshold) and cone_on_left
#                 is_yellow_cone = (stripe_avg < background_avg - self.intensity_jump_threshold) and not cone_on_left

#                 if not (is_blue_cone or is_yellow_cone):
#                     continue

#                 # Assign color
#                 cone_color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0) if is_blue_cone else ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)

#                 # RViz marker
#                 marker = Marker()
#                 marker.header.frame_id = self.visualization_frame
#                 marker.header.stamp = self.get_clock().now().to_msg()
#                 marker.id = marker_id
#                 marker.type = Marker.CYLINDER
#                 marker.action = Marker.ADD
#                 marker.pose.position.x = cone_center_x
#                 marker.pose.position.y = cone_center_y
#                 marker.pose.position.z = cone_base_z + 0.155  # Cone midpoint for marker height
#                 marker.pose.orientation.w = 1.0
#                 marker.scale.x = 0.15
#                 marker.scale.y = 0.15
#                 marker.scale.z = 0.45
#                 marker.color = cone_color
#                 marker.lifetime.sec = self.marker_ttl_sec
#                 marker_array.markers.append(marker)
#                 marker_id += 1

#             self.marker_publisher.publish(marker_array)

#         except Exception as e:
#             self.get_logger().error(f"Error during LiDAR processing: {str(e)}", throttle_duration_sec=1.0)

#     def safe_mean(self, arr):
#         return np.mean(arr) if len(arr) > 0 else np.nan

# def main(args=None):
#     rclpy.init(args=args)
#     node = ConeClassifierNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
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