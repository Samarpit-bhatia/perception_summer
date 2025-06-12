import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sklearn.cluster import DBSCAN
import numpy as np
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker, MarkerArray

class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector')
        
        # Declare parameters with default values
        self.declare_parameters(
            namespace='',
            parameters=[
                ('dbscan_eps', 2.0),
                ('dbscan_min_samples', 2),
                ('z_filter_min', -0.16),
                ('top_intensity_z_range', [0.1, 0.14]),
                ('mid_intensity_z_range', [-0.01, 0.0]),
                ('intensity_diff_threshold', 1e-10),
                ('marker_lifetime', 0.1),
                ('cone_height', 0.31),
                ('cone_radius', 0.2),
                ('yellow_color', [1.0, 1.0, 0.0, 1.0]),
                ('blue_color', [0.0, 0.0, 1.0, 1.0])
            ]
        )

        # Configure QoS for point cloud (typically BEST_EFFORT for sensors)
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )

        # Setup subscribers and publishers
        self.subscription = self.create_subscription(
            PointCloud,
            '/carmaker/pointcloud',
            self.pointcloud_callback,
            qos_profile
        )
        
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/detected_cones',
            10
        )
        
        self.marker_id = 0
        self.get_logger().info('Cone detector node initialized')

    def pointcloud_callback(self, msg):
        """Process incoming point cloud and detect cones"""
        if not msg.points or not msg.channels or len(msg.channels[0].values) != len(msg.points):
            self.get_logger().warn('Invalid point cloud received')
            return

        try:
            # Convert to numpy array and normalize intensity
            points = np.array([[p.x, p.y, p.z] for p in msg.points])
            intensities = np.array(msg.channels[0].values)
            intensities = intensities / np.max(intensities)  # Normalize
            
            # Filter by height and prepare data
            filtered_idx = points[:, 2] > self.get_parameter('z_filter_min').value
            points = points[filtered_idx]
            intensities = intensities[filtered_idx]
            
            if points.size == 0:
                return

            # Cluster points in XY plane
            clustering = DBSCAN(
                eps=self.get_parameter('dbscan_eps').value,
                min_samples=self.get_parameter('dbscan_min_samples').value
            ).fit(points[:, :2])
            
            # Create marker array
            markers = MarkerArray()
            self._add_delete_all_markers(markers)
            self.marker_id = 0

            # Process each cluster
            for cluster_id in np.unique(clustering.labels_):
                if cluster_id == -1:  # Skip noise
                    continue
                    
                self._process_cluster(
                    markers,
                    points[clustering.labels_ == cluster_id],
                    intensities[clustering.labels_ == cluster_id]
                )

            self.marker_pub.publish(markers)

        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {str(e)}')

    def _process_cluster(self, markers, cluster_points, cluster_intensities):
        """Process a single cluster and add appropriate marker"""
        # Get parameters
        top_range = self.get_parameter('top_intensity_z_range').value
        mid_range = self.get_parameter('mid_intensity_z_range').value
        diff_thresh = self.get_parameter('intensity_diff_threshold').value
        
        # Calculate centroid
        centroid = np.mean(cluster_points[:, :2], axis=0)
        
        # Analyze intensity in different height regions
        top_mask = (cluster_points[:, 2] > top_range[0]) & (cluster_points[:, 2] < top_range[1])
        mid_mask = (cluster_points[:, 2] > mid_range[0]) & (cluster_points[:, 2] < mid_range[1])
        
        if not np.any(top_mask) or not np.any(mid_mask):
            return
            
        top_int = np.max(cluster_intensities[top_mask])
        mid_int = np.mean(cluster_intensities[mid_mask])
        
        # Determine cone color
        if (top_int - mid_int) > diff_thresh:
            color = self.get_parameter('yellow_color').value
            ns = "yellow_cones"
        else:
            color = self.get_parameter('blue_color').value
            ns = "blue_cones"
        
        # Create marker
        marker = Marker()
        marker.header.frame_id = "Lidar_F"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = self.marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(centroid[0])
        marker.pose.position.y = float(centroid[1])
        marker.pose.position.z = self.get_parameter('cone_height').value / 2  # Center of cylinder
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.get_parameter('cone_radius').value * 2
        marker.scale.y = self.get_parameter('cone_radius').value * 2
        marker.scale.z = self.get_parameter('cone_height').value
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.lifetime = Duration(seconds=self.get_parameter('marker_lifetime').value).to_msg()
        
        markers.markers.append(marker)
        self.marker_id += 1

    def _add_delete_all_markers(self, marker_array):
        """Add a marker to clear all previous markers"""
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
