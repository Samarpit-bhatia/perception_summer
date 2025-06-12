import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sklearn.cluster import DBSCAN
import numpy as np
from sensor_msgs.msg import PointCloud
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from typing import List

class ConeDetector(Node):
    """
    A ROS2 node that detects cones from point cloud data and visualizes them as markers.
    
    This node subscribes to point cloud data, clusters the points using DBSCAN,
    classifies clusters as either yellow or blue cones based on intensity patterns,
    and publishes visualization markers for RViz.
    """
    
    def __init__(self):
        super().__init__('cone_detector')
        
        # Parameters (could be made configurable via ROS parameters)
        self._min_cluster_size = 2
        self._cluster_eps = 2.0  # meters
        self._min_height = -0.16  # meters (filter points below this)
        self._top_cone_height_range = (0.10, 0.14)  # meters
        self._mid_cone_height_range = (-0.01, 0.0)  # meters
        self._intensity_diff_threshold = 1e-10
        self._cone_radius = 0.2  # meters (for visualization)
        self._cone_height = 0.31  # meters (for visualization)
        self._cone_visualization_z = 0.155  # meters (half height)
        self._marker_lifetime = Duration(seconds=0.1)
        
        # Subscriber and publisher
        self._pointcloud_sub = self.create_subscription(
            PointCloud, 
            '/carmaker/pointcloud', 
            self._process_pointcloud, 
            10
        )
        self._marker_pub = self.create_publisher(
            MarkerArray, 
            '/cones', 
            10
        )
        
        self._marker_id_counter = 0

    def _process_pointcloud(self, msg: PointCloud) -> None:
        """
        Process incoming point cloud message to detect and visualize cones.
        
        Args:
            msg: The incoming PointCloud message
        """
        if not msg.points:
            self.get_logger().debug("Received empty point cloud")
            return
            
        try:
            # Convert point cloud to numpy array
            points = self._convert_pointcloud_to_array(msg)
            
            # Filter points by height
            filtered_points = self._filter_by_height(points)
            if filtered_points.size == 0:
                return
                
            # Cluster points
            clusters = self._cluster_points(filtered_points)
            if not clusters:
                return
                
            # Create visualization markers
            marker_array = self._create_markers(clusters)
            self._marker_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {str(e)}")

    def _convert_pointcloud_to_array(self, msg: PointCloud) -> np.ndarray:
        """
        Convert PointCloud message to numpy array with x,y,z,intensity.
        
        Args:
            msg: PointCloud message
            
        Returns:
            Numpy array of shape (n_points, 4) containing [x, y, z, intensity]
        """
        # Extract points and intensity (assuming intensity is in first channel)
        points = np.array([[p.x, p.y, p.z] for p in msg.points])
        intensities = np.array(msg.channels[0].values)
        
        # Normalize intensity (0-1 range)
        if intensities.max() > 0:
            intensities = intensities / intensities.max()
            
        return np.column_stack((points, intensities))

    def _filter_by_height(self, points: np.ndarray) -> np.ndarray:
        """
        Filter points above a minimum height threshold.
        
        Args:
            points: Numpy array of points (x,y,z,intensity)
            
        Returns:
            Filtered array of points
        """
        return points[points[:, 2] > self._min_height]

    def _cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Cluster points using DBSCAN algorithm.
        
        Args:
            points: Numpy array of points (x,y,z,intensity)
            
        Returns:
            List of clusters (each cluster is a numpy array)
        """
        # Use only x,y coordinates for clustering
        xy_coords = points[:, :2]
        
        dbscan = DBSCAN(
            eps=self._cluster_eps, 
            min_samples=self._min_cluster_size
        ).fit(xy_coords)
        
        labels = dbscan.labels_
        unique_labels = set(labels)
        
        clusters = []
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            clusters.append(points[labels == label])
            
        return clusters

    def _create_markers(self, clusters: List[np.ndarray]) -> MarkerArray:
        """
        Create visualization markers for detected cones.
        
        Args:
            clusters: List of point clusters
            
        Returns:
            MarkerArray containing visualization markers
        """
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        for cluster in clusters:
            marker = self._create_cone_marker(cluster)
            if marker:
                marker_array.markers.append(marker)
                self._marker_id_counter += 1
                
        return marker_array

    def _create_cone_marker(self, cluster: np.ndarray) -> Marker:
        """
        Create a single cone marker from a cluster of points.
        
        Args:
            cluster: Numpy array of points in the cluster
            
        Returns:
            Marker object or None if cluster doesn't meet criteria
        """
        # Calculate cluster centroid
        centroid = np.mean(cluster[:, :2], axis=0)
        
        # Extract points from different height regions
        top_points = cluster[
            (cluster[:, 2] > self._top_cone_height_range[0]) & 
            (cluster[:, 2] < self._top_cone_height_range[1])
        ]
        
        mid_points = cluster[
            (cluster[:, 2] > self._mid_cone_height_range[0]) & 
            (cluster[:, 2] < self._mid_cone_height_range[1])
        ]
        
        # Skip if we don't have points in both regions
        if top_points.size == 0 or mid_points.size == 0:
            return None
            
        # Calculate intensity characteristics
        top_intensity = np.max(top_points[:, 3])
        mid_intensity = np.mean(mid_points[:, 3])
        
        # Determine cone color based on intensity pattern
        if (top_intensity - mid_intensity) > self._intensity_diff_threshold:
            return self._create_marker(
                position=centroid,
                ns="yellow_cones",
                color=(1.0, 1.0, 0.0)  # Yellow
            )
        else:
            return self._create_marker(
                position=centroid,
                ns="blue_cones",
                color=(0.0, 0.0, 1.0)  # Blue
            )

    def _create_marker(
        self, 
        position: np.ndarray, 
        ns: str, 
        color: tuple
    ) -> Marker:
        """
        Create a visualization marker for a cone.
        
        Args:
            position: (x,y) position of the cone
            ns: Namespace for the marker
            color: (r,g,b) tuple for marker color
            
        Returns:
            Configured Marker object
        """
        marker = Marker()
        marker.header.frame_id = "Lidar_F"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = self._marker_id_counter
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Set position and orientation
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = self._cone_visualization_z
        marker.pose.orientation.w = 1.0  # Identity quaternion
        
        # Set scale (size)
        marker.scale.x = self._cone_radius
        marker.scale.y = self._cone_radius
        marker.scale.z = self._cone_height
        
        # Set color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0  # Fully opaque
        
        # Set lifetime
        marker.lifetime = self._marker_lifetime.to_msg()
        
        return marker

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
