import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/samarpit_bhatia/perception_summer/install/lidar_cone_detector'
