import pyrealsense2 as rs
import cv2
import numpy as np

class setup_cam:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        self.pc = rs.pointcloud()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        depth_sensor = device.first_depth_sensor()
        depth_sensor.set_option(rs.option.min_distance, 0)

        device_product_line = str(device.get_info(rs.camera_info.product_line))
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            # config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.data = self._frame_depth_from_video()

    def _frame_depth_from_video(self):
        while True:
            frames = self.pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_map = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            points = self.pc.calculate(aligned_depth_frame)
            v = points.get_vertices()
            point_cloud = np.asanyarray(v).view(np.float32).reshape(depth_map.shape[0], depth_map.shape[1], 3) 

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)

            yield [color_image, depth_map, point_cloud]

    def single_data(self):
        while True:
            frames = self.pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_map = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            points = self.pc.calculate(aligned_depth_frame)
            v = points.get_vertices()
            point_cloud = np.asanyarray(v).view(np.float32).reshape(depth_map.shape[0], depth_map.shape[1], 3) 

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            break
        return color_image, depth_map, point_cloud