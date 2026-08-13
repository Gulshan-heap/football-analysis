import numpy as np
import cv2


class ViewTransformer:
    def __init__(self, pixel_vertices=None, court_width=68, court_length=23.32):
        """
        pixel_vertices: list of 4 (x, y) pixel points on the source frame,
                         in order TL, TR, BR, BL, marking the region you're
                         calibrating (often just the visible penalty-area
                         strip, not full 105m pitch, since only that much
                         is usually visible from one camera angle).
        court_width / court_length: real-world meters for that same region.
        """
        self.court_width = court_width
        self.court_length = court_length

        if pixel_vertices is None:
            # fallback: original hardcoded video's calibration
            pixel_vertices = [[110, 1035], [265, 275], [910, 260], [1640, 915]]

        self.pixel_vertices = np.array(pixel_vertices, dtype=np.float32)

        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ], dtype=np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped_point = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    transformed_position = self.transform_point(position)
                    if transformed_position is not None:
                        transformed_position = transformed_position.squeeze().tolist()
                    tracks[object_name][frame_num][track_id]['position_transformed'] = transformed_position