import argparse
import os
import cv2
from utils import read_video, save_video
from trackers import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from analytics import export_frame_data_csv, generate_player_heatmap, team_possession_summary


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24  # fallback if metadata missing
    cap.release()
    return fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default=None, help="Path to output video")
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--use_cache", action="store_true",
                         help="Reuse cached tracking stubs (only for re-runs on the SAME video)")
    args = parser.parse_args()

    video_name = os.path.splitext(os.path.basename(args.input))[0]
    output_path = args.output or f"output_videos/{video_name}_output.avi"
    stub_dir = f"stubs/{video_name}"
    os.makedirs(stub_dir, exist_ok=True)

    fps = get_video_fps(args.input)
    video_frames = read_video(args.input)

    tracker = Tracker(args.model)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=args.use_cache,
        stub_path=f"{stub_dir}/track_stubs.pkl"
    )
    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=args.use_cache,
        stub_path=f"{stub_dir}/camera_movement_stub.pkl"
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    # after pick_corners.py gives you coordinates:
    pixel_vertices = [[110, 1035], [265, 275], [910, 260], [1640, 915]]  # your picked points
    view_transformer = ViewTransformer(pixel_vertices=pixel_vertices, court_width=68, court_length=23.32)

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    speed_and_distance_estimator = SpeedAndDistance_Estimator(fps=fps)
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
    team_ball_control = np.array(team_ball_control)

    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    export_frame_data_csv(tracks, team_ball_control)
    team_possession_summary(team_ball_control)
    generate_player_heatmap(tracks, player_id=7, output_path="output_videos/player_7_heatmap.png")

    save_video(output_video_frames, output_path)
    print(f"Done. Output saved to {output_path}")


if __name__ == '__main__':
    main()