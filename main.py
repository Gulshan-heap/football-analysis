from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():

    print("\n========== PROGRAM STARTED ==========\n")

    # --------------------------------------------------
    # Read Video
    # --------------------------------------------------
    print("[1] Reading video...")

    video_frames = read_video("input_video/08fd33_4.mp4")

    print("[1] Video reading completed")
    print("[1] Number of frames:", len(video_frames))

    if len(video_frames) == 0:
        print("[ERROR] No frames were loaded from the video.")
        return

    print("[1] First frame shape:", video_frames[0].shape)


    # --------------------------------------------------
    # Initialize Tracker
    # --------------------------------------------------
    print("\n[2] Initializing Tracker...")

    tracker = Tracker('models/best.pt')

    print("[2] Tracker initialized successfully")


    # --------------------------------------------------
    # Get Object Tracks
    # --------------------------------------------------
    print("\n[3] Getting object tracks...")
    print("[3] Reading from stub: stubs/track_stubs.pkl")

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )

    print("[3] Object tracking completed")

    print("[3] Track keys:", tracks.keys())

    if "players" in tracks:
        print("[3] Player frames:", len(tracks["players"]))

    if "ball" in tracks:
        print("[3] Ball frames:", len(tracks["ball"]))


    # --------------------------------------------------
    # Add Object Positions
    # --------------------------------------------------
    print("\n[4] Adding positions to tracks...")

    tracker.add_position_to_tracks(tracks)

    print("[4] Positions added successfully")


    # --------------------------------------------------
    # Camera Movement Estimator
    # --------------------------------------------------
    print("\n[5] Initializing Camera Movement Estimator...")

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])

    print("[5] Camera Movement Estimator initialized")


    print("[6] Calculating camera movement...")

    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )

    print("[6] Camera movement calculation completed")

    print("[6] Camera movement frames:",
          len(camera_movement_per_frame))


    print("[7] Adjusting track positions for camera movement...")

    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks,
        camera_movement_per_frame
    )

    print("[7] Track positions adjusted successfully")


    # --------------------------------------------------
    # View Transformer
    # --------------------------------------------------
    print("\n[8] Initializing View Transformer...")

    view_transformer = ViewTransformer()

    print("[8] View Transformer initialized")


    print("[9] Transforming player positions...")

    view_transformer.add_transformed_position_to_tracks(tracks)

    print("[9] Player positions transformed successfully")


    # --------------------------------------------------
    # Interpolate Ball Positions
    # --------------------------------------------------
    print("\n[10] Interpolating ball positions...")

    tracks["ball"] = tracker.interpolate_ball_positions(
        tracks["ball"]
    )

    print("[10] Ball positions interpolated successfully")


    # --------------------------------------------------
    # Speed and Distance
    # --------------------------------------------------
    print("\n[11] Initializing Speed and Distance Estimator...")

    speed_and_distance_estimator = SpeedAndDistance_Estimator()

    print("[11] Speed and Distance Estimator initialized")


    print("[12] Calculating speed and distance...")

    speed_and_distance_estimator.add_speed_and_distance_to_tracks(
        tracks
    )

    print("[12] Speed and distance calculated successfully")


    # --------------------------------------------------
    # Assign Player Teams
    # --------------------------------------------------
    print("\n[13] Initializing Team Assigner...")

    team_assigner = TeamAssigner()

    print("[13] Team Assigner initialized")


    print("[14] Assigning team colors...")

    team_assigner.assign_team_color(
        video_frames[0],
        tracks['players'][0]
    )

    print("[14] Team colors assigned")

    print("[14] Team colors:", team_assigner.team_colors)


    # --------------------------------------------------
    # Assign Team To Each Player
    # --------------------------------------------------
    print("\n[15] Assigning teams to players...")

    for frame_num, player_track in enumerate(tracks['players']):

        if frame_num % 50 == 0:
            print(
                f"[15] Processing player teams - "
                f"frame {frame_num}/{len(tracks['players'])}"
            )

        for player_id, track in player_track.items():

            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )

            tracks['players'][frame_num][player_id]['team'] = team

            tracks['players'][frame_num][player_id]['team_color'] = (
                team_assigner.team_colors[team]
            )

    print("[15] Player team assignment completed")


    # --------------------------------------------------
    # Assign Ball Acquisition
    # --------------------------------------------------
    print("\n[16] Initializing Player Ball Assigner...")

    player_assigner = PlayerBallAssigner()

    print("[16] Player Ball Assigner initialized")


    print("[17] Assigning ball possession...")

    team_ball_control = []

    for frame_num, player_track in enumerate(tracks['players']):

        if frame_num % 50 == 0:
            print(
                f"[17] Processing ball possession - "
                f"frame {frame_num}/{len(tracks['players'])}"
            )

        ball_bbox = tracks['ball'][frame_num][1]['bbox']

        assigned_player = player_assigner.assign_ball_to_player(
            player_track,
            ball_bbox
        )

        if assigned_player != -1:

            tracks['players'][frame_num][assigned_player]['has_ball'] = True

            team_ball_control.append(
                tracks['players'][frame_num][assigned_player]['team']
            )

        else:

            # Make sure we have a previous value
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(0)

    team_ball_control = np.array(team_ball_control)

    print("[17] Ball possession assignment completed")
    print("[17] Team ball control shape:", team_ball_control.shape)


    # --------------------------------------------------
    # Draw Object Tracks
    # --------------------------------------------------
    print("\n[18] Drawing object annotations...")

    output_video_frames = tracker.draw_annotations(
        video_frames,
        tracks,
        team_ball_control
    )

    print("[18] Object annotations drawn")
    print("[18] Output frames:", len(output_video_frames))


    # --------------------------------------------------
    # Draw Camera Movement
    # --------------------------------------------------
    print("\n[19] Drawing camera movement...")

    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames,
        camera_movement_per_frame
    )

    print("[19] Camera movement annotations drawn")


    # --------------------------------------------------
    # Draw Speed and Distance
    # --------------------------------------------------
    print("\n[20] Drawing speed and distance...")

    speed_and_distance_estimator.draw_speed_and_distance(
        output_video_frames,
        tracks
    )

    print("[20] Speed and distance annotations drawn")


    # --------------------------------------------------
    # Save Video
    # --------------------------------------------------
    print("\n[21] Saving output video...")

    save_video(
        output_video_frames,
        'output_videos/output_video.avi'
    )

    print("[21] Output video saved successfully")


    print("\n========== PROGRAM FINISHED ==========\n")


if __name__ == "__main__":
    main()