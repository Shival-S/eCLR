#!/usr/bin/env python
"""
CARLA UE5 Data Collection for UniLCD (Sidewalk Robot)

Collects RGB/segmentation images + action data from pedestrian POV.

Output format:
  - rgb_up/, rgb_down/, seg_up/, seg_down/: Image folders (.png)
  - actions/: Action data (.npy) with format:
      [steering, speed, throttle, cur_x, cur_y, next_x, next_y]
    Where:
      - steering: [-1, 1] direction to target (-1=left, 0=straight, 1=right)
      - speed: m/s velocity magnitude
      - throttle: [0, 1] normalized speed
      - cur_x, cur_y: current position
      - next_x, next_y: target waypoint position
"""

import glob
import os
import sys
import argparse
import random
import queue
import numpy as np
from datetime import datetime
import math 
import collections 

try:
    import cv2
except ImportError:
    raise RuntimeError('cannot import cv2, make sure opencv-python package is installed')

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

class CarlaSyncMode(object):
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        
    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False, 
            synchronous_mode=True, 
            fixed_delta_seconds=self.delta_seconds))
        
        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)
            
        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self
        
    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data
        
    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)
        for sensor in self.sensors:
            if sensor and sensor.is_alive:
                sensor.destroy()
                
    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

def compute_steering(player, target_location):
    """
    Compute normalized steering value based on direction to target.
    Returns value in [-1, 1] where:
      -1 = hard left, 0 = straight, 1 = hard right
    """
    # Get player's current transform and forward vector
    transform = player.get_transform()
    forward = transform.get_forward_vector()
    location = transform.location

    # Direction to target
    direction = carla.Vector3D(
        target_location.x - location.x,
        target_location.y - location.y,
        0.0
    )

    # Normalize direction
    dir_length = math.sqrt(direction.x**2 + direction.y**2)
    if dir_length < 0.001:
        return 0.0
    direction.x /= dir_length
    direction.y /= dir_length

    # Cross product (z component) gives signed angle
    # Positive = target is to the right, Negative = target is to the left
    cross_z = forward.x * direction.y - forward.y * direction.x

    # Dot product for forward/backward check
    dot = forward.x * direction.x + forward.y * direction.y

    # Compute steering: use atan2 for proper angle, normalize to [-1, 1]
    angle = math.atan2(cross_z, dot)  # Range: [-pi, pi]
    steering = np.clip(angle / math.pi, -1.0, 1.0)

    return steering


def process_bev_image(img_bev, trajectory_points, current_location, bev_height_m, bev_size_px, fov_deg):
    bev_array = np.frombuffer(img_bev.raw_data, dtype=np.uint8)
    bev_array = np.reshape(bev_array, (img_bev.height, img_bev.width, 4))
    bev_image = bev_array[:, :, :3].copy()
    
    fov_rad = math.radians(fov_deg)
    view_width_m = 2 * bev_height_m * math.tan(fov_rad / 2)
    meters_per_pixel = view_width_m / bev_size_px
    
    center_pixel_x = bev_size_px / 2
    center_pixel_y = bev_size_px / 2
    
    pixel_points = []
    for point in trajectory_points:
        delta_loc = point - current_location
        pixel_x = center_pixel_x + (delta_loc.y / meters_per_pixel)
        pixel_y = center_pixel_y - (delta_loc.x / meters_per_pixel)
        pixel_points.append((int(pixel_x), int(pixel_y)))
    
    for i in range(1, len(pixel_points)):
        cv2.line(bev_image, pixel_points[i-1], pixel_points[i], (0, 0, 255), 2)
    
    return bev_image

def run_simulation(args):
    actor_list = []
    client = None
    sensors_to_sync = []
    cam_bev = None
    
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        main_output_folder = f"_output_{timestamp}"
        os.makedirs(main_output_folder, exist_ok=True)
        print(f"Main output folder created at: '{main_output_folder}/'")

        output_paths = [] # Will store the 1 or 4 output folder paths

        # Create actions folder for .npy files (used by all modes)
        actions_folder = os.path.join(main_output_folder, 'actions')
        os.makedirs(actions_folder, exist_ok=True)
        print(f"Saving action data (.npy) to: '{actions_folder}/'")

        if args.mode == 'custom' or args.mode == 'old':
            mode_folder = main_output_folder#os.path.join(main_output_folder, f'{args.mode}_waist')
            print(f"Saving '{args.mode}' data to: '{mode_folder}/'")

            folder_names = ['rgb_up', 'rgb_down', 'seg_up', 'seg_down']
            for name in folder_names:
                path = os.path.join(mode_folder, name)
                os.makedirs(path)
                output_paths.append(path) # Add the 4 subfolder paths
        
        elif args.mode == 'bev':
            path = os.path.join(main_output_folder, 'bev')
            os.makedirs(path)
            output_paths.append(path) # Add the single BEV path
            print(f"Saving '{args.mode}' data to: '{path}/'")

        client = carla.Client(args.host, args.port)
        client.set_timeout(15.0)
        sim_world = client.get_world()
        original_settings = sim_world.get_settings()
        blueprint_library = sim_world.get_blueprint_library()
        
        walker_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
        walker_bp.set_attribute('is_invincible', 'true')
        spawn_points = sim_world.get_map().get_spawn_points()
        if not spawn_points: raise RuntimeError("Map has no spawn points!")
        spawn_point = random.choice(spawn_points)
        player = sim_world.spawn_actor(walker_bp, spawn_point)
        actor_list.append(player)
        print(f"Spawned main player '{player.type_id}' with ID {player.id}.")
        
        player_controller_bp = blueprint_library.find('controller.ai.walker')
        player_controller = sim_world.spawn_actor(player_controller_bp, carla.Transform(), attach_to=player)
        actor_list.append(player_controller)
        player_controller.start()
        player_controller.set_max_speed(1.4) 
        target_location = sim_world.get_random_location_from_navigation()
        player_controller.go_to_location(target_location)

        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        spawn_points = sim_world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        for i in range(min(args.cars, len(spawn_points))):
            vehicle_bp = random.choice(vehicle_blueprints)
            spawn_point = spawn_points.pop(0) 
            if spawn_point.location.distance(player.get_location()) < 10.0:
                spawn_point = spawn_points.pop()
            vehicle = sim_world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)
                actor_list.append(vehicle)
        print(f"Spawned {len([x for x in actor_list if 'vehicle' in x.type_id])} vehicles on autopilot.")

        walker_blueprints = blueprint_library.filter('walker.pedestrian.*')
        controller_bp = blueprint_library.find('controller.ai.walker')
        spawned_peds = 0
        for i in range(args.peds):
            spawn_location = sim_world.get_random_location_from_navigation()
            if spawn_location:
                if spawn_location.distance(player.get_location()) < 10.0:
                    continue
                walker_bp = random.choice(walker_blueprints)
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'true')
                ped = sim_world.try_spawn_actor(walker_bp, carla.Transform(spawn_location))
                if ped:
                    controller = sim_world.spawn_actor(controller_bp, carla.Transform(), attach_to=ped)
                    actor_list.extend([ped, controller])
                    controller.start()
                    controller.go_to_location(sim_world.get_random_location_from_navigation())
                    controller.set_max_speed(1.0 + random.random())
                    spawned_peds += 1
        print(f"Spawned {spawned_peds} wandering pedestrians.")

        waist_location = carla.Location(x=0.2, z=0.7)
        transform_up = carla.Transform(waist_location, carla.Rotation(pitch=7.0))
        transform_down = carla.Transform(waist_location, carla.Rotation(pitch=-7.0))
        trajectory_points = None

        if args.mode == 'custom':
            fx = 231.542587
            fy = 231.516769
            cx = 214.011078
            cy = 120.209908

            # --- 2. Calculate the ORIGINAL resolution ---
            ORIG_WIDTH = int(round(cx * 2))  # 428
            ORIG_HEIGHT = int(round(cy * 2)) # 240
            
            # --- 3. Calculate the TRUE Field of View using ORIGINAL parameters ---
            # This FOV is the physical truth of the lens.
            fov_y = 2 * math.degrees(math.atan( ORIG_HEIGHT / (2 * fy) ))
            
            # --- 4. Set your NEW desired resolution ---
            # This is the "downsampled" x2 resolution you wanted.
            CUSTOM_VIEW_WIDTH = 848
            CUSTOM_VIEW_HEIGHT = 480
            
            cam_bp_rgb = blueprint_library.find('sensor.camera.rgb')
            cam_bp_seg = blueprint_library.find('sensor.camera.semantic_segmentation')
            for bp in [cam_bp_rgb, cam_bp_seg]:
                bp.set_attribute('image_size_x', str(CUSTOM_VIEW_WIDTH))
                bp.set_attribute('image_size_y', str(CUSTOM_VIEW_HEIGHT))
                bp.set_attribute('fov', str(fov_y))
            cam_rgb_up = sim_world.spawn_actor(cam_bp_rgb, transform_up, attach_to=player)
            cam_rgb_down = sim_world.spawn_actor(cam_bp_rgb, transform_down, attach_to=player)
            cam_seg_up = sim_world.spawn_actor(cam_bp_seg, transform_up, attach_to=player)
            cam_seg_down = sim_world.spawn_actor(cam_bp_seg, transform_down, attach_to=player)
            sensors_to_sync = [cam_rgb_up, cam_rgb_down, cam_seg_up, cam_seg_down]

        elif args.mode == 'old':
            OLD_VIEW_WIDTH, OLD_VIEW_HEIGHT = args.width // 2, args.height // 2
            cam_bp_rgb = blueprint_library.find('sensor.camera.rgb')
            cam_bp_rgb.set_attribute('image_size_x', str(OLD_VIEW_WIDTH))
            cam_bp_rgb.set_attribute('image_size_y', str(OLD_VIEW_HEIGHT))
            cam_bp_seg = blueprint_library.find('sensor.camera.semantic_segmentation')
            cam_bp_seg.set_attribute('image_size_x', str(OLD_VIEW_WIDTH))
            cam_bp_seg.set_attribute('image_size_y', str(OLD_VIEW_HEIGHT))
            cam_rgb_up = sim_world.spawn_actor(cam_bp_rgb, transform_up, attach_to=player)
            cam_rgb_down = sim_world.spawn_actor(cam_bp_rgb, transform_down, attach_to=player)
            cam_seg_up = sim_world.spawn_actor(cam_bp_seg, transform_up, attach_to=player)
            cam_seg_down = sim_world.spawn_actor(cam_bp_seg, transform_down, attach_to=player)
            sensors_to_sync = [cam_rgb_up, cam_rgb_down, cam_seg_up, cam_seg_down]

        elif args.mode == 'bev':
            trajectory_points = collections.deque(maxlen=args.trajectory_length)
            cam_bp_bev = blueprint_library.find('sensor.camera.rgb')
            cam_bp_bev.set_attribute('image_size_x', str(args.bev_size))
            cam_bp_bev.set_attribute('image_size_y', str(args.bev_size))
            cam_bp_bev.set_attribute('fov', '90') 
            transform_bev = carla.Transform(carla.Location(z=args.bev_height), carla.Rotation(pitch=-90))
            cam_bev = sim_world.spawn_actor(cam_bp_bev, transform_bev, attach_to=player)
            sensors_to_sync = [cam_bev]

        actor_list.extend(sensors_to_sync)
        
        with CarlaSyncMode(sim_world, *sensors_to_sync, fps=args.fps) as sync_mode:
            
            saved_frame_count = 0
            last_goal_frame = 0
            
            for frame_count in range(args.frames):
                try:
                    current_location = player.get_location()
                    
                    if current_location.distance(target_location) < 2.0:
                        sys.stdout.write(f'\r[Frame {frame_count}] Arrived at destination. Getting new one...')
                        target_location = sim_world.get_random_location_from_navigation()
                        player_controller.go_to_location(target_location)
                        last_goal_frame = frame_count
                    
                    elif (frame_count - last_goal_frame) > (args.fps * 5):
                        if player.get_velocity().length() < 0.1:
                            sys.stdout.write(f'\r[Frame {frame_count}] Player is STUCK. Getting new destination...')
                            target_location = sim_world.get_random_location_from_navigation()
                            player_controller.go_to_location(target_location)
                            last_goal_frame = frame_count

                    data = sync_mode.tick(timeout=5.0)
                    
                    if args.mode == 'bev':
                        trajectory_points.append(current_location)

                    if frame_count % args.save_every == 0:
                        sensor_data = data[1:]

                        # === Compute and save action data (.npy) ===
                        # Get velocity for speed calculation
                        velocity = player.get_velocity()
                        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

                        # Skip frames where player is stationary (no useful training data)
                        if speed < 0.05:
                            continue

                        # Compute steering (direction to target)
                        steering = compute_steering(player, target_location)

                        # Throttle: 1.0 if moving at decent speed, scaled otherwise
                        max_speed = 1.4  # matches player_controller.set_max_speed()
                        throttle = min(speed / max_speed, 1.0)

                        # Locations: current and next waypoint (target)
                        cur_x, cur_y = current_location.x, current_location.y
                        next_x, next_y = target_location.x, target_location.y

                        # Save action data: [steering, speed, throttle, cur_x, cur_y, next_x, next_y]
                        action_data = np.array([
                            steering, speed, throttle,
                            cur_x, cur_y, next_x, next_y
                        ], dtype=np.float32)
                        npy_filename = os.path.join(actions_folder, f'{saved_frame_count:06d}.npy')
                        np.save(npy_filename, action_data)

                        # === Save images ===
                        if args.mode == 'custom' or args.mode == 'old':
                            # Convert segmentation images
                            sensor_data[2].convert(carla.ColorConverter.CityScapesPalette)
                            sensor_data[3].convert(carla.ColorConverter.CityScapesPalette)

                            # Loop and save each image individually
                            for i in range(4):
                                img = sensor_data[i]
                                save_folder = output_paths[i]

                                array = np.frombuffer(img.raw_data, dtype=np.uint8)
                                array = np.reshape(array, (img.height, img.width, 4))
                                bgr_array = array[:, :, :3]

                                filename = os.path.join(save_folder, f'{saved_frame_count:06d}.png')
                                cv2.imwrite(filename, bgr_array)

                        elif args.mode == 'bev':
                            img_bev = sensor_data[0]
                            bev_image = process_bev_image(
                                img_bev,
                                trajectory_points,
                                current_location,
                                args.bev_height,
                                args.bev_size,
                                float(cam_bev.attributes['fov'])
                            )
                            # Save to the single BEV path
                            cv2.imwrite(os.path.join(output_paths[0], f'{saved_frame_count:06d}.png'), bev_image)

                        saved_frame_count += 1

                        status_msg = f'Saved {args.mode} frame {saved_frame_count} + action .npy'
                        if args.mode != 'bev':
                            status_msg = f'Saved 4 {args.mode} images + action .npy for frame {saved_frame_count}'
                        
                        sys.stdout.write(f'\rSim frame {frame_count + 1}/{args.frames} | {status_msg}')
                        sys.stdout.flush()
                    
                except (RuntimeError, TimeoutError) as e:
                    print(f"\n\n-*-*- WARNING: Skipped frame {frame_count + 1} due to error: {e} -*-*-\n")
                    target_location = sim_world.get_random_location_from_navigation()
                    player_controller.go_to_location(target_location)
                    last_goal_frame = frame_count
                    continue
    finally:
        print('\nDestroying actors and cleaning up.')
        if 'sim_world' in locals() and 'original_settings' in locals():
             sim_world.apply_settings(original_settings)
        if 'client' in locals() and client:
            client.apply_batch([carla.command.DestroyActor(x) for x in actor_list if x is not None and x.is_alive])

def main():
    argparser = argparse.ArgumentParser(description='CARLA Multi-Camera Off-Screen Frame Saver')
    
    argparser.add_argument(
        '--mode', 
        choices=['custom', 'old', 'bev'], 
        default='custom', 
        help='Which camera setup to run: "custom" (intrinsic), "old" (arg-based), "bev" (trajectory)'
    )
    
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to')
    argparser.add_argument('-f', '--frames', default=100, type=int, help='Total simulation frames to run')
    argparser.add_argument('--fps', default=10, type=int, help='Simulation FPS (Default: 10)')
    
    argparser.add_argument(
        '--save_every', 
        default=1, 
        type=int, 
        help='Frequency to save images (e.g., 1 = every frame, 10 = every 10th frame)'
    )
    
    argparser.add_argument('--cars', default=50, type=int, help='Number of vehicles to spawn')
    argparser.add_argument('--peds', default=500, type=int, help='Number of pedestrians to spawn')
    
    argparser.add_argument('--width', default=1280, type=int, help="[Mode: old] Final combined image width")
    argparser.add_argument('--height', default=720, type=int, help="[Mode: old] Final combined image height")
    
    argparser.add_argument('--bev_size', default=600, type=int, help='[Mode: bev] Width and height of the BEV camera (pixels)')
    argparser.add_argument('--bev_height', default=50, type=int, help='[Mode: bev] Height of the BEV camera (meters)')
    argparser.add_argument('--trajectory_length', default=500, type=int, help='[Mode: bev] Max points for BEV trajectory')

    args = argparser.parse_args()
    
    print(f"--- Running in '{args.mode}' mode ---")
    if args.mode == 'custom':
        print("  -> Using hardcoded intrinsic camera settings.")
    elif args.mode == 'old':
        print(f"  -> Using argument-based settings ({args.width}x{args.height} total).")
    elif args.mode == 'bev':
        print(f"  -> Using BEV settings ({args.bev_size}x{args.bev_size}).")
    
    print(f"  -> Simulating at {args.fps} FPS, saving every {args.save_every} frame(s).")

    try:
        run_simulation(args)
    except Exception as e:
        print(f"\nAn unrecoverable error occurred: {e}")
        
if __name__ == '__main__':
    main()