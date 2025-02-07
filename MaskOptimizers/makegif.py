import cv2
import os
import numpy as np

def create_video_from_images(image_folder, output_video, final_duration, quality=90, fps=30):
    """
    Creates a video from a series of images.

    Parameters:
        image_folder (str): Path to the folder containing images.
        output_video (str): Path to save the output video.
        final_duration (float): Total duration of the output video in seconds.
        quality (int): Quality of the output video (default 90, range 1-100).
        fps (int): Frames per second (default 30).
    """
    images = []
    valid_extensions = {'.pgm', '.jpg', '.jpeg'}
    
    # Load images
    l = os.listdir(image_folder)
    l.sort(reverse = False, key = lambda x : int(x[4:-4]))
    for filename in l:
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    
    if not images:
        raise ValueError("No valid images found in the specified folder.")
    
    height, width, _ = images[0].shape
    total_frames = int(fps * final_duration)
    frame_count = min(total_frames, len(images))
    
    # Define video codec and writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # Determine frame step to fit duration
    step = max(1, len(images) // frame_count)
    
    for i in range(0, len(images), step):
        video_writer.write(images[i])
    
    video_writer.release()
    print(f"Video saved at {output_video}")

# Example usage
create_video_from_images('tem1', 'output.mp4', final_duration=10, quality=50, fps=30)
