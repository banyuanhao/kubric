import cv2
from PIL import Image
video_path = "/fsx/yban/myproject/kubric/generated_dataset/reverse_time/videos/video_00000.mp4"
cap = cv2.VideoCapture("/fsx/yban/myproject/kubric/generated_dataset/reverse_time/videos/video_00000.mp4")


total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

frames = []
for idx in frame_inds:
    if idx >= total_frames:
        idx = total_frames - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    # HACK: sometimes OpenCV fails to read frames, return a black frame instead
    try:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
    except Exception as e:
        print(f"[Warning] Error reading frame {idx} from {video_path}: {e}")
        # First, try to read the first frame
        try:
            print(f"[Warning] Try reading first frame.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
        # If that fails, return a black frame
        except Exception as e:
            print(f"[Warning] Error in reading first frame from {video_path}: {e}")
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame = Image.new("RGB", (width, height), (0, 0, 0))

    # HACK: if height or width is 0, return a black frame instead
    if frame.height == 0 or frame.width == 0:
        height = width = 256
        frame = Image.new("RGB", (width, height), (0, 0, 0))

    frames.append(frame)