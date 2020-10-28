import argparse
import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Path to video")
    args = parser.parse_args()

    # Baseline
    # tracker = cv2.TrackerMIL_create()

    # Faster FPS lower accuracy
    # tracker = cv2.TrackerKCF_create()

    # High accuracy slower FPS
    # tracker = cv2.TrackerCSRT_create()

    # Very very fast
    tracker = cv2.TrackerMOSSE_create()


    init_bbox = None

    video = cv2.VideoCapture(args.video)
    fps = None

    while True:

        success, frame = video.read()
        if not success: break
        # Resize frame
        frame = imutils.resize(frame, width=800)
        height, width = frame.shape[:2]
        # If bbox if defined
        if init_bbox:
            # Track object
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(x) for x in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Success: {success}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            fps.update()
            fps.stop()
            cv2.putText(frame, f"FPS: {fps.fps():.4f}", (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Show frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(2) & 0xFF

        # Press 's' key to select bbox, then 'Enter' to confirm. Use 'Esc' to reselect
        if key == ord("s"):
            init_bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            # Initialize tracker
            tracker.init(frame, init_bbox)
            fps = FPS().start()

        elif key == ord("q"):
            break


    video.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
