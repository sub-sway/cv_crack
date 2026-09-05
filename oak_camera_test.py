import depthai as dai
import cv2
import numpy as np

pipeline = dai.Pipeline()

# RGB
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(
    dai.ColorCameraProperties.SensorResolution.THE_1080_P
)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(15)

# LEFT
left = pipeline.create(dai.node.MonoCamera)
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
left.setResolution(
    dai.MonoCameraProperties.SensorResolution.THE_480_P
)
left.setFps(15)

# RIGHT
right = pipeline.create(dai.node.MonoCamera)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
right.setResolution(
    dai.MonoCameraProperties.SensorResolution.THE_480_P
)
right.setFps(15)

# Stereo
stereo = pipeline.create(dai.node.StereoDepth)

# 처음엔 옵션 최소화
stereo.setLeftRightCheck(True)

left.out.link(stereo.left)
right.out.link(stereo.right)

# 출력
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

print("OAK-D Lite basic test")

with dai.Device(pipeline) as device:

    print("USB speed:", device.getUsbSpeed())

    q_rgb = device.getOutputQueue(
        "rgb",
        maxSize=2,
        blocking=False
    )

    q_depth = device.getOutputQueue(
        "depth",
        maxSize=2,
        blocking=False
    )

    while True:

        rgb_packet = q_rgb.tryGet()
        depth_packet = q_depth.tryGet()

        if rgb_packet is not None:
            rgb = rgb_packet.getCvFrame()
            cv2.imshow("RGB", rgb)

        if depth_packet is not None:

            depth = depth_packet.getFrame()

            depth_vis = cv2.normalize(
                depth,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            )

            depth_vis = depth_vis.astype(np.uint8)

            depth_vis = cv2.applyColorMap(
                depth_vis,
                cv2.COLORMAP_JET
            )

            cv2.imshow("Depth", depth_vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()