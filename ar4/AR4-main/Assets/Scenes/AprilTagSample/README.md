# AprilTag Sample Scene

This sample demonstrates how to use the AprilTag package (jp.keijiro.apriltag) to detect **tagStandard41h12** markers in real-time using a webcam.

## Files Created

- **AprilTagSample.unity** - The sample scene with a camera setup
- **AprilTagDetectorSample.cs** - Main detection script (attach to Main Camera)
- **TagVisualizer.cs** - Optional script for visualizing detected tags with coordinate axes

## Setup Instructions

### 1. Get AprilTag Images

Download tagStandard41h12 images from the official repository:
https://github.com/AprilRobotics/apriltag-imgs/tree/master/tagStandard41h12

Print some tags or display them on a smartphone/tablet screen for testing.

### 2. Open the Scene

1. Open `Assets/Scenes/AprilTagSample/AprilTagSample.unity`
2. The scene contains a Main Camera with the `AprilTagDetectorSample` component already attached

### 3. Configure the Detector (Optional)

Select the Main Camera and adjust these settings in the Inspector:

#### Detection Settings
- **Decimation** (1-4): Higher values = faster but less accurate detection. Default: 2
- **Tag Size**: Physical size of your printed tag in meters. Default: 0.05m (5cm)

#### Camera Settings
- **Horizontal FOV**: Your camera's horizontal field of view in degrees. Default: 60°
  - For accurate 3D position estimation, match this to your actual camera's FOV
  - You can find your webcam's FOV in its specifications
  - Common values: 60-90° for most webcams

#### Visualization (Optional)
- **Tag Visualization Prefab**: Assign a prefab to visualize detected tags in 3D space
  - Create an empty GameObject
  - Add the `TagVisualizer` component to it
  - Save as a prefab
  - Assign it to this field

#### Debug
- **Debug Log**: Enable to see detection info in the console

### 4. Create a Tag Visualization Prefab (Optional)

To see detected tags visualized in 3D:

1. Create an empty GameObject in the scene
2. Add the `TagVisualizer` script component
3. Adjust the visualization settings:
   - **Tag Color**: Color of the cube representing the tag
   - **Size**: Size of the visualization (should match your actual tag size)
4. Drag the GameObject into your Project window to create a prefab
5. Delete the GameObject from the scene
6. Assign the prefab to the `Tag Visualization Prefab` field on the detector

### 5. Run the Scene

1. Press Play in Unity Editor
2. Grant webcam permissions when prompted
3. Show a tagStandard41h12 marker to your webcam
4. The script will:
   - Display the webcam feed in the top-right corner
   - Show detection stats on the top-left
   - Log detected tag IDs and positions to the console (if Debug Log is enabled)
   - Display 3D visualizations at tag positions (if prefab is assigned)

## Understanding the Detection

### Tag Coordinate System

Detected tags provide:
- **Position** (Vector3): 3D position in camera space (meters)
- **Rotation** (Quaternion): 3D orientation
- **ID** (int): Unique tag identifier

### Adjusting for Accuracy

For accurate 3D position estimation:

1. **Measure your tag size** - Measure the actual printed tag size (black square outer edge)
2. **Set the tag size** - Enter the measured size in meters in the `Tag Size` field
3. **Match camera FOV** - Find your webcam's actual horizontal FOV and set it in `Horizontal FOV`
   - Example: Logitech C920 has ~78° horizontal FOV
   - You can also calibrate by measuring known distances

### Performance Tips

- Increase **Decimation** for better performance at the cost of accuracy
- Lower webcam resolution if detection is too slow
- Ensure good lighting - tags need clear contrast
- Keep tags flat and avoid reflections

## Troubleshooting

### No Webcam Detected
- Check that your webcam is connected and enabled
- Check Unity console for error messages
- Try different USB ports if using external webcam

### Tags Not Detected
- Ensure you're using **tagStandard41h12** family tags (not tag36h11 or other families)
- Improve lighting - avoid shadows on the tag
- Make sure the tag is clearly visible and in focus
- Try moving the tag closer to the camera
- Check that the tag is not distorted or wrinkled

### Inaccurate Positions
- Verify the **Tag Size** matches your actual printed tag size
- Calibrate the **Horizontal FOV** to match your camera
- Use lower **Decimation** for better accuracy
- Ensure the tag surface is flat

### Low FPS
- Increase **Decimation** value
- Reduce webcam resolution
- Close other applications using the webcam

## Script Reference

### AprilTagDetectorSample.cs

Main component for detecting AprilTags from webcam input.

**Key Methods:**
- `Start()` - Initializes webcam and detector
- `Update()` - Processes each frame and detects tags
- `OnDestroy()` - Cleanup and disposal

**Properties:**
- `detector.DetectedTags` - IEnumerable of all detected tags
- Each tag has: `ID`, `Position`, `Rotation`

### TagVisualizer.cs

Simple visualization component that creates a cube with coordinate axes.

## Resources

- **AprilTag Package**: https://github.com/keijiro/jp.keijiro.apriltag
- **AprilTag Images**: https://github.com/AprilRobotics/apriltag-imgs
- **AprilTag Homepage**: https://april.eecs.umich.edu/software/apriltag

## Notes

- This package currently only supports **tagStandard41h12** family
- For other tag families (tag36h11, tag25h9, etc.), you would need to modify the native plugin bindings
- Detection runs on a separate job system for better performance
- The detector automatically uses multiple threads based on your CPU
