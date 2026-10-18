using UnityEngine;
using AprilTag;
using System.Collections.Generic;
using System.Linq;

/// <summary>
/// Estimates the position and rotation of a cube with AprilTags on each face.
/// Uses exponential smoothing for stable tracking and maintains estimation
/// even when tags are temporarily not visible.
/// </summary>
public class CubePositionEstimator : MonoBehaviour
{
    [Header("Detection Source")]
    [SerializeField, Tooltip("Reference to DetectionTest component for tag detection")]
    private DetectionTest detectionSource = null;

    [SerializeField, Tooltip("Camera used for detection (if null, uses Camera.main)")]
    private Camera detectionCamera = null;

    [Header("Cube Configuration")]
    [SerializeField, Tooltip("Tag IDs for each face in order: +X, -X, +Y, -Y, +Z, -Z")]
    private int[] cubeFaceTagIDs = new int[6] { 0, 1, 2, 3, 4, 5 };

    [SerializeField, Tooltip("Distance from cube center to face (half the cube size)")]
    private float cubeHalfSize = 0.05f; // 5cm = 10cm cube

    [Header("Smoothing")]
    [SerializeField, Tooltip("Smoothing factor (0-1). Higher = faster response, lower = smoother")]
    [Range(0.01f, 1f)]
    private float smoothingAlpha = 0.3f;

    [SerializeField, Tooltip("Frames without detection before IsTracking becomes false")]
    private int lostTrackingThreshold = 30;

    [Header("Tracking Output")]
    [SerializeField, Tooltip("Optional: Transform to automatically update with estimated pose")]
    private Transform trackedObject = null;

    [SerializeField, Tooltip("Show debug visualization")]
    private bool showDebug = true;

    // Private state
    private Vector3 estimatedPosition;
    private Quaternion estimatedRotation = Quaternion.identity;
    private bool isInitialized = false;
    private int framesSinceLastDetection = 0;

    // Public API
    public Vector3 EstimatedPosition => estimatedPosition;
    public Quaternion EstimatedRotation => estimatedRotation;
    public bool IsTracking => isInitialized && framesSinceLastDetection < lostTrackingThreshold;
    public int VisibleTagCount { get; private set; }

    // Face normals in local cube space (corresponding to tag ID array order)
    private static readonly Vector3[] faceNormals = new Vector3[6]
    {
        Vector3.right,   // +X (face 0)
        Vector3.left,    // -X (face 1)
        Vector3.up,      // +Y (face 2)
        Vector3.down,    // -Y (face 3)
        Vector3.forward, // +Z (face 4)
        Vector3.back     // -Z (face 5)
    };

    private void Start()
    {
        // Auto-find DetectionTest if not assigned
        if (detectionSource == null)
        {
            detectionSource = FindObjectOfType<DetectionTest>();
            if (detectionSource == null)
            {
                Debug.LogError("CubePositionEstimator: No DetectionTest found in scene! Please assign one.");
            }
        }

        // Auto-find camera if not assigned
        if (detectionCamera == null)
        {
            detectionCamera = Camera.main;
        }
    }

    private void LateUpdate()
    {
        if (detectionSource == null || detectionCamera == null)
        {
            return;
        }

        // Get detected tags
        var detectedTags = detectionSource.GetDetectedTags();
        if (detectedTags == null)
        {
            return;
        }

        // Collect observations from cube tags
        List<CubeObservation> observations = new List<CubeObservation>();

        foreach (var tag in detectedTags)
        {
            int faceIndex = GetFaceIndexForTag(tag.ID);
            if (faceIndex >= 0)
            {
                // Calculate implied cube center from this tag
                CubeObservation observation = CalculateCubePoseFromTag(tag, faceIndex);
                observations.Add(observation);
            }
        }

        VisibleTagCount = observations.Count;

        // Update estimation if we have observations
        if (observations.Count > 0)
        {
            // Average all positions
            Vector3 avgPosition = AveragePositions(observations);

            // Average all rotations (should be consistent now that calculation is fixed)
            Quaternion avgRotation = GetMostFrontalRotation(observations);

            // Apply smoothing
            if (!isInitialized)
            {
                // First detection - initialize directly
                estimatedPosition = avgPosition;
                estimatedRotation = avgRotation;
                isInitialized = true;
            }
            else
            {
                // Ensure quaternions are in same hemisphere before Slerp (prevents flipping)
                if (Quaternion.Dot(estimatedRotation, avgRotation) < 0)
                {
                    // Negate to ensure shortest path
                    avgRotation = new Quaternion(-avgRotation.x, -avgRotation.y, -avgRotation.z, -avgRotation.w);
                }

                estimatedRotation = avgRotation;
            }

            framesSinceLastDetection = 0;
        }
        else
        {
            // No observations - keep last estimate but increment lost counter
            framesSinceLastDetection++;
        }

        // Update tracked object if assigned and initialized
        if (trackedObject != null && isInitialized)
        {
            trackedObject.position = estimatedPosition;
            trackedObject.rotation = estimatedRotation;
        }
    }

    /// <summary>
    /// Finds which cube face (index 0-5) corresponds to the given tag ID.
    /// Returns -1 if tag ID is not part of this cube.
    /// </summary>
    private int GetFaceIndexForTag(int tagID)
    {
        for (int i = 0; i < cubeFaceTagIDs.Length && i < 6; i++)
        {
            if (cubeFaceTagIDs[i] == tagID)
            {
                return i;
            }
        }
        return -1;
    }

    /// <summary>
    /// Calculates the cube's world position and rotation from a single tag observation.
    /// </summary>
    private CubeObservation CalculateCubePoseFromTag(TagPose tag, int faceIndex)
    {
        // Transform tag from camera-local to world space
        Vector3 tagWorldPos = detectionCamera.transform.TransformPoint(tag.Position);
        Quaternion tagWorldRot = detectionCamera.transform.rotation * tag.Rotation;

        // Tag axes in world space:
        // IMPORTANT: Tags are mounted with Z+ pointing INWARD toward cube center!
        // Z+ = inward (opposite of face normal)
        // Y+ = varies by face
        // X+ = varies by face
        Vector3 tagX = tagWorldRot * Vector3.right;
        Vector3 tagY = tagWorldRot * Vector3.up;
        Vector3 tagZ = tagWorldRot * Vector3.forward;  // Points INWARD

        // Cube center position: tag Z+ points inward, so move along it
        Vector3 cubeCenter = tagWorldPos + tagZ * cubeHalfSize;

        // Now calculate cube rotation by mapping tag axes to cube axes
        // Based on actual mounting (tags have Z+ pointing INWARD):
        //
        // Front +Z face (faceIndex 4): tagX→-X, tagY→+Y, tagZ→-Z (inward)
        // Top +Y face (faceIndex 2): tagX→+X, tagY→+Z, tagZ→-Y (inward)

        Vector3 cubeXWorld, cubeYWorld, cubeZWorld;

        // Side faces: tag Y+ points up (cube +Y), tag Z+ points inward
        // Top/bottom: tag is rotated 90° around X

        if (faceIndex == 0) // +X face (right)
        {
            // Measured: tagX→+Z, tagY→+Y, tagZ→-X (inward)
            cubeXWorld = -tagZ;  // cube +X = opposite of inward = +X
            cubeYWorld = tagY;   // cube +Y = tag up = +Y
            cubeZWorld = Vector3.Cross(cubeXWorld, cubeYWorld).normalized;  // Ensure right-handed
        }
        else if (faceIndex == 1) // -X face (left)
        {
            // Inferred: tagX→-Z, tagY→+Y, tagZ→+X (inward)
            cubeXWorld = tagZ;   // cube +X = inward direction = +X
            cubeYWorld = tagY;   // cube +Y = tag up = +Y
            cubeZWorld = Vector3.Cross(cubeXWorld, cubeYWorld).normalized;  // Ensure right-handed
        }
        else if (faceIndex == 2) // +Y face (top)
        {
            // Measured: tagX→+X, tagY→+Z, tagZ→-Y (inward)
            cubeXWorld = tagX;   // cube +X = tag right = +X
            cubeYWorld = -tagZ;  // cube +Y = opposite of inward = +Y
            cubeZWorld = Vector3.Cross(cubeXWorld, cubeYWorld).normalized;  // Ensure right-handed
        }
        else if (faceIndex == 3) // -Y face (bottom)
        {
            // Inferred (rotated 180° from top around X axis): tagX→+X, tagY→-Z, tagZ→+Y (inward)
            cubeXWorld = tagX;   // cube +X = tag right = +X
            cubeYWorld = tagZ;   // cube +Y = inward direction = +Y
            cubeZWorld = Vector3.Cross(cubeXWorld, cubeYWorld).normalized;  // Ensure right-handed
        }
        else if (faceIndex == 4) // +Z face (front)
        {
            // Measured: tagX→-X, tagY→+Y, tagZ→-Z (inward)
            cubeXWorld = -tagX;  // cube +X = opposite of tag right = +X
            cubeYWorld = tagY;   // cube +Y = tag up = +Y
            cubeZWorld = Vector3.Cross(cubeXWorld, cubeYWorld).normalized;  // Ensure right-handed
        }
        else // faceIndex == 5, -Z face (back)
        {
            // Inferred: tagX→+X, tagY→+Y, tagZ→+Z (inward)
            cubeXWorld = tagX;   // cube +X = tag right = +X
            cubeYWorld = tagY;   // cube +Y = tag up = +Y
            cubeZWorld = Vector3.Cross(cubeXWorld, cubeYWorld).normalized;  // Ensure right-handed
        }

        // Construct rotation from basis vectors
        Matrix4x4 rotMatrix = new Matrix4x4();
        rotMatrix.SetColumn(0, cubeXWorld);
        rotMatrix.SetColumn(1, cubeYWorld);
        rotMatrix.SetColumn(2, cubeZWorld);
        rotMatrix.SetColumn(3, new Vector4(0, 0, 0, 1));

        Quaternion cubeRotation = rotMatrix.rotation;

        // Calculate confidence based on viewing angle
        // Tag Z+ points inward, so face normal is -tagZ
        Vector3 faceNormalOutward = -tagZ;
        Vector3 toCamera = (detectionCamera.transform.position - tagWorldPos).normalized;
        float facingAmount = Vector3.Dot(faceNormalOutward, toCamera);
        float confidence = Mathf.Max(0.1f, facingAmount);

        return new CubeObservation
        {
            position = cubeCenter,
            rotation = cubeRotation,
            confidence = confidence,
            faceIndex = faceIndex
        };
    }

    /// <summary>
    /// Gets the rotation from the most frontal (most directly facing camera) observation.
    /// This is more stable than averaging rotations from different faces.
    /// </summary>
    private Quaternion GetMostFrontalRotation(List<CubeObservation> observations)
    {
        if (observations.Count == 0)
            return Quaternion.identity;

        if (observations.Count == 1)
            return observations[0].rotation;

        // Find observation with highest confidence (most frontal)
        CubeObservation best = observations[0];
        for (int i = 1; i < observations.Count; i++)
        {
            if (observations[i].confidence > best.confidence)
            {
                best = observations[i];
            }
        }

        return best.rotation;
    }

    /// <summary>
    /// Averages multiple position observations.
    /// </summary>
    private Vector3 AveragePositions(List<CubeObservation> observations)
    {
        Vector3 sum = Vector3.zero;
        float totalWeight = 0f;

        foreach (var obs in observations)
        {
            sum += obs.position * obs.confidence;
            totalWeight += obs.confidence;
        }

        return totalWeight > 0 ? sum / totalWeight : sum;
    }

    /// <summary>
    /// Resets the tracking state.
    /// </summary>
    public void ResetTracking()
    {
        isInitialized = false;
        framesSinceLastDetection = 0;
        estimatedRotation = Quaternion.identity;
    }

    // Helper struct for storing observations
    private struct CubeObservation
    {
        public Vector3 position;
        public Quaternion rotation;
        public float confidence;
        public int faceIndex;
    }

    // Gizmos visualization (only visible in Scene view)
    private void OnDrawGizmos()
    {
        if (!showDebug || !isInitialized) return;

        // Draw cube wireframe using Gizmos
        Gizmos.color = IsTracking ? Color.yellow : Color.gray;
        Gizmos.matrix = Matrix4x4.TRS(estimatedPosition, estimatedRotation, Vector3.one);
        Gizmos.DrawWireCube(Vector3.zero, Vector3.one * cubeHalfSize * 2f);
        Gizmos.matrix = Matrix4x4.identity;

        // Draw coordinate axes in world space with arrows
        float axisLength = cubeHalfSize * 2f;
        float arrowHeadLength = cubeHalfSize * 0.3f;
        float arrowHeadAngle = 20f;

        // X axis (RED)
        Vector3 xEnd = estimatedPosition + estimatedRotation * Vector3.right * axisLength;
        Gizmos.color = Color.red;
        Gizmos.DrawLine(estimatedPosition, xEnd);
        DrawArrowHead(xEnd, estimatedRotation * Vector3.right, arrowHeadLength, arrowHeadAngle, Color.red);

        // Y axis (GREEN)
        Vector3 yEnd = estimatedPosition + estimatedRotation * Vector3.up * axisLength;
        Gizmos.color = Color.green;
        Gizmos.DrawLine(estimatedPosition, yEnd);
        DrawArrowHead(yEnd, estimatedRotation * Vector3.up, arrowHeadLength, arrowHeadAngle, Color.green);

        // Z axis (BLUE)
        Vector3 zEnd = estimatedPosition + estimatedRotation * Vector3.forward * axisLength;
        Gizmos.color = Color.blue;
        Gizmos.DrawLine(estimatedPosition, zEnd);
        DrawArrowHead(zEnd, estimatedRotation * Vector3.forward, arrowHeadLength, arrowHeadAngle, Color.blue);
    }

    private void DrawArrowHead(Vector3 tip, Vector3 direction, float length, float angle, Color color)
    {
        Vector3 right = Quaternion.LookRotation(direction) * Quaternion.Euler(0, 180 + angle, 0) * Vector3.forward;
        Vector3 left = Quaternion.LookRotation(direction) * Quaternion.Euler(0, 180 - angle, 0) * Vector3.forward;
        Vector3 up = Quaternion.LookRotation(direction) * Quaternion.Euler(180 + angle, 0, 0) * Vector3.forward;
        Vector3 down = Quaternion.LookRotation(direction) * Quaternion.Euler(180 - angle, 0, 0) * Vector3.forward;

        Gizmos.color = color;
        Gizmos.DrawLine(tip, tip + right * length);
        Gizmos.DrawLine(tip, tip + left * length);
        Gizmos.DrawLine(tip, tip + up * length);
        Gizmos.DrawLine(tip, tip + down * length);
    }

#if UNITY_EDITOR
    // Show debug info in Scene view only
    private void OnDrawGizmosSelected()
    {
        if (!showDebug || !isInitialized) return;

        // Draw debug label in scene view
        UnityEditor.Handles.Label(estimatedPosition + Vector3.up * cubeHalfSize * 2.5f,
            $"Cube Tracker\nTracking: {IsTracking}\nTags: {VisibleTagCount}/6\nFrames: {framesSinceLastDetection}");
    }
#endif
}
