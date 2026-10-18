using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using AprilTag;

/// <summary>
/// AprilTag detector that processes the Unity camera's rendered view
/// Detects tags placed in the scene and visualizes their positions
/// </summary>
[RequireComponent(typeof(Camera))]
public class AprilTagDetectorSample : MonoBehaviour
{
    [Header("Detection Settings")]
    [Range(1, 4)]
    public int decimation = 2;

    public float tagSize = 1.0f;

    [Header("Camera Settings")]
    public int detectionWidth = 640;
    public int detectionHeight = 480;

    [Header("Visualization")]
    public GameObject tagVisualizationPrefab;
    public Transform visualizationParent;
    public UnityEngine.UI.RawImage debugView;

    private TagDetector detector;
    private RenderTexture renderTexture;
    private Texture2D readbackTexture;
    private Color32[] imageBuffer;
    private Dictionary<int, List<GameObject>> tagVisualizations = new Dictionary<int, List<GameObject>>();
    private Camera mainCamera;
    private int visualizationCounter = 0;

    void Start()
    {
        mainCamera = GetComponent<Camera>();

        // Create render texture
        renderTexture = new RenderTexture(detectionWidth, detectionHeight, 24);
        renderTexture.format = RenderTextureFormat.ARGB32;
        renderTexture.filterMode = FilterMode.Point;
        mainCamera.targetTexture = renderTexture;

        // Create display camera to render to screen
        GameObject displayCamObj = new GameObject("DisplayCamera");
        displayCamObj.transform.SetParent(transform);
        displayCamObj.transform.localPosition = Vector3.zero;
        displayCamObj.transform.localRotation = Quaternion.identity;

        Camera displayCam = displayCamObj.AddComponent<Camera>();
        displayCam.CopyFrom(mainCamera);
        displayCam.targetTexture = null;
        displayCam.depth = mainCamera.depth - 1;

        // Initialize detector
        readbackTexture = new Texture2D(detectionWidth, detectionHeight, TextureFormat.RGB24, false);
        imageBuffer = new Color32[detectionWidth * detectionHeight];
        detector = new TagDetector(detectionWidth, detectionHeight, decimation);

        // Setup debug view
        if (debugView != null)
        {
            debugView.texture = renderTexture;
        }

        StartCoroutine(DetectionCoroutine());
    }

    IEnumerator DetectionCoroutine()
    {
        while (true)
        {
            yield return new WaitForEndOfFrame();

            if (detector == null)
                continue;

            // Read from render texture
            RenderTexture.active = renderTexture;
            readbackTexture.ReadPixels(new Rect(0, 0, detectionWidth, detectionHeight), 0, 0);
            readbackTexture.Apply();
            RenderTexture.active = null;

            // Process image
            imageBuffer = readbackTexture.GetPixels32();
            float fov = mainCamera.fieldOfView;
            if (mainCamera.aspect > 1)
            {
                fov = Camera.VerticalToHorizontalFieldOfView(fov, mainCamera.aspect);
            }

            detector.ProcessImage(imageBuffer, fov, tagSize);

            // Group detections by tag ID
            var detectionsByID = new Dictionary<int, System.Collections.Generic.List<TagPose>>();
            foreach (var tag in detector.DetectedTags)
            {
                if (!detectionsByID.ContainsKey(tag.ID))
                {
                    detectionsByID[tag.ID] = new System.Collections.Generic.List<TagPose>();
                }
                detectionsByID[tag.ID].Add(tag);
            }

            // Hide all visualizations
            foreach (var list in tagVisualizations.Values)
            {
                foreach (var vis in list)
                {
                    if (vis != null) vis.SetActive(false);
                }
            }

            // Update visualizations
            foreach (var kvp in detectionsByID)
            {
                int tagID = kvp.Key;
                var detections = kvp.Value;

                if (!tagVisualizations.ContainsKey(tagID))
                {
                    tagVisualizations[tagID] = new List<GameObject>();
                }

                // Create visualizations if needed
                while (tagVisualizations[tagID].Count < detections.Count)
                {
                    if (tagVisualizationPrefab != null)
                    {
                        GameObject newVis = Instantiate(tagVisualizationPrefab, visualizationParent);
                        newVis.name = $"TagVis_{tagID}_{visualizationCounter++}";

                        TagVisualizer visualizer = newVis.GetComponent<TagVisualizer>();
                        if (visualizer != null)
                        {
                            visualizer.tagID = tagID;
                        }

                        tagVisualizations[tagID].Add(newVis);
                    }
                }

                // Update positions and rotations
                for (int i = 0; i < detections.Count; i++)
                {
                    var tag = detections[i];

                    if (tagVisualizationPrefab != null && i < tagVisualizations[tagID].Count)
                    {
                        var tagObj = tagVisualizations[tagID][i];
                        tagObj.SetActive(true);

                        // Only use rotation, position based on rotation to place on cube face
                        Quaternion worldRot = mainCamera.transform.rotation * tag.Rotation;

                        // Convert to local space if there's a parent
                        if (visualizationParent != null)
                        {
                            // Calculate face position in world space first
                            Vector3 faceNormal = worldRot * Vector3.forward;
                            Vector3 worldFacePos = visualizationParent.position + visualizationParent.TransformDirection(faceNormal * (tagSize / 2f));

                            // Convert world rotation to local rotation
                            Quaternion localRot = Quaternion.Inverse(visualizationParent.rotation) * worldRot;

                            tagObj.transform.localPosition = worldFacePos;
                            tagObj.transform.localRotation = worldRot;
                        }
                        else
                        {
                            // No parent, use world space
                            Vector3 faceNormal = worldRot * Vector3.forward;
                            Vector3 facePos = faceNormal * (tagSize / 2f);

                            tagObj.transform.position = facePos;
                            tagObj.transform.rotation = worldRot;
                        }
                    }
                }
            }
        }
    }

    void OnDestroy()
    {
        detector?.Dispose();

        if (renderTexture != null)
        {
            mainCamera.targetTexture = null;
            Destroy(renderTexture);
        }

        if (readbackTexture != null)
        {
            Destroy(readbackTexture);
        }

        foreach (var list in tagVisualizations.Values)
        {
            foreach (var vis in list)
            {
                if (vis != null)
                    Destroy(vis);
            }
        }
    }
}
