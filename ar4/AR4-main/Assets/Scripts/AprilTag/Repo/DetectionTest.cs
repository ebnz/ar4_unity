using UnityEngine;
using System.Linq;
using System.Collections.Generic;
using UI = UnityEngine.UI;
using Klak.TestTools;

sealed class DetectionTest : MonoBehaviour
{
    [SerializeField] ImageSource _source = null;
    [SerializeField] int _decimation = 4;
    [SerializeField] float _tagSize = 0.05f;
    [SerializeField] Material _tagMaterial = null;
    [SerializeField] UI.RawImage _webcamPreview = null;
    [SerializeField] UI.Text _debugText = null;
    [SerializeField] Camera _camera = null;

    [Header("Tag Tracking")]
    [SerializeField] Transform trackedObject = null;
    [SerializeField, Tooltip("Which tag ID to track (-1 = track first detected tag)")]
    int targetTagID = -1;

    AprilTag.TagDetector _detector;
    TagDrawer _drawer;

    void Start()
    {
        var dims = _source.OutputResolution;
        _detector = new AprilTag.TagDetector(dims.x, dims.y, _decimation);
        _drawer = new TagDrawer(_tagMaterial);
    }

    void OnDestroy()
    {
        _detector.Dispose();
        _drawer.Dispose();
    }

    void LateUpdate()
    {
        _webcamPreview.texture = _source.AsTexture;

        // Source image acquisition
        var image = _source.AsTexture.AsSpan();
        if (image.IsEmpty) return;

        // Use _camera if assigned, otherwise default to Camera.main
        var cam = _camera != null ? _camera : Camera.main;
        var fov = cam.fieldOfView * Mathf.Deg2Rad;
        _detector.ProcessImage(image, fov, _tagSize);


        // Detected tag visualization and tracking
        bool foundTargetTag = false;
        foreach (var tag in _detector.DetectedTags)
        {
            // Transform from camera-local space to world space
            Vector3 worldPos = cam.transform.TransformPoint(tag.Position);
            Quaternion worldRot = cam.transform.rotation * tag.Rotation;

            _drawer.Draw(tag.ID, worldPos, worldRot, _tagSize);

            // Update tracked object position/rotation
            if (trackedObject != null)
            {
                // Track specific tag ID, or first detected tag if targetTagID is -1
                if (targetTagID == -1 || tag.ID == targetTagID)
                {
                    trackedObject.position = worldPos;
                    trackedObject.rotation = worldRot;
                    foundTargetTag = true;
                    if (targetTagID != -1) break; // Stop if we found our specific tag
                }
            }
        }

        // Optionally hide tracked object if tag not detected
        // Uncomment if you want the object to disappear when tag is not visible
        // if (trackedObject != null && !foundTargetTag)
        //     trackedObject.gameObject.SetActive(false);
        // else if (trackedObject != null && foundTargetTag)
        //     trackedObject.gameObject.SetActive(true);

        // Profile data output (with 30 frame interval)
        if (Time.frameCount % 30 == 0)
            _debugText.text = _detector.ProfileData.Aggregate
              ("Profile (usec)", (c, n) => $"{c}\n{n.name} : {n.time}");
    }

    public IEnumerable<AprilTag.TagPose> GetDetectedTags()
    {
        return _detector.DetectedTags;
    }
}
