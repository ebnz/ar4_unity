using UnityEngine;

/// <summary>
/// Detects if an object is being grasped (in trigger volume AND colliding with both inside gripper colliders)
/// </summary>
public class GraspDetector : MonoBehaviour
{
    [Header("Gripper Collision Detection")]
    public Collider leftGripperInsideCollider;
    public Collider rightGripperInsideCollider;
    public Collider leftGripperOutsideCollider;
    public Collider rightGripperOutsideCollider;

    [Header("Grip Volume Trigger")]
    public GameObject gripVolumeTrigger;
    [Range(0f, 1f)]
    public float volumeScaleFactor = 0.2f;
    private BoxCollider gripVolumeTriggerCollider;
    private Vector3 initialGripVolumeLocalPosition; // Store initial local Y and Z position (relative to parent)

    [Header("State")]
    public bool isGrasping = false;
    public bool isTouchingOutside = false;
    public bool isInGripVolume = false;
    public bool leftInsideTouching = false;
    public bool rightInsideTouching = false;

    [Header("Debug Indicators")]
    public GameObject GraspingIndicator;
    public GameObject TouchingOutsideIndicator;
    public GameObject CubeBetweenIndicator;

    private bool leftOutsideTouching = false;
    private bool rightOutsideTouching = false;

    void Awake()
    {
        if (GraspingIndicator != null)
            GraspingIndicator.SetActive(false);
        if (TouchingOutsideIndicator != null)
            TouchingOutsideIndicator.SetActive(false);
        if (CubeBetweenIndicator != null)
            CubeBetweenIndicator.SetActive(false);

        // Get the grip volume trigger collider and store initial local position
        if (gripVolumeTrigger != null)
        {
            // Store the initial local Y and Z position - these should remain constant relative to parent
            initialGripVolumeLocalPosition = gripVolumeTrigger.transform.localPosition;

            gripVolumeTriggerCollider = gripVolumeTrigger.GetComponent<BoxCollider>();
            if (gripVolumeTriggerCollider != null)
            {
                gripVolumeTriggerCollider.isTrigger = true;
            }
        }
    }

    void FixedUpdate()
    {
        UpdateGripVolume();
    }

    /// <summary>
    /// Reset all grasp detection state (call on episode begin)
    /// </summary>
    public void ResetGraspState()
    {
        // Clear all collision states
        isGrasping = false;
        isTouchingOutside = false;
        isInGripVolume = false;
        leftInsideTouching = false;
        rightInsideTouching = false;
        leftOutsideTouching = false;
        rightOutsideTouching = false;

        // Update debug indicators
        if (GraspingIndicator != null)
            GraspingIndicator.SetActive(false);
        if (TouchingOutsideIndicator != null)
            TouchingOutsideIndicator.SetActive(false);
        if (CubeBetweenIndicator != null)
            CubeBetweenIndicator.SetActive(false);

        // Restore Y and Z to initial local position (in case they drifted)
        if (gripVolumeTrigger != null)
        {
            Vector3 resetLocalPos = gripVolumeTrigger.transform.localPosition;
            resetLocalPos.y = initialGripVolumeLocalPosition.y;
            resetLocalPos.z = initialGripVolumeLocalPosition.z;
            gripVolumeTrigger.transform.localPosition = resetLocalPos;
        }

        // Force immediate grip volume update to match reset gripper positions (updates X)
        UpdateGripVolume();
    }

    private void UpdateGripVolume()
    {
        if (gripVolumeTrigger == null || leftGripperInsideCollider == null || rightGripperInsideCollider == null) return;

        // Get the inside-facing edges of the gripper colliders
        Vector3 leftInsideEdge = GetInsideEdge((BoxCollider)leftGripperInsideCollider, true);
        Vector3 rightInsideEdge = GetInsideEdge((BoxCollider)rightGripperInsideCollider, false);

        // Position between the inside edges (only X position in world space)
        Vector3 currentPos = gripVolumeTrigger.transform.position;
        currentPos.x = (leftInsideEdge.x + rightInsideEdge.x) * 0.5f;
        gripVolumeTrigger.transform.position = currentPos;

        // Enforce local Y and Z remain at initial values (prevents drift)
        Vector3 localPos = gripVolumeTrigger.transform.localPosition;
        localPos.y = initialGripVolumeLocalPosition.y;
        localPos.z = initialGripVolumeLocalPosition.z;
        gripVolumeTrigger.transform.localPosition = localPos;

        // Size based on distance between inside edges (only X scale)
        float distance = Vector3.Distance(leftInsideEdge, rightInsideEdge);
        Vector3 currentScale = gripVolumeTrigger.transform.localScale;
        currentScale.x = volumeScaleFactor * distance;
        gripVolumeTrigger.transform.localScale = currentScale;
    }

    private Vector3 GetInsideEdge(BoxCollider collider, bool isLeftGripper)
    {
        // Get the center of the collider
        Vector3 center = collider.bounds.center;

        // Get the local bounds
        Bounds bounds = collider.bounds;

        // Calculate the inside-facing edge based on gripper side
        // Assuming the grippers face each other along the X-axis
        Vector3 insideEdge = center;

        if (isLeftGripper)
        {
            // Left gripper's right edge (facing right)
            insideEdge.x = bounds.min.x;
        }
        else
        {
            // Right gripper's left edge (facing left)
            insideEdge.x = bounds.max.x;
        }

        return insideEdge;
    }

    void OnCollisionEnter(Collision collision)
    {
        CheckGraspState(collision.collider, true);
    }

    void OnCollisionStay(Collision collision)
    {
        CheckGraspState(collision.collider, true);
    }

    void OnCollisionExit(Collision collision)
    {
        CheckGraspState(collision.collider, false);
    }

    void OnTriggerEnter(Collider other)
    {
        if (other == gripVolumeTriggerCollider || (gripVolumeTrigger != null && other.transform.IsChildOf(gripVolumeTrigger.transform)))
        {
            isInGripVolume = true;
            UpdateGraspState();
        }
    }

    void OnTriggerExit(Collider other)
    {
        if (other == gripVolumeTriggerCollider || (gripVolumeTrigger != null && other.transform.IsChildOf(gripVolumeTrigger.transform)))
        {
            isInGripVolume = false;
            UpdateGraspState();
        }
    }

    private void CheckGraspState(Collider other, bool isEntering)
    {
        // 'other' is the collider we collided with (the gripper colliders)
        // Check which gripper collider is involved
        if (other == leftGripperInsideCollider)
        {
            leftInsideTouching = isEntering;
        }
        else if (other == rightGripperInsideCollider)
        {
            rightInsideTouching = isEntering;
        }
        else if (other == leftGripperOutsideCollider)
        {
            leftOutsideTouching = isEntering;
        }
        else if (other == rightGripperOutsideCollider)
        {
            rightOutsideTouching = isEntering;
        }

        // Update outside touch state (OR condition - either outside collider)
        bool wasTouchingOutside = isTouchingOutside;
        isTouchingOutside = leftOutsideTouching || rightOutsideTouching;

        // Update grasp state
        UpdateGraspState();

        // Update debug indicators
        if (TouchingOutsideIndicator != null && wasTouchingOutside != isTouchingOutside)
        {
            TouchingOutsideIndicator.SetActive(isTouchingOutside);
        }
    }

    private void UpdateGraspState()
    {
        // Grasping requires: in volume AND touching both inside colliders AND NOT touching outside
        bool wasGrasping = isGrasping;
        isGrasping = isInGripVolume && leftInsideTouching && rightInsideTouching && !isTouchingOutside;

        // Update debug indicators
        if (GraspingIndicator != null && wasGrasping != isGrasping)
        {
            GraspingIndicator.SetActive(isGrasping);
        }

        if (CubeBetweenIndicator != null)
        {
            CubeBetweenIndicator.SetActive(isInGripVolume);
        }
    }

}
