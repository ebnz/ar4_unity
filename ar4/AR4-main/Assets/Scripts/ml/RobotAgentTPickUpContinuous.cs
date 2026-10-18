using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using System.Threading.Tasks;
using System;
using System.IO;
using System.Linq;
using Unity.MLAgents.Policies;

/// @task-name: TPickUpContinuous
/// <summary>
/// RobotAgent for TPickUp with continuous gripper control
/// </summary>
/// <taskdescription>
/// TPickUp Task: Pick up a cube and hold it at y = 1.5 for 3 seconds.
/// The agent controls the target position that the inverse kinematics system follows and continuous gripper position.
/// Success is achieved when the cube is grasped and held at the target height for the required duration.
/// </taskdescription>
/// <successcriteria>
/// Cube is grasped, positioned at y = 1.5 (within tolerance), and held there for 3 seconds
/// </successcriteria>
/// <actionspace>
/// Continuous Actions (4): [x_delta, y_delta, z_delta, gripper_position]
/// where gripper_position ranges from -1 (fully open) to 1 (fully closed)
/// </actionspace>
public class RobotAgentTPickUpContinuous : Agent, IEurekaAgent
{
    [Header("References")]
    public RobotManager robotManager;
    public Transform targetCube;  // Direct reference to cube instead of spawn location
    public GripperController gripperController;

    [Header("Settings")]
    private Vector3 smoothedAction;
    private float targetGripperPosition = -1f; // -1 = open, 1 = closed
    private ObservationNormalizer observationNormalizer;

    [Header("Task Objects")]
    private bool cubeIsGrasped = false;

    [Header("Reward Tracking")]
    private Dictionary<string, float> rewardsByType = new Dictionary<string, float>();

    private Vector3 initialCubePosition;
    private int currentEpisodeSteps = 0;
    private float previousDistanceToCube = 0f;
    private float previousTargetDistanceToCube = 0f;
    private float previousCubeHeight = 0f;
    private float previousGripperPosition = -1f;
    private Dictionary<string, float> currentObservations = new Dictionary<string, float>();
    private BehaviorParameters behaviorParameters;

    // Reference to stats logger for Eureka integration
    private TrainingStatsLogger statsLogger;

    /// <summary>
    /// Distance threshold for cube grasping detection
    /// </summary>
    public float graspThreshold = 0.15f;

    /// <summary>
    /// Target height for holding the cube
    /// </summary>
    public float targetHeight = 1.5f;

    /// <summary>
    /// Required duration to hold cube at target height (in seconds)
    /// </summary>
    public float holdDuration = 3.0f;

    /// <summary>
    /// Tolerance for height matching (cube must be within this range of targetHeight)
    /// </summary>
    public float heightTolerance = 0.1f;

    [Header("Debug")]
    [Tooltip("GameObject to activate when cube is grasped (for debugging)")]
    public GameObject graspDebugIndicator;

    /// <summary>
    /// Current time cube has been held at target height
    /// </summary>
    private float currentHoldTime = 0f;

    // Velocity tracking for rigid motion detection
    private Vector3 previousGripperCenter;
    private Vector3 previousCubePosition;

    // Table collision tracking
    private bool isTouchingTable = false;

    async void Start()
    {
        if (robotManager == null)
            robotManager = FindFirstObjectByType<RobotManager>();

        if (gripperController == null)
            gripperController = FindFirstObjectByType<GripperController>();

        if (targetCube != null)
            initialCubePosition = targetCube.position;
        else
        {
            Debug.LogError("Target cube not assigned to RobotAgentTPickUpContinuous!");
            throw new System.InvalidOperationException("Target cube must be assigned to RobotAgentTPickUpContinuous!");
        }

        statsLogger = FindFirstObjectByType<TrainingStatsLogger>();
        observationNormalizer = GetComponent<ObservationNormalizer>();

        // Wait for RewardFunctionManager async initialization to complete
        Debug.Log("RobotAgentTPickUpContinuous waiting for RewardFunctionManager initialization...");
        while (!RewardFunctionManager.IsInitialized)
        {
            await Task.Yield(); // Wait one frame
        }
        Debug.Log("RobotAgentTPickUpContinuous: RewardFunctionManager ready, starting agent...");

        behaviorParameters = GetComponent<BehaviorParameters>();
    }


    public override void OnEpisodeBegin()
    {
        currentEpisodeSteps = 0;
        rewardsByType.Clear();
        cubeIsGrasped = false;
        targetGripperPosition = -1f; // Start with gripper open
        currentHoldTime = 0f;
        isTouchingTable = false;
        tableCollisionCount = 0;

        // Reset gripper controller to open state
        if (gripperController != null)
        {
            gripperController.OpenGrippers();
        }

        robotManager.ResetRobotInstantly();

        // Reset cube position and physics
        if (targetCube != null)
        {
            targetCube.position = initialCubePosition;
            targetCube.rotation = Quaternion.identity;

            // Ensure cube is detached from any parent
            targetCube.SetParent(null);

            // Reset cube physics if it has a rigidbody
            if (targetCube.TryGetComponent<Rigidbody>(out var rb))
            {
                rb.isKinematic = false;
                rb.linearVelocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
            }
        }

        // Initialize velocity tracking and distance
        if (gripperController != null && targetCube != null)
        {
            Vector3 gripperCenter = (gripperController.leftGripper.transform.position + gripperController.rightGripper.transform.position) * 0.5f;
            previousGripperCenter = gripperCenter;
            previousDistanceToCube = Vector3.Distance(gripperCenter, targetCube.position);
            previousTargetDistanceToCube = Vector3.Distance(robotManager.target.position, targetCube.position);
            previousCubeHeight = targetCube.position.y;
            previousGripperPosition = -1f;
        }
        previousCubePosition = targetCube.position;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        currentObservations.Clear();

        currentObservations["RobotHeight"] = robotManager.target.position.y; // @obs: Current Y position (height) of the robot

        // Calculate gripper center position
        Vector3 gripperCenter = (gripperController.leftGripper.transform.position + gripperController.rightGripper.transform.position) * 0.5f;

        // Gripper center to cube vector
        Vector3 gripperToCube = targetCube.position - gripperCenter;
        currentObservations["GripperToCube_X"] = gripperToCube.x; // @obs: X component of vector from gripper center to cube
        currentObservations["GripperToCube_Y"] = gripperToCube.y; // @obs: Y component of vector from gripper center to cube
        currentObservations["GripperToCube_Z"] = gripperToCube.z; // @obs: Z component of vector from gripper center to cube

        // Distance from gripper center to cube
        float gripperDistanceToCube = Vector3.Distance(gripperCenter, targetCube.position);
        currentObservations["GripperDistanceToCube"] = gripperDistanceToCube; // @obs: Euclidean distance from gripper center to cube

        // IK target to cube (to prevent exploitation of moving target away from cube)
        Vector3 targetToCube = targetCube.position - robotManager.target.position;
        currentObservations["TargetToCube_X"] = targetToCube.x; // @obs: X component of vector from IK target to cube
        currentObservations["TargetToCube_Y"] = targetToCube.y; // @obs: Y component of vector from IK target to cube
        currentObservations["TargetToCube_Z"] = targetToCube.z; // @obs: Z component of vector from IK target to cube

        float targetDistanceToCube = Vector3.Distance(robotManager.target.position, targetCube.position);
        currentObservations["TargetDistanceToCube"] = targetDistanceToCube; // @obs: Distance from IK target to cube

        // Distance between IK target and actual gripper (to detect IK solving issues)
        float targetToGripperDistance = Vector3.Distance(robotManager.target.position, gripperCenter);
        currentObservations["TargetToGripperDistance"] = targetToGripperDistance; // @obs: Distance between IK target and actual gripper position

        // Check if cube is positioned between grippers (ready to grasp)
        // Transform cube position to gripper local space
        Vector3 cubeToGripperCenter = targetCube.position - gripperCenter;
        Vector3 leftToRight = (gripperController.rightGripper.transform.position - gripperController.leftGripper.transform.position).normalized;
        Vector3 gripperForward = gripperController.leftGripper.transform.up; // Assuming grippers face upward

        // Project cube position onto gripper coordinate frame
        float lateralOffset = Mathf.Abs(Vector3.Dot(cubeToGripperCenter, leftToRight)); // Distance along gripper opening axis
        float forwardOffset = Mathf.Abs(Vector3.Dot(cubeToGripperCenter, gripperForward)); // Distance in front/behind

        // Cube is "graspable" if it's centered between grippers (small lateral offset) and in front
        bool cubeInGraspPosition = (lateralOffset < 0.05f) && (forwardOffset < 0.08f) && (gripperDistanceToCube < graspThreshold);
        currentObservations["CubeInGraspPosition"] = cubeInGraspPosition ? 1f : 0f; // @obs: 1 if cube is positioned to be grasped, 0 otherwise

        // Cube height information
        currentObservations["CubeHeight"] = targetCube.position.y; // @obs: Current Y position (height) of the cube

        // Height difference from target
        float heightDifference = Mathf.Abs(targetCube.position.y - targetHeight);
        currentObservations["HeightDifference"] = heightDifference; // @obs: Absolute difference between cube height and target height (1.5)

        // Is cube at target height
        bool isAtTargetHeight = cubeIsGrasped && heightDifference < heightTolerance;
        currentObservations["IsAtTargetHeight"] = isAtTargetHeight ? 1f : 0f; // @obs: 1 if cube is at target height (within tolerance), 0 otherwise

        // Hold time tracking
        currentObservations["HoldTime"] = currentHoldTime; // @obs: Current time (in seconds) cube has been held at target height

        // Hold progress (0 to 1)
        currentObservations["HoldProgress"] = Mathf.Clamp01(currentHoldTime / holdDuration); // @obs: Progress toward hold duration goal (0 to 1)

        // Gripper position (continuous: -1 = open, 1 = closed)
        float normalizedGripperPosition = (gripperController.currentPosition / gripperController.leftGripper.xDrive.upperLimit) * 2f - 1f;
        currentObservations["GripperPosition"] = normalizedGripperPosition; // @obs: Current gripper position (-1 = fully open, 1 = fully closed)

        // Cube grasped state
        currentObservations["CubeGrasped"] = cubeIsGrasped ? 1f : 0f; // @obs: 1 if cube is grasped, 0 if not

        // Table contact state
        currentObservations["TouchingTable"] = isTouchingTable ? 1f : 0f; // @obs: 1 if robot is touching the table, 0 if not

        // Collect raw observations
        float[] rawObs = new float[] {
            currentObservations["GripperToCube_X"],
            currentObservations["GripperToCube_Y"],
            currentObservations["GripperToCube_Z"],
            currentObservations["GripperDistanceToCube"],
            currentObservations["TargetToCube_X"],
            currentObservations["TargetToCube_Y"],
            currentObservations["TargetToCube_Z"],
            currentObservations["TargetDistanceToCube"],
            currentObservations["TargetToGripperDistance"],
            currentObservations["CubeInGraspPosition"],
            currentObservations["CubeHeight"],
            currentObservations["HeightDifference"],
            currentObservations["IsAtTargetHeight"],
            currentObservations["HoldTime"],
            currentObservations["HoldProgress"],
            currentObservations["GripperPosition"],
            currentObservations["CubeGrasped"],
            currentObservations["TouchingTable"],
            currentObservations["RobotHeight"]
        };

        // Apply normalization if in inference mode
        float[] observations = observationNormalizer != null ? observationNormalizer.Normalize(rawObs) : rawObs;

        // Add observations to sensor
        foreach (float obs in observations)
        {
            sensor.AddObservation(obs);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Apply movement actions
        Vector3 targetDeltaAction = new Vector3(
            actions.ContinuousActions[0],
            actions.ContinuousActions[1],
            actions.ContinuousActions[2]
        );

        // Gripper action (continuous: -1 = open, 1 = close)
        targetGripperPosition = actions.ContinuousActions[3];

        // Apply gripper position smoothly
        if (gripperController != null)
        {
            // Map from [-1, 1] to [0, upperLimit]
            float normalizedPosition = (targetGripperPosition + 1f) * 0.5f; // [0, 1]
            float gripperPosition = normalizedPosition * gripperController.leftGripper.xDrive.upperLimit;
            gripperController.SetGripperPosition(gripperPosition);
        }

        smoothedAction = Vector3.Lerp(smoothedAction, targetDeltaAction, ConfigManager.Instance.ActionSmoothing);
        Vector3 targetDelta = smoothedAction * ConfigManager.Instance.ActionSpeed * Time.fixedDeltaTime;
        robotManager.target.position += targetDelta;

        // Handle cube grasping/releasing
        UpdateCubeGrasping();

        // Update hold time tracking
        UpdateHoldTime();

        // Execute dynamic reward function
        if (!behaviorParameters.IsInHeuristicMode() && RewardFunctionManager.IsInitialized)
        {
            ExecuteRewardFunction();
        }

        // Check if task completed successfully
        if (IsTaskComplete())
        {
            // Success reward is now handled by the dynamic reward function
            EndDynamicEpisode();
        }

        // Update previous values for next frame
        if (targetCube != null)
        {
            previousDistanceToCube = currentObservations["GripperDistanceToCube"];
            previousTargetDistanceToCube = currentObservations["TargetDistanceToCube"];
            previousCubeHeight = targetCube.position.y;
            previousGripperPosition = currentObservations["GripperPosition"];
        }

        // Update velocity tracking positions
        if (gripperController != null)
        {
            previousGripperCenter = (gripperController.leftGripper.transform.position + gripperController.rightGripper.transform.position) * 0.5f;
        }
        previousCubePosition = targetCube.position;

        if (currentEpisodeSteps >= ConfigManager.Instance.MaxStepsPerEpisode)
        {
            AddRewardWithType("Timeout", -1.0f);
            EndDynamicEpisode();
        }
    }

    private void UpdateCubeGrasping()
    {
        if (targetCube == null || gripperController == null) return;

        // Simple grasp detection based on cube height
        bool previousGraspState = cubeIsGrasped;
        cubeIsGrasped = targetCube.position.y > 1.0f;

        // Update debug indicator
        if (graspDebugIndicator != null)
        {
            graspDebugIndicator.SetActive(cubeIsGrasped);
        }

        // Reset hold time if cube is no longer grasped
        if (!cubeIsGrasped)
        {
            currentHoldTime = 0f;
        }
    }

    private void UpdateHoldTime()
    {
        if (targetCube == null) return;

        // Check if cube is at target height
        float heightDifference = Mathf.Abs(targetCube.position.y - targetHeight);
        bool isAtTargetHeight = cubeIsGrasped && heightDifference < heightTolerance;

        if (isAtTargetHeight)
        {
            // Increment hold time
            currentHoldTime += Time.fixedDeltaTime;
        }
        else
        {
            // Reset hold time if not at target height
            currentHoldTime = 0f;
        }
    }

    private bool IsTaskComplete()
    {
        if (targetCube == null) return false;

        // Task is complete when cube is grasped, at target height, and held for required duration
        float heightDifference = Mathf.Abs(targetCube.position.y - targetHeight);
        bool isAtTargetHeight = heightDifference < heightTolerance;
        bool hasBeenHeldLongEnough = currentHoldTime >= holdDuration;

        return cubeIsGrasped && isAtTargetHeight && hasBeenHeldLongEnough;
    }

    // Not an override; use a custom method name
    public void EndDynamicEpisode()
    {
        // Determine if episode was successful based on task completion
        bool isSuccess = IsTaskComplete();

        // Track episode statistics via TrainingStatsLogger
        if (statsLogger != null)
        {
            float episodeReward = rewardsByType.Values.Sum();
            statsLogger.OnDynamicAgentEpisodeEnd(this, episodeReward, currentEpisodeSteps, new Dictionary<string, float>(rewardsByType), isSuccess);
        }

        base.EndEpisode();
    }

    private async void ExecuteRewardFunction()
    {
        try
        {
            var context = new Dictionary<string, object>
            {
                ["previousDistanceToCube"] = previousDistanceToCube, // @context: Gripper distance to cube in previous timestep
                ["previousTargetDistanceToCube"] = previousTargetDistanceToCube, // @context: IK target distance to cube in previous timestep
                ["previousCubeHeight"] = previousCubeHeight, // @context: Cube height in previous timestep
                ["previousGripperPosition"] = previousGripperPosition, // @context: Gripper position in previous timestep (-1 to 1)
                ["graspThreshold"] = graspThreshold, // @context: Distance threshold for grasping cube (0.15f)
                ["targetHeight"] = targetHeight, // @context: Target height for holding cube (1.5f)
                ["heightTolerance"] = heightTolerance, // @context: Tolerance for height matching (0.1f)
                ["holdDuration"] = holdDuration, // @context: Required hold duration in seconds (3.0f)
                ["currentHoldTime"] = currentHoldTime, // @context: Current time cube has been held at target height
                ["episodeSteps"] = currentEpisodeSteps, // @context: Current number of steps in this episode
                ["maxSteps"] = ConfigManager.Instance.MaxStepsPerEpisode, // @context: Maximum steps per episode (2000)
                ["timeStep"] = Time.fixedDeltaTime, // @context: Fixed timestep duration (0.02f)
                ["targetPosition"] = robotManager.target.position, // @context: Current target position (Vector3)
                ["cubePosition"] = targetCube != null ? targetCube.position : Vector3.zero, // @context: Current cube position (Vector3)
                ["cubeSpawnPosition"] = initialCubePosition, // @context: Initial cube spawn position (Vector3)
                ["gripperPosition"] = targetGripperPosition, // @context: Current gripper position (-1 = open, 1 = closed)
                ["cubeGrasped"] = cubeIsGrasped, // @context: Whether cube is currently grasped (bool)
                ["isSuccess"] = IsTaskComplete(), // @context: Whether the task has been successfully completed (bool)
            };

            // Execute the reward function via the manager
            var result = await RewardFunctionManager.ExecuteRewardFunction(currentObservations, context);

            if (result != null)
            {
                // Apply rewards
                foreach (var kvp in result.Components)
                {
                    AddRewardWithType(kvp.Key, kvp.Value);
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error executing reward function: {e.Message}");
        }
    }

    private void AddRewardWithType(string rewardType, float reward)
    {
        if (!rewardsByType.ContainsKey(rewardType))
        {
            rewardsByType[rewardType] = 0f;
        }

        rewardsByType[rewardType] += reward;
        base.AddReward(reward);
    }

    /// <summary>
    /// Get current observations for external tracking
    /// </summary>
    public Dictionary<string, float> GetCurrentObservations()
    {
        return new Dictionary<string, float>(currentObservations);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;

        // Movement controls
        continuousActionsOut[0] = -Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetKey(KeyCode.E) ? 1.0f : (Input.GetKey(KeyCode.Q) ? -1.0f : 0.0f);
        continuousActionsOut[2] = -Input.GetAxis("Vertical");

        // Gripper control (G = open, H = close)
        if (Input.GetKey(KeyCode.G))
        {
            continuousActionsOut[3] = -1f; // Open gripper
        }
        else if (Input.GetKey(KeyCode.H))
        {
            continuousActionsOut[3] = 1f; // Close gripper
        }
        else
        {
            continuousActionsOut[3] = 0f; // Hold current position
        }
    }

    // Recompile button for testing in Inspector
    [ContextMenu("Reload Reward Function")]
    public void ReloadReward()
    {
        if (ConfigManager.Instance.UseFileBasedReward && !string.IsNullOrEmpty(ConfigManager.Instance.RewardPath))
        {
            // For file-based rewards, we need to reload from file
            Debug.LogWarning("File-based reward reloading not implemented yet. Use RewardFunctionManagerInit for inspector rewards.");
        }
        else
        {
            Debug.LogWarning("Use RewardFunctionManagerInit component to reload inspector-based rewards.");
        }
    }

    // Public getters for visualizer
    public int GetCurrentEpisodeSteps() => currentEpisodeSteps;
    public int GetMaxStepsPerEpisode() => ConfigManager.Instance.MaxStepsPerEpisode;
    public Dictionary<string, float> GetRewardsByType() => new Dictionary<string, float>(rewardsByType);

    /// <summary>
    /// Get current cube reference for external systems
    /// </summary>
    public GameObject GetCurrentCube() => targetCube?.gameObject;

    /// <summary>
    /// Check if cube is currently grasped
    /// </summary>
    public bool IsCubeGrasped() => cubeIsGrasped;

    /// <summary>
    /// Get current gripper position (-1 = open, 1 = closed)
    /// </summary>
    public float GetGripperPosition() => targetGripperPosition;

    /// <summary>
    /// Get current hold time at target height
    /// </summary>
    public float GetCurrentHoldTime() => currentHoldTime;

    /// <summary>
    /// Called when a decision is requested (not on action repetitions)
    /// Increments episode step counter
    /// </summary>
    public void OnDecisionRequested()
    {
        currentEpisodeSteps++;
    }

    // Table collision tracking - called from TableCollisionDetector components on joints
    private int tableCollisionCount = 0;

    public void OnTableCollisionEnter(Collision collision)
    {
        tableCollisionCount++;
        isTouchingTable = tableCollisionCount > 0;
    }

    public void OnTableCollisionStay(Collision collision)
    {
        isTouchingTable = tableCollisionCount > 0;
    }

    public void OnTableCollisionExit(Collision collision)
    {
        tableCollisionCount = Mathf.Max(0, tableCollisionCount - 1);
        isTouchingTable = tableCollisionCount > 0;
    }
}
