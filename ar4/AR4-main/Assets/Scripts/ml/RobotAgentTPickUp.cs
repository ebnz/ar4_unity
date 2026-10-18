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

/// @task-name: TPickUp
/// <summary>
/// RobotAgent for TPickUp with dynamic reward function evaluation via RewardFunctionManager
/// </summary>
/// <taskdescription>
/// TPickUp Task: Pick up a cube and hold it at y = 1.5 for 3 seconds.
/// The agent controls the target position that the inverse kinematics system follows and gripper open/close actions.
/// Success is achieved when the cube is grasped and held at the target height for the required duration.
/// </taskdescription>
/// <successcriteria>
/// Cube is grasped, positioned at y = 1.5 (within tolerance), and held there for 3 seconds
/// </successcriteria>
/// <actionspace>
/// Continuous Actions (3): [x_delta, y_delta, z_delta] - Target position deltas
/// Discrete Actions (1): [gripper_action] where 0=no_action, 1=open_gripper, 2=close_gripper
/// </actionspace>
public class RobotAgentTPickUp : Agent, IEurekaAgent
{
    [Header("References")]
    public RobotManager robotManager;
    public Transform targetCube;  // Direct reference to cube instead of spawn location
    public GripperController gripperController;
    public GraspDetector graspDetector;  // Grasp detection system on the cube
    public BoxCollider TableTop;

    [Header("Settings")]
    private Vector3 smoothedAction;
    private bool gripperOpen = true;
    private ObservationNormalizer observationNormalizer;

    [Header("Reward Tracking")]
    private Dictionary<string, float> rewardsByType = new Dictionary<string, float>();

    private Vector3 initialCubePosition;
    private int currentEpisodeSteps = 0;
    private float previousDistanceToCube = 0f;
    private float previousTargetDistanceToCube = 0f;
    private float previousCubeHeight = 0f;
    private Dictionary<string, float> currentObservations = new Dictionary<string, float>();
    private BehaviorParameters behaviorParameters;

    // Track current gripper action for contextual rewards
    private int currentGripperAction = -1; // 0=no_action, 1=open, 2=close


    // Reference to stats logger for Eureka integration
    private TrainingStatsLogger statsLogger;

    [Header("Debug Visualization")]
    [SerializeField] private bool _showIKTarget = true;
    public bool ShowIKTarget
    {
        get => _showIKTarget;
        set
        {
            _showIKTarget = value;
            if (robotManager != null && robotManager.target != null)
            {
                robotManager.target.gameObject.SetActive(value);
            }
        }
    }

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

    /// <summary>
    /// Current time cube has been held at target height
    /// </summary>
    private float currentHoldTime = 0f;

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
            Debug.LogError("Target cube not assigned to RobotAgentPickUp!");
            throw new System.InvalidOperationException("Target cube must be assigned to RobotAgentPickUp!");
        }

        // Initialize GraspDetector from cube
        if (graspDetector == null && targetCube != null)
        {
            graspDetector = targetCube.GetComponent<GraspDetector>();
            if (graspDetector == null)
            {
                Debug.LogWarning("GraspDetector not found on target cube. Grasp detection will not work correctly.");
            }
        }

        statsLogger = FindFirstObjectByType<TrainingStatsLogger>();
        observationNormalizer = GetComponent<ObservationNormalizer>();

        // Wait for RewardFunctionManager async initialization to complete
        Debug.Log("RobotAgentPickUp waiting for RewardFunctionManager initialization...");
        while (!RewardFunctionManager.IsInitialized)
        {
            await Task.Yield(); // Wait one frame
        }
        Debug.Log("RobotAgentPickUp: RewardFunctionManager ready, starting agent...");

        behaviorParameters = GetComponent<BehaviorParameters>();

        // Apply IK target visibility setting
        if (robotManager != null && robotManager.target != null)
        {
            robotManager.target.gameObject.SetActive(_showIKTarget);
        }
    }


    public override void OnEpisodeBegin()
    {
        currentEpisodeSteps = 0;
        rewardsByType.Clear();
        gripperOpen = true;
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

            // Reset GraspDetector state and grip volume position
            if (graspDetector != null)
            {
                graspDetector.ResetGraspState();
            }
        }

        // Initialize distance tracking
        if (gripperController != null && targetCube != null)
        {
            Vector3 gripperCenter = (gripperController.leftGripper.transform.position + gripperController.rightGripper.transform.position) * 0.5f;
            previousDistanceToCube = Vector3.Distance(gripperCenter, targetCube.position);
            previousTargetDistanceToCube = Vector3.Distance(robotManager.target.position, targetCube.position);
            previousCubeHeight = targetCube.position.y;
        }
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

        // GraspDetector observations - accurate grasp state from physics collisions
        bool isGrasping = graspDetector != null && graspDetector.isGrasping;
        bool isInGripVolume = graspDetector != null && graspDetector.isInGripVolume;
        bool isTouchingOutside = graspDetector != null && graspDetector.isTouchingOutside;
        bool leftInsideTouching = graspDetector != null && graspDetector.leftInsideTouching;
        bool rightInsideTouching = graspDetector != null && graspDetector.rightInsideTouching;

        currentObservations["IsGrasping"] = isGrasping ? 1f : 0f; // @obs: 1 if cube is being grasped (in volume, touching both inside, not touching outside), 0 otherwise
        currentObservations["IsInGripVolume"] = isInGripVolume ? 1f : 0f; // @obs: 1 if cube is in the grip volume trigger, 0 otherwise
        currentObservations["IsTouchingOutside"] = isTouchingOutside ? 1f : 0f; // @obs: 1 if cube is touching outside of grippers, 0 otherwise
        currentObservations["LeftInsideTouching"] = leftInsideTouching ? 1f : 0f; // @obs: 1 if left inside gripper is touching cube, 0 otherwise
        currentObservations["RightInsideTouching"] = rightInsideTouching ? 1f : 0f; // @obs: 1 if right inside gripper is touching cube, 0 otherwise

        // Cube height information
        currentObservations["CubeHeight"] = targetCube.position.y; // @obs: Current Y position (height) of the cube

        // Height difference from target
        float heightDifference = Mathf.Abs(targetCube.position.y - targetHeight);
        currentObservations["HeightDifference"] = heightDifference; // @obs: Absolute difference between cube height and target height (1.5)

        // Is cube at target height
        bool isAtTargetHeight = isGrasping && heightDifference < heightTolerance;
        currentObservations["IsAtTargetHeight"] = isAtTargetHeight ? 1f : 0f; // @obs: 1 if cube is at target height (within tolerance), 0 otherwise

        // Hold time tracking
        currentObservations["HoldTime"] = currentHoldTime; // @obs: Current time (in seconds) cube has been held at target height

        // Hold progress (0 to 1)
        currentObservations["HoldProgress"] = Mathf.Clamp01(currentHoldTime / holdDuration); // @obs: Progress toward hold duration goal (0 to 1)

        // Gripper state - sync with actual controller if available
        if (gripperController != null)
        {
            // Consider gripper open if position is above a threshold (close to upper limit)
            float normalizedPosition = gripperController.currentPosition / gripperController.leftGripper.xDrive.upperLimit;
            gripperOpen = normalizedPosition > 0.5f; // Threshold for considering gripper "open"
        }
        currentObservations["GripperOpen"] = gripperOpen ? 1f : 0f; // @obs: 1 if gripper is open, 0 if closed

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
            currentObservations["IsGrasping"],
            currentObservations["IsInGripVolume"],
            currentObservations["IsTouchingOutside"],
            currentObservations["LeftInsideTouching"],
            currentObservations["RightInsideTouching"],
            currentObservations["CubeHeight"],
            currentObservations["HeightDifference"],
            currentObservations["IsAtTargetHeight"],
            currentObservations["HoldTime"],
            currentObservations["HoldProgress"],
            currentObservations["GripperOpen"],
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

        // Gripper action (discrete: 0 = no action, 1 = open gripper, 2 = close gripper)
        currentGripperAction = actions.DiscreteActions[0];

        if (currentGripperAction == 1)
        {
            // Open gripper (only if currently closed)
            gripperOpen = true;
            if (gripperController != null)
            {
                gripperController.OpenGrippers();
            }
        }
        else if (currentGripperAction == 2)
        {
            // Close gripper (only if currently open)
            gripperOpen = false;
            if (gripperController != null)
            {
                gripperController.CloseGrippers();
            }
        }

        smoothedAction = Vector3.Lerp(smoothedAction, targetDeltaAction, ConfigManager.Instance.ActionSmoothing);
        Vector3 targetDelta = smoothedAction * ConfigManager.Instance.ActionSpeed * Time.fixedDeltaTime;
        robotManager.target.position += targetDelta;

        // Clamp target Y position to prevent going below table surface
        Vector3 clampedPos = robotManager.target.position;
        clampedPos.y = Mathf.Max(clampedPos.y, 0.93f);
        robotManager.target.position = clampedPos;

        // Update hold time tracking
        UpdateHoldTime();

        // Execute dynamic reward function
        if (!behaviorParameters.IsInHeuristicMode() && RewardFunctionManager.IsInitialized)
        {
            ExecuteRewardFunction();
        }

        // Check if cube has fallen off the table
        if (targetCube != null && targetCube.position.y < 0.5f)
        {
            AddRewardWithType("CubeFallen", -10.0f);
            EndDynamicEpisode();
            return;
        }

        // Check if task completed successfully
        if (IsTaskComplete())
        {
            // Success reward is now handled by the dynamic reward function
            EndDynamicEpisode();
        }

        // Update previous distances for next frame
        if (targetCube != null)
        {
            previousDistanceToCube = currentObservations["GripperDistanceToCube"];
            previousTargetDistanceToCube = currentObservations["TargetDistanceToCube"];
            previousCubeHeight = targetCube.position.y;
        }

        if (currentEpisodeSteps >= ConfigManager.Instance.MaxStepsPerEpisode)
        {
            AddRewardWithType("Timeout", -1.0f);
            EndDynamicEpisode();
        }
    }


    private void UpdateHoldTime()
    {
        if (targetCube == null) return;

        // Check if cube is at target height
        float heightDifference = Mathf.Abs(targetCube.position.y - targetHeight);
        bool cubeIsGrasped = graspDetector != null && graspDetector.isGrasping;
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
        bool cubeIsGrasped = graspDetector != null && graspDetector.isGrasping;
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
                ["gripperOpen"] = gripperOpen, // @context: Whether gripper is currently open (bool)
                ["cubeGrasped"] = graspDetector != null && graspDetector.isGrasping, // @context: Whether cube is currently grasped (bool)
                ["isSuccess"] = IsTaskComplete(), // @context: Whether the task has been successfully completed (bool)
                ["gripperAction"] = currentGripperAction, // @context: Current gripper action (0=no_action, 1=open, 2=close)
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
        var discreteActionsOut = actionsOut.DiscreteActions;

        // Movement controls
        continuousActionsOut[0] = -Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetKey(KeyCode.E) ? 1.0f : (Input.GetKey(KeyCode.Q) ? -1.0f : 0.0f);
        continuousActionsOut[2] = -Input.GetAxis("Vertical");

        // Gripper control (G = open, H = close)
        if (Input.GetKey(KeyCode.G))
        {
            discreteActionsOut[0] = 1; // Open gripper
        }
        else if (Input.GetKey(KeyCode.H))
        {
            discreteActionsOut[0] = 2; // Close gripper
        }
        else
        {
            discreteActionsOut[0] = 0; // No action
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
    public bool IsCubeGrasped() => graspDetector != null && graspDetector.isGrasping;

    /// <summary>
    /// Check if gripper is currently open
    /// </summary>
    public bool IsGripperOpen() => gripperOpen;

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

    private void OnValidate()
    {
        // Apply IK target visibility when changed in inspector
        if (robotManager != null && robotManager.target != null)
        {
            robotManager.target.gameObject.SetActive(_showIKTarget);
        }
    }

    [Header("Gizmo Settings")]
    [SerializeField] private bool _showTargetZone = true;

    private void OnDrawGizmos()
    {
        if (!_showTargetZone || TableTop == null) return;

        Bounds tableBounds = TableTop.bounds;
        Vector3 center = new Vector3(tableBounds.center.x, targetHeight, tableBounds.center.z);
        Vector3 size = new Vector3(tableBounds.size.x, heightTolerance * 2f, tableBounds.size.z);

        // Semi-transparent filled cube
        Gizmos.color = new Color(0f, 1f, 0.5f, 0.5f);
        Gizmos.DrawCube(center, size);

        // Wire cube outline
        Gizmos.color = new Color(0f, 1f, 0.5f, 1f);
        Gizmos.DrawWireCube(center, size);
    }
}
