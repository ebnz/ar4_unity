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

/// @task-name: TPickAndPlace
/// <summary>
/// RobotAgent for TPickAndPlace with direct joint control (no IK) including gripper control
/// </summary>
/// <taskdescription>
/// TPickAndPlace Task: Pick up a cube from a fixed location and transfer it to another fixed drop zone.
/// The agent directly controls all joints including the gripper to grasp and transport the cube.
/// Success is achieved when the cube is placed within the drop zone area.
/// </taskdescription>
/// <successcriteria>
/// Cube is positioned within 0.2 units of the drop zone center and gripper is open
/// </successcriteria>
/// <actionspace>
/// Continuous Actions (7): [j1_delta, j2_delta, j3_delta, j4_delta, j5_delta, j6_delta, gripper_delta]
/// </actionspace>
public class RobotAgentTPickAndPlaceDirectControl : Agent, IEurekaAgent
{
    [Header("References")]
    public ArticulationBody[] robotJoints;
    public Transform endEffector;
    public Transform dropZone;
    public Transform targetCube;  // Direct reference to cube instead of spawn location
    public ArticulationBody leftGripper;
    public ArticulationBody rightGripper;
    public GraspDetector graspDetector;  // Grasp detection system on the cube
    private GripperController gripperController;

    [Header("Settings")]
    [Tooltip("Speed multiplier for joint movements")]
    public float jointActionSpeed = 4f;

    [Tooltip("Speed multiplier for gripper movements")]
    public float gripperActionSpeed = 0.03f;

    private ObservationNormalizer observationNormalizer;

    [Header("Reward Tracking")]
    private Dictionary<string, float> rewardsByType = new Dictionary<string, float>();

    private Vector3 initialDropZonePosition;
    private Vector3 initialCubePosition;
    private int currentEpisodeSteps = 0;
    private float previousDistanceToCube = 0f;
    private float previousDistanceToDropZone = 0f;
    private Dictionary<string, float> currentObservations = new Dictionary<string, float>();
    private BehaviorParameters behaviorParameters;

    public BoxCollider TableTop;

    // Reference to stats logger for Eureka integration
    private TrainingStatsLogger statsLogger;

    /// <summary>
    /// Distance threshold for successful drop zone placement
    /// </summary>
    public float dropZoneThreshold = 0.2f;

    // Table collision tracking
    private bool isTouchingTable = false;
    private int tableCollisionCount = 0;

    // Store initial joint positions for reset
    private float[] initialJointPositions;
    private float initialGripperPosition;

    async void Start()
    {
        // Validate required references
        if (robotJoints == null || robotJoints.Length == 0)
        {
            Debug.LogError("RobotAgentTPickAndPlaceDirectControl: robotJoints not assigned!");
        }

        if (endEffector == null)
        {
            Debug.LogError("RobotAgentTPickAndPlaceDirectControl: endEffector not assigned!");
        }

        if (dropZone != null)
            initialDropZonePosition = dropZone.position;

        if (targetCube != null)
            initialCubePosition = targetCube.position;
        else
        {
            Debug.LogError("Target cube not assigned to RobotAgentTPickAndPlaceDirectControl!");
            throw new System.InvalidOperationException("Target cube must be assigned to RobotAgentTPickAndPlaceDirectControl!");
        }

        if (leftGripper == null || rightGripper == null)
        {
            Debug.LogError("RobotAgentTPickAndPlaceDirectControl: gripper references not assigned!");
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

        // Store initial joint positions for reset
        initialJointPositions = new float[robotJoints.Length];
        for (int i = 0; i < robotJoints.Length; i++)
        {
            initialJointPositions[i] = robotJoints[i].xDrive.target;
        }

        // Store initial gripper position
        if (leftGripper != null)
        {
            initialGripperPosition = leftGripper.xDrive.target;
        }

        statsLogger = FindFirstObjectByType<TrainingStatsLogger>();
        observationNormalizer = GetComponent<ObservationNormalizer>();
        gripperController = GetComponentInChildren<GripperController>();

        // Wait for RewardFunctionManager async initialization to complete
        Debug.Log("RobotAgentTPickAndPlaceDirectControl waiting for RewardFunctionManager initialization...");
        while (!RewardFunctionManager.IsInitialized)
        {
            await Task.Yield(); // Wait one frame
        }
        Debug.Log("RobotAgentTPickAndPlaceDirectControl: RewardFunctionManager ready, starting agent...");

        behaviorParameters = GetComponent<BehaviorParameters>();
    }


    public override void OnEpisodeBegin()
    {
        currentEpisodeSteps = 0;
        rewardsByType.Clear();
        isTouchingTable = false;
        tableCollisionCount = 0;

        // Reset all joint positions instantly
        for (int i = 0; i < robotJoints.Length; i++)
        {
            var joint = robotJoints[i];

            // Reset joint drive target
            var drive = joint.xDrive;
            drive.target = initialJointPositions[i];
            joint.xDrive = drive;

            // Reset velocities and forces
            joint.jointPosition = new ArticulationReducedSpace(0f);
            joint.jointForce = new ArticulationReducedSpace(0f);
            joint.jointVelocity = new ArticulationReducedSpace(0f);
        }

        // Reset grippers to open position using GripperController
        if (gripperController != null)
        {
            gripperController.ResetGrippers();
        }

        // Reset drop zone position
        dropZone.position = initialDropZonePosition;
        dropZone.localScale = new Vector3(dropZoneThreshold, 0.05f, dropZoneThreshold);

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

            // Reset GraspDetector state
            if (graspDetector != null)
            {
                graspDetector.ResetGraspState();
            }
        }

        // Initialize distance tracking
        if (leftGripper != null && rightGripper != null && targetCube != null)
        {
            Vector3 gripperCenter = (leftGripper.transform.position + rightGripper.transform.position) * 0.5f;
            previousDistanceToCube = Vector3.Distance(gripperCenter, targetCube.position);
            previousDistanceToDropZone = Vector3.Distance(targetCube.position, dropZone.position);
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        currentObservations.Clear();

        // Calculate gripper center position
        Vector3 gripperCenter = (leftGripper.transform.position + rightGripper.transform.position) * 0.5f;

        // Gripper center to cube vector
        Vector3 gripperToCube = targetCube.position - gripperCenter;
        currentObservations["EndEffectorToCube_X"] = gripperToCube.x; // @obs: X component of vector from gripper center to cube
        currentObservations["EndEffectorToCube_Y"] = gripperToCube.y; // @obs: Y component of vector from gripper center to cube
        currentObservations["EndEffectorToCube_Z"] = gripperToCube.z; // @obs: Z component of vector from gripper center to cube

        // Distance from gripper center to cube
        float gripperDistanceToCube = Vector3.Distance(gripperCenter, targetCube.position);
        currentObservations["DistanceToCube"] = gripperDistanceToCube; // @obs: Euclidean distance from gripper center to cube

        // Cube to drop zone vector
        Vector3 cubeToDropZone = dropZone.position - targetCube.position;
        currentObservations["CubeToDropZone_X"] = cubeToDropZone.x; // @obs: X component of vector from cube to drop zone
        currentObservations["CubeToDropZone_Y"] = cubeToDropZone.y; // @obs: Y component of vector from cube to drop zone
        currentObservations["CubeToDropZone_Z"] = cubeToDropZone.z; // @obs: Z component of vector from cube to drop zone

        // Distance from cube to drop zone
        float distanceToDropZone = Vector3.Distance(targetCube.position, dropZone.position);
        currentObservations["DistanceToDropZone"] = distanceToDropZone; // @obs: Euclidean distance from cube to drop zone

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

        // Table contact state
        currentObservations["TouchingTable"] = isTouchingTable ? 1f : 0f; // @obs: 1 if robot is touching the table, 0 if not

        // Gripper position (normalized)
        float gripperPosition = 0f;
        if (leftGripper != null)
        {
            var drive = leftGripper.xDrive;
            gripperPosition = (drive.target - drive.lowerLimit) / (drive.upperLimit - drive.lowerLimit) * 2f - 1f;
        }
        currentObservations["GripperPosition"] = gripperPosition; // @obs: Normalized gripper position (-1 to 1)

        // Current joint positions (normalized to [-1, 1])
        for (int i = 0; i < robotJoints.Length; i++)
        {
            var joint = robotJoints[i];
            var drive = joint.xDrive;
            float normalizedPosition = (drive.target - drive.lowerLimit) / (drive.upperLimit - drive.lowerLimit) * 2f - 1f;
            currentObservations[$"Joint{i}_Position"] = normalizedPosition; // @obs: Normalized joint position (-1 to 1)
        }

        // Joint velocities
        for (int i = 0; i < robotJoints.Length; i++)
        {
            float velocity = robotJoints[i].jointVelocity[0];
            currentObservations[$"Joint{i}_Velocity"] = velocity; // @obs: Joint angular velocity
        }

        // Collect raw observations
        List<float> rawObsList = new List<float>
        {
            gripperToCube.x,
            gripperToCube.y,
            gripperToCube.z,
            gripperDistanceToCube,
            cubeToDropZone.x,
            cubeToDropZone.y,
            cubeToDropZone.z,
            distanceToDropZone,
            currentObservations["IsGrasping"],
            currentObservations["IsInGripVolume"],
            currentObservations["IsTouchingOutside"],
            currentObservations["LeftInsideTouching"],
            currentObservations["RightInsideTouching"],
            currentObservations["TouchingTable"],
            currentObservations["GripperPosition"]
        };

        // Add joint positions
        for (int i = 0; i < robotJoints.Length; i++)
        {
            rawObsList.Add(currentObservations[$"Joint{i}_Position"]);
        }

        // Add joint velocities
        for (int i = 0; i < robotJoints.Length; i++)
        {
            rawObsList.Add(currentObservations[$"Joint{i}_Velocity"]);
        }

        float[] rawObs = rawObsList.ToArray();

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
        // Apply direct joint control (first 6 actions)
        for (int i = 0; i < robotJoints.Length && i < 6; i++)
        {
            var joint = robotJoints[i];
            var drive = joint.xDrive;

            // Apply action as joint angle delta
            float actionValue = actions.ContinuousActions[i];
            float jointDelta = actionValue * jointActionSpeed * Time.fixedDeltaTime;

            // Clamp to joint limits
            float newTarget = Mathf.Clamp(
                drive.target + jointDelta,
                drive.lowerLimit,
                drive.upperLimit
            );

            drive.target = newTarget;
            joint.xDrive = drive;
        }

        // Apply gripper control (7th action) using GripperController
        if (actions.ContinuousActions.Length > 6 && gripperController != null && leftGripper != null)
        {
            float gripperAction = actions.ContinuousActions[6];
            float gripperDelta = gripperAction * gripperActionSpeed * Time.fixedDeltaTime;

            // GripperController handles symmetric control of both grippers
            var drive = leftGripper.xDrive;
            gripperController.targetPosition += gripperDelta;
        }

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
            previousDistanceToCube = currentObservations["DistanceToCube"];
            previousDistanceToDropZone = currentObservations["DistanceToDropZone"];
        }

        if (currentEpisodeSteps >= ConfigManager.Instance.MaxStepsPerEpisode)
        {
            AddRewardWithType("Timeout", -1.0f);
            EndDynamicEpisode();
        }
    }

    private bool IsTaskComplete()
    {
        if (targetCube == null) return false;

        // Task is complete when cube is in drop zone and gripper is open (cube released)
        bool cubeInDropZone = currentObservations["DistanceToDropZone"] < dropZoneThreshold;
        bool cubeIsGrasped = currentObservations["IsGrasping"] == 1f;

        return cubeInDropZone && !cubeIsGrasped;
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

    private void ExecuteRewardFunction()
    {
        try
        {
            bool cubeIsGrasped = graspDetector != null && graspDetector.isGrasping;

            // Calculate cube lift height (Y difference from initial spawn position)
            float cubeLiftHeight = targetCube != null ? targetCube.position.y - initialCubePosition.y : 0f;

            var context = new Dictionary<string, object>
            {
                ["previousDistanceToCube"] = previousDistanceToCube, // @context: Gripper distance to cube in previous timestep
                ["previousDistanceToDropZone"] = previousDistanceToDropZone, // @context: Distance from cube to drop zone in previous timestep
                ["dropZoneThreshold"] = dropZoneThreshold, // @context: Distance threshold for successful placement (0.2f)
                ["episodeSteps"] = currentEpisodeSteps, // @context: Current number of steps in this episode
                ["maxSteps"] = ConfigManager.Instance.MaxStepsPerEpisode, // @context: Maximum steps per episode (2000)
                ["timeStep"] = Time.fixedDeltaTime, // @context: Fixed timestep duration (0.02f)
                ["endEffectorPosition"] = endEffector.position, // @context: Current end effector position (Vector3)
                ["dropZonePosition"] = dropZone.position, // @context: Drop zone position (Vector3)
                ["cubePosition"] = targetCube != null ? targetCube.position : Vector3.zero, // @context: Current cube position (Vector3)
                ["cubeSpawnPosition"] = initialCubePosition, // @context: Initial cube spawn position (Vector3)
                ["cubeGrasped"] = cubeIsGrasped, // @context: Whether cube is currently grasped (bool)
                ["isSuccess"] = IsTaskComplete(), // @context: Whether the task has been successfully completed (bool)
                ["cubeLiftHeight"] = cubeLiftHeight, // @context: Y difference from initial cube position, positive means lifted (float)
            };

            // Add joint positions to context
            for (int i = 0; i < robotJoints.Length; i++)
            {
                context[$"joint{i}Position"] = robotJoints[i].xDrive.target;
            }

            // Add gripper position
            if (leftGripper != null)
            {
                context["gripperPosition"] = leftGripper.xDrive.target;
            }

            // Execute the reward function synchronously
            var (totalReward, components) = RewardFunctionManager.ComputeReward(currentObservations, context);

            foreach (var kvp in components)
            {
                AddRewardWithType(kvp.Key, kvp.Value);
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

        // Joint controls
        continuousActionsOut[0] = Input.GetKey(KeyCode.A) ? -1.0f : (Input.GetKey(KeyCode.D) ? 1.0f : 0.0f);
        continuousActionsOut[1] = Input.GetKey(KeyCode.W) ? 1.0f : (Input.GetKey(KeyCode.S) ? -1.0f : 0.0f);
        continuousActionsOut[2] = Input.GetKey(KeyCode.Q) ? 1.0f : (Input.GetKey(KeyCode.E) ? -1.0f : 0.0f);
        continuousActionsOut[3] = Input.GetKey(KeyCode.UpArrow) ? 1.0f : (Input.GetKey(KeyCode.DownArrow) ? -1.0f : 0.0f);
        continuousActionsOut[4] = Input.GetKey(KeyCode.LeftArrow) ? -1.0f : (Input.GetKey(KeyCode.RightArrow) ? 1.0f : 0.0f);
        continuousActionsOut[5] = Input.GetKey(KeyCode.Z) ? 1.0f : (Input.GetKey(KeyCode.X) ? -1.0f : 0.0f);

        // Gripper control: G = close (negative), H = open (positive)
        continuousActionsOut[6] = Input.GetKey(KeyCode.G) ? -1.0f : (Input.GetKey(KeyCode.H) ? 1.0f : 0.0f);
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
    /// Called when a decision is requested (not on action repetitions)
    /// Increments episode step counter
    /// </summary>
    public void OnDecisionRequested()
    {
        currentEpisodeSteps++;
    }

    // Table collision tracking - called from TableCollisionDetector components on joints
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
