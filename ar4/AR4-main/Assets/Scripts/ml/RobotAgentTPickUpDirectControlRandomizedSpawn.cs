using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using System.Threading.Tasks;
using System;
using System.Linq;
using Unity.MLAgents.Policies;

/// @task-name: TPickUpDirectControlRandomizedSpawn
/// <summary>
/// RobotAgent for TPickUp with direct joint control (no IK) including gripper control, and randomizable cube spawn position
/// </summary>
/// <taskdescription>
/// TPickUp Task: Pick up a cube placed at a randomized X/Z spawn position and hold it at y = 1.5 for 3 seconds.
/// The agent directly controls all joints including the gripper to grasp and lift the cube.
/// Success is achieved when the cube is grasped and held at the target height for the required duration.
/// </taskdescription>
/// <successcriteria>
/// Cube is grasped, positioned at y = 1.5 (within tolerance), and held there for 3 seconds
/// </successcriteria>
/// <actionspace>
/// Continuous Actions (7): [j1_delta, j2_delta, j3_delta, j4_delta, j5_delta, j6_delta, gripper_delta]
/// </actionspace>
public class RobotAgentTPickUpDirectControlRandomizedSpawn : Agent, IEurekaAgent
{
    [Header("References")]
    public ArticulationBody[] robotJoints;
    public Transform endEffector;
    public Transform targetCube;
    public ArticulationBody leftGripper;
    public ArticulationBody rightGripper;
    public GraspDetector graspDetector;  // Grasp detection system on the cube

    [Header("Settings")]
    [Tooltip("Speed multiplier for joint movements")]
    public float jointActionSpeed = 50f;

    [Tooltip("Speed multiplier for gripper movements")]
    public float gripperActionSpeed = 30f;

    private ObservationNormalizer observationNormalizer;

    [Header("Task Settings")]
    [Tooltip("Distance threshold for cube grasping detection")]
    public float graspThreshold = 0.15f;

    [Tooltip("Target height for holding the cube")]
    public float targetHeight = 1.5f;

    [Tooltip("Required duration to hold cube at target height (in seconds)")]
    public float holdDuration = 3.0f;

    [Tooltip("Tolerance for height matching")]
    public float heightTolerance = 0.1f;

    [Header("Spawn Randomization")]
    [Tooltip("Enable random spawn position within defined bounds")]
    public bool randomizeSpawnPosition = false;

    [Tooltip("Minimum bounds for spawn area (X and Z are used, Y is ignored)")]
    public Vector3 spawnAreaMin = new Vector3(-0.3f, 0f, -0.3f);

    [Tooltip("Maximum bounds for spawn area (X and Z are used, Y is ignored)")]
    public Vector3 spawnAreaMax = new Vector3(0.3f, 0f, 0.3f);

    [Header("Reward Tracking")]
    private Dictionary<string, float> rewardsByType = new Dictionary<string, float>();

    private Vector3 initialCubePosition;
    private int currentEpisodeSteps = 0;
    private float previousDistanceToCube = 0f;
    private float previousCubeHeight = 0f;
    private Dictionary<string, float> currentObservations = new Dictionary<string, float>();

    // Reference to stats logger for Eureka integration
    private TrainingStatsLogger statsLogger;
    private BehaviorParameters behaviorParameters;

    // Store initial joint positions for reset
    private float[] initialJointPositions;
    private float initialGripperPosition;

    // Task state
    private float currentHoldTime = 0f;
    private bool isTouchingTable = false;

    async void Start()
    {
        // Validate required references
        if (robotJoints == null || robotJoints.Length == 0)
        {
            Debug.LogError("RobotAgentTPickUpDirectControl: robotJoints not assigned!");
        }

        if (endEffector == null)
        {
            Debug.LogError("RobotAgentTPickUpDirectControl: endEffector not assigned!");
        }

        if (targetCube == null)
        {
            Debug.LogError("RobotAgentTPickUpDirectControl: targetCube not assigned!");
        }
        else
        {
            initialCubePosition = targetCube.position;
        }

        if (leftGripper == null || rightGripper == null)
        {
            Debug.LogError("RobotAgentTPickUpDirectControl: gripper references not assigned!");
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

        // Wait for RewardFunctionManager async initialization to complete
        Debug.Log("RobotAgentTPickUpDirectControl waiting for RewardFunctionManager initialization...");
        while (!RewardFunctionManager.IsInitialized)
        {
            await Task.Yield(); // Wait one frame
        }
        Debug.Log("RobotAgentTPickUpDirectControl: RewardFunctionManager ready, starting agent...");

        behaviorParameters = GetComponent<BehaviorParameters>();
    }

    public override void OnEpisodeBegin()
    {
        currentEpisodeSteps = 0;
        rewardsByType.Clear();
        currentHoldTime = 0f;
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

        // Reset grippers to open position
        if (leftGripper != null)
        {
            var leftDrive = leftGripper.xDrive;
            leftDrive.target = initialGripperPosition;
            leftGripper.xDrive = leftDrive;
            leftGripper.jointPosition = new ArticulationReducedSpace(0f);
            leftGripper.jointForce = new ArticulationReducedSpace(0f);
            leftGripper.jointVelocity = new ArticulationReducedSpace(0f);
        }

        if (rightGripper != null)
        {
            var rightDrive = rightGripper.xDrive;
            rightDrive.target = initialGripperPosition;
            rightGripper.xDrive = rightDrive;
            rightGripper.jointPosition = new ArticulationReducedSpace(0f);
            rightGripper.jointForce = new ArticulationReducedSpace(0f);
            rightGripper.jointVelocity = new ArticulationReducedSpace(0f);
        }

        // Reset cube position and physics
        if (targetCube != null)
        {
            // Use random spawn position if enabled, otherwise use initial position
            targetCube.position = randomizeSpawnPosition ? GetRandomSpawnPosition() : initialCubePosition;
            targetCube.rotation = Quaternion.identity;
            targetCube.SetParent(null);

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
        if (leftGripper != null && rightGripper != null && targetCube != null)
        {
            Vector3 gripperCenter = (leftGripper.transform.position + rightGripper.transform.position) * 0.5f;
            previousDistanceToCube = Vector3.Distance(gripperCenter, targetCube.position);
            previousCubeHeight = targetCube.position.y;
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

        float distanceToCube = Vector3.Distance(gripperCenter, targetCube.position);
        currentObservations["DistanceToCube"] = distanceToCube; // @obs: Euclidean distance from gripper center to cube

        // Cube height information
        currentObservations["CubeHeight"] = targetCube.position.y; // @obs: Current Y position (height) of the cube

        // Height difference from target
        float heightDifference = Mathf.Abs(targetCube.position.y - targetHeight);
        currentObservations["HeightDifference"] = heightDifference; // @obs: Absolute difference between cube height and target height

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

        // Is cube at target height
        bool isAtTargetHeight = isGrasping && heightDifference < heightTolerance;
        currentObservations["IsAtTargetHeight"] = isAtTargetHeight ? 1f : 0f; // @obs: 1 if cube is at target height, 0 otherwise

        // Hold time tracking
        currentObservations["HoldTime"] = currentHoldTime; // @obs: Current time cube has been held at target height
        currentObservations["HoldProgress"] = Mathf.Clamp01(currentHoldTime / holdDuration); // @obs: Progress toward hold duration goal

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
            distanceToCube,
            currentObservations["CubeHeight"],
            currentObservations["HeightDifference"],
            currentObservations["IsGrasping"],
            currentObservations["IsInGripVolume"],
            currentObservations["IsTouchingOutside"],
            currentObservations["LeftInsideTouching"],
            currentObservations["RightInsideTouching"],
            currentObservations["IsAtTargetHeight"],
            currentObservations["HoldTime"],
            currentObservations["HoldProgress"],
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

        // Apply gripper control (7th action) - symmetric control
        if (actions.ContinuousActions.Length > 6)
        {
            float gripperAction = actions.ContinuousActions[6];
            float gripperDelta = gripperAction * gripperActionSpeed * Time.fixedDeltaTime;

            // Apply to left gripper
            if (leftGripper != null)
            {
                var leftDrive = leftGripper.xDrive;
                float newLeftTarget = Mathf.Clamp(
                    leftDrive.target + gripperDelta,
                    leftDrive.lowerLimit,
                    leftDrive.upperLimit
                );
                leftDrive.target = newLeftTarget;
                leftGripper.xDrive = leftDrive;
            }

            // Apply to right gripper (same value for symmetry)
            if (rightGripper != null)
            {
                var rightDrive = rightGripper.xDrive;
                float newRightTarget = Mathf.Clamp(
                    rightDrive.target + gripperDelta,
                    rightDrive.lowerLimit,
                    rightDrive.upperLimit
                );
                rightDrive.target = newRightTarget;
                rightGripper.xDrive = rightDrive;
            }
        }

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
            EndDynamicEpisode();
        }

        // Update previous values for next frame
        previousDistanceToCube = currentObservations["DistanceToCube"];
        previousCubeHeight = targetCube.position.y;

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
            currentHoldTime += Time.fixedDeltaTime;
        }
        else
        {
            currentHoldTime = 0f;
        }
    }

    private bool IsTaskComplete()
    {
        if (targetCube == null) return false;

        bool cubeIsGrasped = graspDetector != null && graspDetector.isGrasping;
        float heightDifference = Mathf.Abs(targetCube.position.y - targetHeight);
        bool isAtTargetHeight = heightDifference < heightTolerance;
        bool hasBeenHeldLongEnough = currentHoldTime >= holdDuration;

        return cubeIsGrasped && isAtTargetHeight && hasBeenHeldLongEnough;
    }

    public void EndDynamicEpisode()
    {
        bool isSuccess = IsTaskComplete();

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
                ["previousDistanceToCube"] = previousDistanceToCube, // @context: Distance to cube in previous timestep
                ["previousCubeHeight"] = previousCubeHeight, // @context: Cube height in previous timestep
                ["graspThreshold"] = graspThreshold, // @context: Distance threshold for grasping cube
                ["targetHeight"] = targetHeight, // @context: Target height for holding cube
                ["heightTolerance"] = heightTolerance, // @context: Tolerance for height matching
                ["holdDuration"] = holdDuration, // @context: Required hold duration in seconds
                ["currentHoldTime"] = currentHoldTime, // @context: Current time cube has been held at target height
                ["episodeSteps"] = currentEpisodeSteps, // @context: Current number of steps in this episode
                ["maxSteps"] = ConfigManager.Instance.MaxStepsPerEpisode, // @context: Maximum steps per episode
                ["timeStep"] = Time.fixedDeltaTime, // @context: Fixed timestep duration
                ["endEffectorPosition"] = endEffector.position, // @context: Current end effector position
                ["cubePosition"] = targetCube.position, // @context: Current cube position
                ["cubeSpawnPosition"] = initialCubePosition, // @context: Initial cube spawn position
                ["cubeGrasped"] = graspDetector != null && graspDetector.isGrasping, // @context: Whether cube is currently grasped
                ["isSuccess"] = IsTaskComplete(), // @context: Whether the task has been successfully completed
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

            // Execute the reward function via the manager
            var result = await RewardFunctionManager.ExecuteRewardFunction(currentObservations, context);

            if (result != null)
            {
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

    public Dictionary<string, float> GetCurrentObservations()
    {
        return new Dictionary<string, float>(currentObservations);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;

        // Joint controls (same as T0DirectControl)
        continuousActionsOut[0] = Input.GetKey(KeyCode.A) ? -1.0f : (Input.GetKey(KeyCode.D) ? 1.0f : 0.0f);
        continuousActionsOut[1] = Input.GetKey(KeyCode.W) ? 1.0f : (Input.GetKey(KeyCode.S) ? -1.0f : 0.0f);
        continuousActionsOut[2] = Input.GetKey(KeyCode.Q) ? 1.0f : (Input.GetKey(KeyCode.E) ? -1.0f : 0.0f);
        continuousActionsOut[3] = Input.GetKey(KeyCode.UpArrow) ? 1.0f : (Input.GetKey(KeyCode.DownArrow) ? -1.0f : 0.0f);
        continuousActionsOut[4] = Input.GetKey(KeyCode.LeftArrow) ? -1.0f : (Input.GetKey(KeyCode.RightArrow) ? 1.0f : 0.0f);
        continuousActionsOut[5] = Input.GetKey(KeyCode.Z) ? 1.0f : (Input.GetKey(KeyCode.X) ? -1.0f : 0.0f);

        // Gripper control: G = close (negative), H = open (positive)
        continuousActionsOut[6] = Input.GetKey(KeyCode.G) ? -1.0f : (Input.GetKey(KeyCode.H) ? 1.0f : 0.0f);
    }

    [ContextMenu("Reload Reward Function")]
    public void ReloadReward()
    {
        if (ConfigManager.Instance.UseFileBasedReward && !string.IsNullOrEmpty(ConfigManager.Instance.RewardPath))
        {
            Debug.LogWarning("File-based reward reloading not implemented yet. Use RewardFunctionManagerInit for inspector rewards.");
        }
        else
        {
            Debug.LogWarning("Use RewardFunctionManagerInit component to reload inspector-based rewards.");
        }
    }

    // Public getters
    public int GetCurrentEpisodeSteps() => currentEpisodeSteps;
    public int GetMaxStepsPerEpisode() => ConfigManager.Instance.MaxStepsPerEpisode;
    public Dictionary<string, float> GetRewardsByType() => new Dictionary<string, float>(rewardsByType);
    public GameObject GetCurrentCube() => targetCube?.gameObject;
    public bool IsCubeGrasped() => graspDetector != null && graspDetector.isGrasping;
    public float GetCurrentHoldTime() => currentHoldTime;

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

    private Vector3 GetRandomSpawnPosition()
    {
        float randomX = UnityEngine.Random.Range(spawnAreaMin.x, spawnAreaMax.x);
        float randomZ = UnityEngine.Random.Range(spawnAreaMin.z, spawnAreaMax.z);

        // Create local offset relative to agent
        Vector3 localOffset = new Vector3(randomX, 0f, randomZ);

        // Convert to world space relative to agent's transform (position + rotation)
        Vector3 worldPosition = transform.TransformPoint(localOffset);

        // Use world position but maintain original Y height
        return new Vector3(worldPosition.x, initialCubePosition.y, worldPosition.z);
    }

    private void OnDrawGizmosSelected()
    {
        if (!randomizeSpawnPosition || targetCube == null) return;

        // Use initial cube position Y if available, otherwise use targetCube's current Y
        float spawnY = Application.isPlaying ? initialCubePosition.y : targetCube.position.y;

        // Define the corners in local space relative to agent
        Vector3 localBottomLeft = new Vector3(spawnAreaMin.x, 0f, spawnAreaMin.z);
        Vector3 localBottomRight = new Vector3(spawnAreaMax.x, 0f, spawnAreaMin.z);
        Vector3 localTopLeft = new Vector3(spawnAreaMin.x, 0f, spawnAreaMax.z);
        Vector3 localTopRight = new Vector3(spawnAreaMax.x, 0f, spawnAreaMax.z);

        // Convert to world space relative to agent's transform (position + rotation)
        Vector3 bottomLeft = transform.TransformPoint(localBottomLeft);
        Vector3 bottomRight = transform.TransformPoint(localBottomRight);
        Vector3 topLeft = transform.TransformPoint(localTopLeft);
        Vector3 topRight = transform.TransformPoint(localTopRight);

        // Adjust Y to spawn height
        bottomLeft.y = spawnY;
        bottomRight.y = spawnY;
        topLeft.y = spawnY;
        topRight.y = spawnY;

        // Draw the spawn area rectangle
        Gizmos.color = new Color(0f, 1f, 0f, 0.3f); // Green with transparency
        Gizmos.DrawLine(bottomLeft, bottomRight);
        Gizmos.DrawLine(bottomRight, topRight);
        Gizmos.DrawLine(topRight, topLeft);
        Gizmos.DrawLine(topLeft, bottomLeft);

        // Draw a semi-transparent plane to show the area
        Gizmos.color = new Color(0f, 1f, 0f, 0.1f);
        Vector3 center = (bottomLeft + bottomRight + topLeft + topRight) * 0.25f;

        // Calculate size and rotation
        float width = Vector3.Distance(bottomLeft, bottomRight);
        float depth = Vector3.Distance(bottomLeft, topLeft);
        Vector3 size = new Vector3(width, 0.01f, depth);

        // Draw rotated cube
        Matrix4x4 rotationMatrix = Matrix4x4.TRS(center, transform.rotation, size);
        Gizmos.matrix = rotationMatrix;
        Gizmos.DrawCube(Vector3.zero, Vector3.one);
        Gizmos.matrix = Matrix4x4.identity;
    }
}
