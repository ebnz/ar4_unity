using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using System.Threading.Tasks;
using System;
using System.Linq;
using Unity.MLAgents.Policies;

/// @task-name: TPickUpDirectControlDiscreteGripper
/// <summary>
/// RobotAgent for TPickUp with direct joint control (no IK) and DISCRETE gripper control
/// </summary>
/// <taskdescription>
/// TPickUp Task: Pick up a cube and hold it at y = 1.5 for 3 seconds.
/// The agent directly controls all joints. Gripper uses discrete actions: Open/Close/Hold.
/// Success is achieved when the cube is grasped and held at the target height for the required duration.
/// </taskdescription>
/// <successcriteria>
/// Cube is grasped, positioned at y = 1.5 (within tolerance), and held there for 3 seconds
/// </successcriteria>
/// <actionspace>
/// Continuous Actions (6): [j1_delta, j2_delta, j3_delta, j4_delta, j5_delta, j6_delta]
/// Discrete Actions (1): gripper_action [0=Open, 1=Close, 2=Hold]
/// </actionspace>
public class RobotAgentTPickUpDirectControlDiscreteGripper : Agent, IEurekaAgent
{
    [Header("References")]
    public ArticulationBody[] robotJoints;
    public Transform endEffector;
    public Transform targetCube;
    public ArticulationBody leftGripper;
    public ArticulationBody rightGripper;

    [Header("Settings")]
    [Tooltip("Speed multiplier for joint movements")]
    public float jointActionSpeed = 50f;

    [Tooltip("Speed at which gripper opens/closes")]
    public float gripperSpeed = 30f;

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

    [Header("Debug")]
    [Tooltip("GameObject to activate when cube is grasped (for debugging)")]
    public GameObject graspDebugIndicator;

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
    private bool cubeIsGrasped = false;
    private float currentHoldTime = 0f;
    private bool isTouchingTable = false;

    // Velocity tracking for rigid motion detection
    private Vector3 previousGripperCenter;
    private Vector3 previousCubePosition;

    // Gripper state tracking
    private enum GripperState { Open, Closing, Closed }
    private GripperState currentGripperState = GripperState.Open;
    private float targetGripperPosition = 0f;

    async void Start()
    {
        // Validate required references
        if (robotJoints == null || robotJoints.Length == 0)
        {
            Debug.LogError("RobotAgentTPickUpDirectControlDiscreteGripper: robotJoints not assigned!");
        }

        if (endEffector == null)
        {
            Debug.LogError("RobotAgentTPickUpDirectControlDiscreteGripper: endEffector not assigned!");
        }

        if (targetCube == null)
        {
            Debug.LogError("RobotAgentTPickUpDirectControlDiscreteGripper: targetCube not assigned!");
        }
        else
        {
            initialCubePosition = targetCube.position;
        }

        if (leftGripper == null || rightGripper == null)
        {
            Debug.LogError("RobotAgentTPickUpDirectControlDiscreteGripper: gripper references not assigned!");
        }

        // Store initial joint positions for reset
        initialJointPositions = new float[robotJoints.Length];
        for (int i = 0; i < robotJoints.Length; i++)
        {
            initialJointPositions[i] = robotJoints[i].xDrive.target;
        }

        // Store initial gripper position (open)
        if (leftGripper != null)
        {
            initialGripperPosition = leftGripper.xDrive.target;
            targetGripperPosition = initialGripperPosition;
        }

        statsLogger = FindFirstObjectByType<TrainingStatsLogger>();
        observationNormalizer = GetComponent<ObservationNormalizer>();

        // Wait for RewardFunctionManager async initialization to complete
        Debug.Log("RobotAgentTPickUpDirectControlDiscreteGripper waiting for RewardFunctionManager initialization...");
        while (!RewardFunctionManager.IsInitialized)
        {
            await Task.Yield(); // Wait one frame
        }
        Debug.Log("RobotAgentTPickUpDirectControlDiscreteGripper: RewardFunctionManager ready, starting agent...");

        behaviorParameters = GetComponent<BehaviorParameters>();
    }

    public override void OnEpisodeBegin()
    {
        currentEpisodeSteps = 0;
        rewardsByType.Clear();
        cubeIsGrasped = false;
        currentHoldTime = 0f;
        isTouchingTable = false;
        tableCollisionCount = 0;
        currentGripperState = GripperState.Open;

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
        targetGripperPosition = initialGripperPosition;
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
            targetCube.position = initialCubePosition;
            targetCube.rotation = Quaternion.identity;
            targetCube.SetParent(null);

            if (targetCube.TryGetComponent<Rigidbody>(out var rb))
            {
                rb.isKinematic = false;
                rb.linearVelocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
            }
        }

        // Initialize velocity tracking and distance
        if (leftGripper != null && rightGripper != null && targetCube != null)
        {
            Vector3 gripperCenter = (leftGripper.transform.position + rightGripper.transform.position) * 0.5f;
            previousGripperCenter = gripperCenter;
            previousDistanceToCube = Vector3.Distance(gripperCenter, targetCube.position);
            previousCubeHeight = targetCube.position.y;
        }
        previousCubePosition = targetCube.position;
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

        // Is cube at target height
        bool isAtTargetHeight = cubeIsGrasped && heightDifference < heightTolerance;
        currentObservations["IsAtTargetHeight"] = isAtTargetHeight ? 1f : 0f; // @obs: 1 if cube is at target height, 0 otherwise

        // Hold time tracking
        currentObservations["HoldTime"] = currentHoldTime; // @obs: Current time cube has been held at target height
        currentObservations["HoldProgress"] = Mathf.Clamp01(currentHoldTime / holdDuration); // @obs: Progress toward hold duration goal

        // Cube grasped state
        currentObservations["CubeGrasped"] = cubeIsGrasped ? 1f : 0f; // @obs: 1 if cube is grasped, 0 if not

        // Table contact state
        currentObservations["TouchingTable"] = isTouchingTable ? 1f : 0f; // @obs: 1 if robot is touching the table, 0 if not

        // Gripper state (discrete): 0=Open, 0.5=Closing, 1=Closed
        float gripperStateValue = currentGripperState == GripperState.Open ? 0f :
                                   currentGripperState == GripperState.Closing ? 0.5f : 1f;
        currentObservations["GripperState"] = gripperStateValue; // @obs: Gripper state: 0=Open, 0.5=Closing, 1=Closed

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
            currentObservations["IsAtTargetHeight"],
            currentObservations["HoldTime"],
            currentObservations["HoldProgress"],
            currentObservations["CubeGrasped"],
            currentObservations["TouchingTable"],
            currentObservations["GripperState"]
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
        // Apply direct joint control (first 6 continuous actions)
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

        // Handle discrete gripper action
        if (actions.DiscreteActions.Length > 0)
        {
            int gripperAction = actions.DiscreteActions[0];
            HandleDiscreteGripperAction(gripperAction);
        }

        // Update gripper positions toward target
        UpdateGripperPositions();

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
            EndDynamicEpisode();
        }

        // Update previous values for next frame
        previousDistanceToCube = currentObservations["DistanceToCube"];
        previousCubeHeight = targetCube.position.y;

        // Update velocity tracking positions
        if (leftGripper != null && rightGripper != null)
        {
            previousGripperCenter = (leftGripper.transform.position + rightGripper.transform.position) * 0.5f;
        }
        previousCubePosition = targetCube.position;

        if (currentEpisodeSteps >= ConfigManager.Instance.MaxStepsPerEpisode)
        {
            AddRewardWithType("Timeout", -1.0f);
            EndDynamicEpisode();
        }
    }

    private void HandleDiscreteGripperAction(int action)
    {
        if (leftGripper == null) return;

        var drive = leftGripper.xDrive;

        switch (action)
        {
            case 0: // Open
                targetGripperPosition = drive.upperLimit;
                currentGripperState = GripperState.Open;
                break;
            case 1: // Close
                targetGripperPosition = drive.lowerLimit;
                currentGripperState = GripperState.Closing;
                break;
            case 2: // Hold (no change)
                // Keep current target, update state based on position
                float currentPosition = drive.target;
                float closedThreshold = drive.lowerLimit + (drive.upperLimit - drive.lowerLimit) * 0.2f;
                if (currentPosition < closedThreshold)
                {
                    currentGripperState = GripperState.Closed;
                }
                break;
        }
    }

    private void UpdateGripperPositions()
    {
        if (leftGripper == null || rightGripper == null) return;

        // Smoothly move grippers toward target position
        var leftDrive = leftGripper.xDrive;
        float currentPos = leftDrive.target;
        float newPos = Mathf.MoveTowards(currentPos, targetGripperPosition, gripperSpeed * Time.fixedDeltaTime);
        newPos = Mathf.Clamp(newPos, leftDrive.lowerLimit, leftDrive.upperLimit);

        leftDrive.target = newPos;
        leftGripper.xDrive = leftDrive;

        // Apply same to right gripper
        var rightDrive = rightGripper.xDrive;
        rightDrive.target = newPos;
        rightGripper.xDrive = rightDrive;

        // Update state based on actual position
        if (currentGripperState == GripperState.Closing)
        {
            float closedThreshold = leftDrive.lowerLimit + (leftDrive.upperLimit - leftDrive.lowerLimit) * 0.1f;
            if (newPos <= closedThreshold)
            {
                currentGripperState = GripperState.Closed;
            }
        }
    }

    private void UpdateCubeGrasping()
    {
        if (targetCube == null || leftGripper == null || rightGripper == null) return;

        // 1. Check gripper contact force (position error indicates something blocking closure)
        float targetPosition = leftGripper.xDrive.target;
        float actualPosition = leftGripper.jointPosition[0];
        float positionError = actualPosition - targetPosition;
        bool gripperContactingObject = positionError > 0.02f; // 0.02 radians tolerance

        // Calculate gripper center position
        Vector3 gripperCenter = (leftGripper.transform.position + rightGripper.transform.position) * 0.5f;

        // 2. Check cube near gripper (vision analog - distance check using gripper center)
        float distanceToCube = Vector3.Distance(gripperCenter, targetCube.position);
        bool cubeNearGripper = distanceToCube < graspThreshold; // 0.15f default

        // 3. Check cube and gripper center moving rigidly together
        Vector3 gripperVelocity = (gripperCenter - previousGripperCenter) / Time.fixedDeltaTime;
        Vector3 cubeVelocity = targetCube.GetComponent<Rigidbody>().linearVelocity;
        float velocityDifference = Vector3.Distance(cubeVelocity, gripperVelocity);
        bool movingRigidly = velocityDifference < 0.5f; // 50cm/s tolerance

        // Cube is grasped when all three conditions are met AND gripper is closed
        cubeIsGrasped = gripperContactingObject && cubeNearGripper && movingRigidly &&
                        (currentGripperState == GripperState.Closed || currentGripperState == GripperState.Closing);

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
                ["cubeGrasped"] = cubeIsGrasped, // @context: Whether cube is currently grasped
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
        var discreteActionsOut = actionsOut.DiscreteActions;

        // Joint controls (same as before)
        continuousActionsOut[0] = Input.GetKey(KeyCode.A) ? -1.0f : (Input.GetKey(KeyCode.D) ? 1.0f : 0.0f);
        continuousActionsOut[1] = Input.GetKey(KeyCode.W) ? 1.0f : (Input.GetKey(KeyCode.S) ? -1.0f : 0.0f);
        continuousActionsOut[2] = Input.GetKey(KeyCode.Q) ? 1.0f : (Input.GetKey(KeyCode.E) ? -1.0f : 0.0f);
        continuousActionsOut[3] = Input.GetKey(KeyCode.UpArrow) ? 1.0f : (Input.GetKey(KeyCode.DownArrow) ? -1.0f : 0.0f);
        continuousActionsOut[4] = Input.GetKey(KeyCode.LeftArrow) ? -1.0f : (Input.GetKey(KeyCode.RightArrow) ? 1.0f : 0.0f);
        continuousActionsOut[5] = Input.GetKey(KeyCode.Z) ? 1.0f : (Input.GetKey(KeyCode.X) ? -1.0f : 0.0f);

        // Discrete gripper control: G = close, H = open, Space = hold
        if (Input.GetKey(KeyCode.G))
        {
            discreteActionsOut[0] = 1; // Close
        }
        else if (Input.GetKey(KeyCode.H))
        {
            discreteActionsOut[0] = 0; // Open
        }
        else
        {
            discreteActionsOut[0] = 2; // Hold
        }
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
    public bool IsCubeGrasped() => cubeIsGrasped;
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
}
