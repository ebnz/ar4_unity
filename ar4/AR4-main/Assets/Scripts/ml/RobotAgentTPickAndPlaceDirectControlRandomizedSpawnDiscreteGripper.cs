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

/// @task-name: TPickAndPlaceDirectControlRandomizedSpawnDiscreteGripper
/// <summary>
/// RobotAgent for TPickAndPlace with direct joint control (no IK), randomized spawn positions, and DISCRETE gripper control
/// </summary>
/// <taskdescription>
/// TPickAndPlace Task: Pick up a cube from a randomized location and transfer it to a randomized drop zone.
/// The agent directly controls all joints. Gripper uses discrete actions: Open/Close/Hold.
/// Both cube and drop zone positions are randomized each episode with a minimum distance constraint.
/// Success is achieved when the cube is placed within the drop zone area.
/// </taskdescription>
/// <successcriteria>
/// Cube is positioned within 0.2 units of the drop zone center and gripper is open
/// </successcriteria>
/// <actionspace>
/// Continuous Actions (6): [j1_delta, j2_delta, j3_delta, j4_delta, j5_delta, j6_delta]
/// Discrete Actions (1): gripper_action [0=Open, 1=Close, 2=Hold]
/// </actionspace>
public class RobotAgentTPickAndPlaceDirectControlRandomizedSpawnDiscreteGripper : Agent, IEurekaAgent
{
    [Header("References")]
    public ArticulationBody[] robotJoints;
    public Transform endEffector;
    public Transform dropZone;
    public Transform targetCube;
    public ArticulationBody leftGripper;
    public ArticulationBody rightGripper;
    public GraspDetector graspDetector;
    private GripperController gripperController;

    [Header("Settings")]
    [Tooltip("Speed multiplier for joint movements")]
    public float jointActionSpeed = 4f;

    private ObservationNormalizer observationNormalizer;

    [Header("Spawn Randomization")]
    [Tooltip("Enable random spawn positions for cube and drop zone")]
    public bool randomizeSpawnPositions = true;

    [Tooltip("Minimum bounds for spawn area (X and Z are used, Y is ignored)")]
    public Vector3 spawnAreaMin = new Vector3(-0.3f, 0f, -0.3f);

    [Tooltip("Maximum bounds for spawn area (X and Z are used, Y is ignored)")]
    public Vector3 spawnAreaMax = new Vector3(0.3f, 0f, 0.3f);

    [Tooltip("Minimum distance between cube and drop zone")]
    public float minCubeDropZoneDistance = 0.15f;

    [Tooltip("Maximum attempts to find valid spawn positions")]
    public int maxSpawnAttempts = 100;

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

    // Gripper state tracking for discrete control
    private enum GripperState { Open, Closing, Closed }
    private GripperState currentGripperState = GripperState.Open;

    async void Start()
    {
        // Validate required references
        if (robotJoints == null || robotJoints.Length == 0)
        {
            Debug.LogError("RobotAgentTPickAndPlaceDirectControlRandomizedSpawnDiscreteGripper: robotJoints not assigned!");
        }

        if (endEffector == null)
        {
            Debug.LogError("RobotAgentTPickAndPlaceDirectControlRandomizedSpawnDiscreteGripper: endEffector not assigned!");
        }

        if (dropZone != null)
            initialDropZonePosition = dropZone.position;

        if (targetCube != null)
            initialCubePosition = targetCube.position;
        else
        {
            Debug.LogError("Target cube not assigned to RobotAgentTPickAndPlaceDirectControlRandomizedSpawnDiscreteGripper!");
            throw new System.InvalidOperationException("Target cube must be assigned!");
        }

        if (leftGripper == null || rightGripper == null)
        {
            Debug.LogError("RobotAgentTPickAndPlaceDirectControlRandomizedSpawnDiscreteGripper: gripper references not assigned!");
        }

        // Get GripperController from children
        gripperController = GetComponentInChildren<GripperController>();
        if (gripperController == null)
        {
            Debug.LogError("RobotAgentTPickAndPlaceDirectControlRandomizedSpawnDiscreteGripper: GripperController not found in children!");
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


        statsLogger = FindFirstObjectByType<TrainingStatsLogger>();
        observationNormalizer = GetComponent<ObservationNormalizer>();

        // Wait for RewardFunctionManager async initialization to complete
        Debug.Log("RobotAgentTPickAndPlaceDirectControlRandomizedSpawnDiscreteGripper waiting for RewardFunctionManager initialization...");
        while (!RewardFunctionManager.IsInitialized)
        {
            await Task.Yield();
        }
        Debug.Log("RobotAgentTPickAndPlaceDirectControlRandomizedSpawnDiscreteGripper: RewardFunctionManager ready, starting agent...");

        behaviorParameters = GetComponent<BehaviorParameters>();
    }


    public override void OnEpisodeBegin()
    {
        currentEpisodeSteps = 0;
        rewardsByType.Clear();
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

        // Reset grippers to open position via GripperController
        if (gripperController != null)
        {
            gripperController.OpenGrippers();
        }

        // Get spawn positions (randomized or initial)
        Vector3 cubeSpawnPos, dropZoneSpawnPos;
        if (randomizeSpawnPositions)
        {
            GetRandomSpawnPositions(out cubeSpawnPos, out dropZoneSpawnPos);
        }
        else
        {
            cubeSpawnPos = initialCubePosition;
            dropZoneSpawnPos = initialDropZonePosition;
        }

        // Reset drop zone position
        dropZone.position = dropZoneSpawnPos;
        dropZone.localScale = new Vector3(dropZoneThreshold, 0.05f, dropZoneThreshold);

        // Reset cube position and physics
        if (targetCube != null)
        {
            targetCube.position = cubeSpawnPos;
            targetCube.rotation = Quaternion.identity;

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

    private void GetRandomSpawnPositions(out Vector3 cubePosition, out Vector3 dropZonePosition)
    {
        // Try to find valid positions with minimum distance constraint
        for (int attempt = 0; attempt < maxSpawnAttempts; attempt++)
        {
            cubePosition = GetRandomPositionInArea(initialCubePosition.y);
            dropZonePosition = GetRandomPositionInArea(initialDropZonePosition.y);

            // Check if distance constraint is satisfied (XZ plane only)
            float distance = Vector2.Distance(
                new Vector2(cubePosition.x, cubePosition.z),
                new Vector2(dropZonePosition.x, dropZonePosition.z)
            );

            if (distance >= minCubeDropZoneDistance)
            {
                return;
            }
        }

        // Fallback: if no valid positions found, use a simple approach
        Debug.LogWarning($"Could not find valid spawn positions after {maxSpawnAttempts} attempts. Using fallback positions.");
        cubePosition = GetRandomPositionInArea(initialCubePosition.y);

        // Place drop zone on opposite side
        Vector3 centerOffset = cubePosition - transform.position;
        Vector3 oppositeOffset = new Vector3(-centerOffset.x, 0f, -centerOffset.z);
        dropZonePosition = transform.position + oppositeOffset;
        dropZonePosition.y = initialDropZonePosition.y;
    }

    private Vector3 GetRandomPositionInArea(float yHeight)
    {
        float randomX = UnityEngine.Random.Range(spawnAreaMin.x, spawnAreaMax.x);
        float randomZ = UnityEngine.Random.Range(spawnAreaMin.z, spawnAreaMax.z);

        // Create local offset relative to agent
        Vector3 localOffset = new Vector3(randomX, 0f, randomZ);

        // Convert to world space relative to agent's transform (position + rotation)
        Vector3 worldPosition = transform.TransformPoint(localOffset);

        // Use world position but maintain specified Y height
        return new Vector3(worldPosition.x, yHeight, worldPosition.z);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        currentObservations.Clear();

        // Calculate gripper center position
        Vector3 gripperCenter = (leftGripper.transform.position + rightGripper.transform.position) * 0.5f;

        // Gripper center to cube vector
        Vector3 gripperToCube = targetCube.position - gripperCenter;
        currentObservations["EndEffectorToCube_X"] = gripperToCube.x; // @obs: X component of vector from end effector (gripper center) to cube
        currentObservations["EndEffectorToCube_Y"] = gripperToCube.y; // @obs: Y component of vector from end effector to cube
        currentObservations["EndEffectorToCube_Z"] = gripperToCube.z; // @obs: Z component of vector from end effector to cube

        // Distance from gripper center to cube
        float gripperDistanceToCube = Vector3.Distance(gripperCenter, targetCube.position);
        currentObservations["DistanceToCube"] = gripperDistanceToCube; // @obs: Euclidean distance from end effector to cube

        // Cube to drop zone vector
        Vector3 cubeToDropZone = dropZone.position - targetCube.position;
        currentObservations["CubeToDropZone_X"] = cubeToDropZone.x; // @obs: X component of vector from cube to drop zone
        currentObservations["CubeToDropZone_Y"] = cubeToDropZone.y; // @obs: Y component of vector from cube to drop zone
        currentObservations["CubeToDropZone_Z"] = cubeToDropZone.z; // @obs: Z component of vector from cube to drop zone

        // Distance from cube to drop zone
        float distanceToDropZone = Vector3.Distance(targetCube.position, dropZone.position);
        currentObservations["DistanceToDropZone"] = distanceToDropZone; // @obs: Euclidean distance from cube to drop zone center

        // End effector to drop zone vector (helps agent plan trajectory before grasping)
        Vector3 endEffectorToDropZone = dropZone.position - gripperCenter;
        currentObservations["EndEffectorToDropZone_X"] = endEffectorToDropZone.x; // @obs: X component of vector from end effector to drop zone
        currentObservations["EndEffectorToDropZone_Y"] = endEffectorToDropZone.y; // @obs: Y component of vector from end effector to drop zone
        currentObservations["EndEffectorToDropZone_Z"] = endEffectorToDropZone.z; // @obs: Z component of vector from end effector to drop zone
        currentObservations["DistanceToDropZoneFromEndEffector"] = Vector3.Distance(gripperCenter, dropZone.position); // @obs: Euclidean distance from end effector to drop zone

        // GraspDetector observations
        bool isGrasping = graspDetector != null && graspDetector.isGrasping;
        bool isInGripVolume = graspDetector != null && graspDetector.isInGripVolume;
        bool isTouchingOutside = graspDetector != null && graspDetector.isTouchingOutside;
        bool leftInsideTouching = graspDetector != null && graspDetector.leftInsideTouching;
        bool rightInsideTouching = graspDetector != null && graspDetector.rightInsideTouching;

        currentObservations["IsGrasping"] = isGrasping ? 1f : 0f; // @obs: 1 if cube is being grasped (in grip volume, both fingers touching inside, not touching outside), 0 otherwise
        currentObservations["IsInGripVolume"] = isInGripVolume ? 1f : 0f; // @obs: 1 if cube is within the gripper volume trigger, 0 otherwise
        currentObservations["IsTouchingOutside"] = isTouchingOutside ? 1f : 0f; // @obs: 1 if cube is touching outside of gripper fingers, 0 otherwise
        currentObservations["LeftInsideTouching"] = leftInsideTouching ? 1f : 0f; // @obs: 1 if left inside finger is touching cube, 0 otherwise
        currentObservations["RightInsideTouching"] = rightInsideTouching ? 1f : 0f; // @obs: 1 if right inside finger is touching cube, 0 otherwise

        // Table contact state
        currentObservations["TouchingTable"] = isTouchingTable ? 1f : 0f; // @obs: 1 if robot is touching the table, 0 otherwise

        // Gripper state (discrete): 0=Open, 0.5=Closing, 1=Closed
        float gripperStateValue = currentGripperState == GripperState.Open ? 0f :
                                   currentGripperState == GripperState.Closing ? 0.5f : 1f;
        currentObservations["GripperState"] = gripperStateValue; // @obs: Discrete gripper state (0=Open, 0.5=Closing, 1=Closed)

        // Current joint positions (normalized to [-1, 1])
        for (int i = 0; i < robotJoints.Length; i++)
        {
            var joint = robotJoints[i];
            var drive = joint.xDrive;
            float normalizedPosition = (drive.target - drive.lowerLimit) / (drive.upperLimit - drive.lowerLimit) * 2f - 1f;
            currentObservations[$"Joint{i}_Position"] = normalizedPosition; // @obs: Normalized position of joint {i} (-1 to 1)
        }

        // Joint velocities
        for (int i = 0; i < robotJoints.Length; i++)
        {
            float velocity = robotJoints[i].jointVelocity[0];
            currentObservations[$"Joint{i}_Velocity"] = velocity; // @obs: Angular velocity of joint {i}
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
            endEffectorToDropZone.x,
            endEffectorToDropZone.y,
            endEffectorToDropZone.z,
            currentObservations["DistanceToDropZoneFromEndEffector"],
            currentObservations["IsGrasping"],
            currentObservations["IsInGripVolume"],
            currentObservations["IsTouchingOutside"],
            currentObservations["LeftInsideTouching"],
            currentObservations["RightInsideTouching"],
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

        // Handle discrete gripper action (GripperController handles smooth movement)
        if (actions.DiscreteActions.Length > 0)
        {
            int gripperAction = actions.DiscreteActions[0];
            HandleDiscreteGripperAction(gripperAction);
        }

        // Update gripper state based on actual position (GripperController moves the gripper)
        UpdateGripperState();

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

    private void HandleDiscreteGripperAction(int action)
    {
        if (gripperController == null)
        {
            Debug.LogWarning("HandleDiscreteGripperAction: gripperController is null!");
            return;
        }

        switch (action)
        {
            case 0: // Open
                gripperController.OpenGrippers();
                currentGripperState = GripperState.Open;
                break;
            case 1: // Close
                gripperController.CloseGrippers();
                currentGripperState = GripperState.Closing;
                break;
            case 2: // Hold (no change)
                // Keep current target, update state based on position
                if (leftGripper != null)
                {
                    var drive = leftGripper.xDrive;
                    float currentPosition = drive.target;
                    float closedThreshold = drive.lowerLimit + (drive.upperLimit - drive.lowerLimit) * 0.2f;
                    if (currentPosition < closedThreshold)
                    {
                        currentGripperState = GripperState.Closed;
                    }
                }
                break;
        }
    }

    private void UpdateGripperState()
    {
        // Update state based on actual gripper position (GripperController handles movement)
        if (leftGripper == null) return;

        var drive = leftGripper.xDrive;
        float currentPos = drive.target;
        float closedThreshold = drive.lowerLimit + (drive.upperLimit - drive.lowerLimit) * 0.1f;

        // Transition from Closing to Closed when gripper reaches closed position
        if (currentGripperState == GripperState.Closing && currentPos <= closedThreshold)
        {
            currentGripperState = GripperState.Closed;
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

    private void ExecuteRewardFunction()
    {
        try
        {
            bool cubeIsGrasped = graspDetector != null && graspDetector.isGrasping;

            // Calculate cube lift height (Y difference from initial spawn position)
            float cubeLiftHeight = targetCube != null ? targetCube.position.y - initialCubePosition.y : 0f;

            var context = new Dictionary<string, object>
            {
                ["previousDistanceToCube"] = previousDistanceToCube, // @context: Distance from end effector to cube in previous timestep (float)
                ["previousDistanceToDropZone"] = previousDistanceToDropZone, // @context: Distance from cube to drop zone in previous timestep (float)
                ["dropZoneThreshold"] = dropZoneThreshold, // @context: Distance threshold for successful cube placement (float, default 0.2)
                ["episodeSteps"] = currentEpisodeSteps, // @context: Current step number within this episode (int)
                ["maxSteps"] = ConfigManager.Instance.MaxStepsPerEpisode, // @context: Maximum steps allowed per episode (int)
                ["timeStep"] = Time.fixedDeltaTime, // @context: Fixed physics timestep duration (float, typically 0.02)
                ["endEffectorPosition"] = endEffector.position, // @context: Current world position of end effector (Vector3)
                ["dropZonePosition"] = dropZone.position, // @context: Current world position of drop zone center (Vector3)
                ["cubePosition"] = targetCube != null ? targetCube.position : Vector3.zero, // @context: Current world position of cube (Vector3)
                ["cubeSpawnPosition"] = initialCubePosition, // @context: Initial spawn position of cube at episode start (Vector3)
                ["cubeGrasped"] = cubeIsGrasped, // @context: Whether cube is currently being grasped (bool)
                ["isSuccess"] = IsTaskComplete(), // @context: Whether task has been completed successfully (bool)
                ["cubeLiftHeight"] = cubeLiftHeight, // @context: Y difference from initial cube position, positive means lifted (float)
                ["gripperState"] = (int)currentGripperState, // @context: Current gripper state (0=Open, 1=Closing, 2=Closed)
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

    public Dictionary<string, float> GetCurrentObservations()
    {
        return new Dictionary<string, float>(currentObservations);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        var discreteActionsOut = actionsOut.DiscreteActions;

        // Joint controls (6 continuous actions)
        if (continuousActionsOut.Length >= 6)
        {
            continuousActionsOut[0] = Input.GetKey(KeyCode.A) ? -1.0f : (Input.GetKey(KeyCode.D) ? 1.0f : 0.0f);
            continuousActionsOut[1] = Input.GetKey(KeyCode.W) ? 1.0f : (Input.GetKey(KeyCode.S) ? -1.0f : 0.0f);
            continuousActionsOut[2] = Input.GetKey(KeyCode.Q) ? 1.0f : (Input.GetKey(KeyCode.E) ? -1.0f : 0.0f);
            continuousActionsOut[3] = Input.GetKey(KeyCode.UpArrow) ? 1.0f : (Input.GetKey(KeyCode.DownArrow) ? -1.0f : 0.0f);
            continuousActionsOut[4] = Input.GetKey(KeyCode.LeftArrow) ? -1.0f : (Input.GetKey(KeyCode.RightArrow) ? 1.0f : 0.0f);
            continuousActionsOut[5] = Input.GetKey(KeyCode.Z) ? 1.0f : (Input.GetKey(KeyCode.X) ? -1.0f : 0.0f);
        }

        // Discrete gripper control: G = close, H = open, default = hold
        if (discreteActionsOut.Length >= 1)
        {
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

    public void OnDecisionRequested()
    {
        currentEpisodeSteps++;
    }

    // Table collision tracking
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

    private void OnDrawGizmosSelected()
    {
        if (!randomizeSpawnPositions) return;

        float cubeY = Application.isPlaying ? initialCubePosition.y : (targetCube != null ? targetCube.position.y : 1f);
        float dropZoneY = Application.isPlaying ? initialDropZonePosition.y : (dropZone != null ? dropZone.position.y : 1f);

        Vector3 localBottomLeft = new Vector3(spawnAreaMin.x, 0f, spawnAreaMin.z);
        Vector3 localBottomRight = new Vector3(spawnAreaMax.x, 0f, spawnAreaMin.z);
        Vector3 localTopLeft = new Vector3(spawnAreaMin.x, 0f, spawnAreaMax.z);
        Vector3 localTopRight = new Vector3(spawnAreaMax.x, 0f, spawnAreaMax.z);

        Vector3 bottomLeft = transform.TransformPoint(localBottomLeft);
        Vector3 bottomRight = transform.TransformPoint(localBottomRight);
        Vector3 topLeft = transform.TransformPoint(localTopLeft);
        Vector3 topRight = transform.TransformPoint(localTopRight);

        DrawSpawnArea(bottomLeft, bottomRight, topLeft, topRight, cubeY, new Color(0f, 1f, 0f, 0.3f), "Cube");
        DrawSpawnArea(bottomLeft, bottomRight, topLeft, topRight, dropZoneY + 0.01f, new Color(0f, 0.5f, 1f, 0.3f), "DropZone");

        Vector3 center = (bottomLeft + bottomRight + topLeft + topRight) * 0.25f;
        center.y = (cubeY + dropZoneY) / 2f;
        Gizmos.color = new Color(1f, 0.5f, 0f, 0.5f);
        DrawWireCircle(center, minCubeDropZoneDistance / 2f, 32);
    }

    private void DrawSpawnArea(Vector3 bl, Vector3 br, Vector3 tl, Vector3 tr, float y, Color color, string label)
    {
        bl.y = y; br.y = y; tl.y = y; tr.y = y;

        Gizmos.color = color;
        Gizmos.DrawLine(bl, br);
        Gizmos.DrawLine(br, tr);
        Gizmos.DrawLine(tr, tl);
        Gizmos.DrawLine(tl, bl);

        Gizmos.color = new Color(color.r, color.g, color.b, 0.1f);
        Vector3 center = (bl + br + tl + tr) * 0.25f;
        float width = Vector3.Distance(bl, br);
        float depth = Vector3.Distance(bl, tl);
        Vector3 size = new Vector3(width, 0.01f, depth);

        Matrix4x4 rotationMatrix = Matrix4x4.TRS(center, transform.rotation, size);
        Gizmos.matrix = rotationMatrix;
        Gizmos.DrawCube(Vector3.zero, Vector3.one);
        Gizmos.matrix = Matrix4x4.identity;
    }

    private void DrawWireCircle(Vector3 center, float radius, int segments)
    {
        float angleStep = 360f / segments;
        Vector3 prevPoint = center + new Vector3(radius, 0, 0);

        for (int i = 1; i <= segments; i++)
        {
            float angle = i * angleStep * Mathf.Deg2Rad;
            Vector3 newPoint = center + new Vector3(Mathf.Cos(angle) * radius, 0, Mathf.Sin(angle) * radius);
            Gizmos.DrawLine(prevPoint, newPoint);
            prevPoint = newPoint;
        }
    }
}
