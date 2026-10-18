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

/// @task-name: TPickAndPlaceDirectControlRandomizedSpawn
/// <summary>
/// RobotAgent for TPickAndPlace with direct joint control (no IK) and randomized cube/drop zone spawn positions
/// </summary>
/// <taskdescription>
/// TPickAndPlace Task: Pick up a cube from a randomized location and transfer it to a randomized drop zone.
/// The agent directly controls all joints including the gripper to grasp and transport the cube.
/// Both cube and drop zone positions are randomized each episode with a minimum distance constraint.
/// Success is achieved when the cube is placed within the drop zone area.
/// </taskdescription>
/// <successcriteria>
/// Cube is positioned within 0.2 units of the drop zone center and gripper is open
/// </successcriteria>
/// <actionspace>
/// Continuous Actions (7): [j1_delta, j2_delta, j3_delta, j4_delta, j5_delta, j6_delta, gripper_delta]
/// </actionspace>
public class RobotAgentTPickAndPlaceDirectControlRandomizedSpawn : Agent, IEurekaAgent
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

    [Tooltip("Speed multiplier for gripper movements")]
    public float gripperActionSpeed = 0.03f;

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

    [Header("Curriculum Learning")]
    [Tooltip("Enable curriculum learning - gradual spawn area expansion based on success rate")]
    public bool useCurriculum = true;

    [Tooltip("Starting spawn bounds for CUBE (X,Z relative to agent transform - auto-set from cube position)")]
    public Vector2 curriculumCubeStartMin = new Vector2(0f, 0f);
    public Vector2 curriculumCubeStartMax = new Vector2(0f, 0f);

    [Tooltip("Target spawn bounds for CUBE (X,Z relative to agent transform - full difficulty)")]
    public Vector2 curriculumCubeEndMin = new Vector2(-0.3f, -0.3f);
    public Vector2 curriculumCubeEndMax = new Vector2(0.3f, 0.3f);

    [Tooltip("Starting spawn bounds for DROP ZONE (X,Z relative to agent transform - auto-set from drop zone position)")]
    public Vector2 curriculumDropZoneStartMin = new Vector2(0f, 0f);
    public Vector2 curriculumDropZoneStartMax = new Vector2(0f, 0f);

    [Tooltip("Target spawn bounds for DROP ZONE (X,Z relative to agent transform - full difficulty)")]
    public Vector2 curriculumDropZoneEndMin = new Vector2(-0.3f, -0.3f);
    public Vector2 curriculumDropZoneEndMax = new Vector2(0.3f, 0.3f);

    [Tooltip("Success rate threshold to advance curriculum (0-1) - ALL checkpoints in window must exceed this")]
    public float curriculumSuccessThreshold = 0.9f;

    [Tooltip("Number of recent checkpoints that ALL must exceed threshold for advancement")]
    public int curriculumCheckpointWindow = 10;

    [Tooltip("How much to expand spawn area each advancement (0-1, as fraction of total range)")]
    public float curriculumExpansionStep = 0.1f;

    [Tooltip("If success rate drops below this, regress curriculum one step (0 to disable)")]
    public float curriculumRegressionThreshold = 0.3f;

    [Tooltip("Number of consecutive checkpoints below regression threshold to trigger regression")]
    public int curriculumRegressionWindow = 5;

    // Curriculum runtime state
    private float curriculumProgress = 0f;  // 0 = start bounds, 1 = end bounds
    private Queue<float> recentCheckpointSuccessRates = new Queue<float>();
    private int consecutiveLowSuccessCount = 0;  // For regression tracking

    // Current curriculum spawn bounds (computed from progress, relative to agent transform)
    private Vector2 currentCubeSpawnMin;
    private Vector2 currentCubeSpawnMax;
    private Vector2 currentDropZoneSpawnMin;
    private Vector2 currentDropZoneSpawnMax;

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

    /// <summary>
    /// Called when component is first added or Reset is clicked in inspector.
    /// Auto-initializes curriculum bounds based on current cube/drop zone positions.
    /// </summary>
    private void Reset()
    {
        AutoSetCurriculumStartBounds();
    }

    /// <summary>
    /// Called when inspector values change. Updates curriculum start bounds if references change.
    /// </summary>
    private void OnValidate()
    {
        // Only auto-set if bounds are at default zero values (not yet configured)
        if (curriculumCubeStartMin == curriculumCubeStartMax &&
            curriculumDropZoneStartMin == curriculumDropZoneStartMax)
        {
            AutoSetCurriculumStartBounds();
        }
    }

    /// <summary>
    /// Sets curriculum start bounds based on current cube and drop zone positions relative to agent.
    /// </summary>
    [ContextMenu("Auto-Set Curriculum Start Bounds")]
    public void AutoSetCurriculumStartBounds()
    {
        if (targetCube != null)
        {
            // Get cube position in agent's local space
            Vector3 localCubePos = transform.InverseTransformPoint(targetCube.position);
            curriculumCubeStartMin = new Vector2(localCubePos.x, localCubePos.z);
            curriculumCubeStartMax = new Vector2(localCubePos.x, localCubePos.z);
        }

        if (dropZone != null)
        {
            // Get drop zone position in agent's local space
            Vector3 localDropZonePos = transform.InverseTransformPoint(dropZone.position);
            curriculumDropZoneStartMin = new Vector2(localDropZonePos.x, localDropZonePos.z);
            curriculumDropZoneStartMax = new Vector2(localDropZonePos.x, localDropZonePos.z);
        }

        #if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            UnityEditor.EditorUtility.SetDirty(this);
        }
        #endif
    }

    async void Start()
    {
        // Validate required references
        if (robotJoints == null || robotJoints.Length == 0)
        {
            Debug.LogError("RobotAgentTPickAndPlaceDirectControlRandomizedSpawn: robotJoints not assigned!");
        }

        if (endEffector == null)
        {
            Debug.LogError("RobotAgentTPickAndPlaceDirectControlRandomizedSpawn: endEffector not assigned!");
        }

        if (dropZone != null)
            initialDropZonePosition = dropZone.position;

        if (targetCube != null)
            initialCubePosition = targetCube.position;
        else
        {
            Debug.LogError("Target cube not assigned to RobotAgentTPickAndPlaceDirectControlRandomizedSpawn!");
            throw new System.InvalidOperationException("Target cube must be assigned!");
        }

        if (leftGripper == null || rightGripper == null)
        {
            Debug.LogError("RobotAgentTPickAndPlaceDirectControlRandomizedSpawn: gripper references not assigned!");
        }

        // Get GripperController from children
        gripperController = GetComponentInChildren<GripperController>();
        if (gripperController == null)
        {
            Debug.LogError("RobotAgentTPickAndPlaceDirectControlRandomizedSpawn: GripperController not found in children!");
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

        // Initialize curriculum learning
        InitializeCurriculum();

        // Wait for RewardFunctionManager async initialization to complete
        Debug.Log("RobotAgentTPickAndPlaceDirectControlRandomizedSpawn waiting for RewardFunctionManager initialization...");
        while (!RewardFunctionManager.IsInitialized)
        {
            await Task.Yield();
        }
        Debug.Log("RobotAgentTPickAndPlaceDirectControlRandomizedSpawn: RewardFunctionManager ready, starting agent...");

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

        // Reset grippers to open position via GripperController
        if (gripperController != null)
        {
            gripperController.ResetGrippers();
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
        // Use curriculum bounds if enabled, otherwise use the inspector spawn area
        Vector2 cubeMin, cubeMax, dropZoneMin, dropZoneMax;

        if (useCurriculum)
        {
            cubeMin = currentCubeSpawnMin;
            cubeMax = currentCubeSpawnMax;
            dropZoneMin = currentDropZoneSpawnMin;
            dropZoneMax = currentDropZoneSpawnMax;
        }
        else
        {
            // Use shared spawn area for both (original behavior)
            cubeMin = new Vector2(spawnAreaMin.x, spawnAreaMin.z);
            cubeMax = new Vector2(spawnAreaMax.x, spawnAreaMax.z);
            dropZoneMin = cubeMin;
            dropZoneMax = cubeMax;
        }

        // Try to find valid positions with minimum distance constraint
        for (int attempt = 0; attempt < maxSpawnAttempts; attempt++)
        {
            cubePosition = GetRandomPositionInBounds(initialCubePosition.y, cubeMin, cubeMax);
            dropZonePosition = GetRandomPositionInBounds(initialDropZonePosition.y, dropZoneMin, dropZoneMax);

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

        // Fallback: if no valid positions found, use initial positions
        Debug.LogWarning($"Could not find valid spawn positions after {maxSpawnAttempts} attempts. Using initial positions.");
        cubePosition = initialCubePosition;
        dropZonePosition = initialDropZonePosition;
    }

    /// <summary>
    /// Gets a random position within bounds relative to agent transform.
    /// </summary>
    private Vector3 GetRandomPositionInBounds(float yHeight, Vector2 boundsMin, Vector2 boundsMax)
    {
        float randomX = UnityEngine.Random.Range(boundsMin.x, boundsMax.x);
        float randomZ = UnityEngine.Random.Range(boundsMin.y, boundsMax.y);

        // Create local offset relative to agent
        Vector3 localOffset = new Vector3(randomX, 0f, randomZ);

        // Convert to world space relative to agent's transform (position + rotation)
        Vector3 worldPosition = transform.TransformPoint(localOffset);

        // Use world position but maintain specified Y height
        return new Vector3(worldPosition.x, yHeight, worldPosition.z);
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

    #region Curriculum Learning

    /// <summary>
    /// Called by TrainingStatsLogger when a checkpoint is saved.
    /// Updates curriculum progress based on recent success rates.
    /// </summary>
    public void OnCheckpointSaved(float successRate)
    {
        if (!useCurriculum) return;

        // Track recent checkpoint success rates
        recentCheckpointSuccessRates.Enqueue(successRate);
        while (recentCheckpointSuccessRates.Count > curriculumCheckpointWindow)
            recentCheckpointSuccessRates.Dequeue();

        // Check for regression first (if enabled and not already at 0%)
        if (curriculumRegressionThreshold > 0f && curriculumProgress > 0f)
        {
            if (successRate < curriculumRegressionThreshold)
            {
                consecutiveLowSuccessCount++;
                if (consecutiveLowSuccessCount >= curriculumRegressionWindow)
                {
                    float previousProgress = curriculumProgress;
                    curriculumProgress = Mathf.Max(curriculumProgress - curriculumExpansionStep, 0f);
                    UpdateCurriculumSpawnBounds();

                    Debug.Log($"[Curriculum] REGRESSED from {previousProgress:P0} to {curriculumProgress:P0}! " +
                             $"{consecutiveLowSuccessCount} consecutive checkpoints below {curriculumRegressionThreshold:P0}. " +
                             $"Cube bounds: [{currentCubeSpawnMin.x:F2},{currentCubeSpawnMin.y:F2}] to [{currentCubeSpawnMax.x:F2},{currentCubeSpawnMax.y:F2}], " +
                             $"DropZone bounds: [{currentDropZoneSpawnMin.x:F2},{currentDropZoneSpawnMin.y:F2}] to [{currentDropZoneSpawnMax.x:F2},{currentDropZoneSpawnMax.y:F2}]");

                    // Reset counters after regression
                    consecutiveLowSuccessCount = 0;
                    recentCheckpointSuccessRates.Clear();
                }
            }
            else
            {
                consecutiveLowSuccessCount = 0;  // Reset on any success above threshold
            }
        }

        // Check if we should advance curriculum
        // Require ALL checkpoints in window to exceed threshold (not just average)
        if (recentCheckpointSuccessRates.Count >= curriculumCheckpointWindow)
        {
            float minSuccess = recentCheckpointSuccessRates.Min();

            if (minSuccess >= curriculumSuccessThreshold && curriculumProgress < 1f)
            {
                float previousProgress = curriculumProgress;
                curriculumProgress = Mathf.Min(curriculumProgress + curriculumExpansionStep, 1f);
                UpdateCurriculumSpawnBounds();

                Debug.Log($"[Curriculum] Advanced from {previousProgress:P0} to {curriculumProgress:P0}! " +
                         $"ALL {curriculumCheckpointWindow} checkpoints exceeded {curriculumSuccessThreshold:P0} (min was {minSuccess:P0}). " +
                         $"Cube bounds: [{currentCubeSpawnMin.x:F2},{currentCubeSpawnMin.y:F2}] to [{currentCubeSpawnMax.x:F2},{currentCubeSpawnMax.y:F2}], " +
                         $"DropZone bounds: [{currentDropZoneSpawnMin.x:F2},{currentDropZoneSpawnMin.y:F2}] to [{currentDropZoneSpawnMax.x:F2},{currentDropZoneSpawnMax.y:F2}]");

                // Reset counters after advancing
                consecutiveLowSuccessCount = 0;
                recentCheckpointSuccessRates.Clear();
            }
        }
    }

    /// <summary>
    /// Updates spawn bounds based on current curriculum progress.
    /// Progress 0 = start bounds (easy), Progress 1 = end bounds (full difficulty)
    /// </summary>
    private void UpdateCurriculumSpawnBounds()
    {
        // Lerp cube bounds
        currentCubeSpawnMin = Vector2.Lerp(curriculumCubeStartMin, curriculumCubeEndMin, curriculumProgress);
        currentCubeSpawnMax = Vector2.Lerp(curriculumCubeStartMax, curriculumCubeEndMax, curriculumProgress);

        // Lerp drop zone bounds
        currentDropZoneSpawnMin = Vector2.Lerp(curriculumDropZoneStartMin, curriculumDropZoneEndMin, curriculumProgress);
        currentDropZoneSpawnMax = Vector2.Lerp(curriculumDropZoneStartMax, curriculumDropZoneEndMax, curriculumProgress);
    }

    /// <summary>
    /// Initializes curriculum spawn bounds at the start of training.
    /// </summary>
    private void InitializeCurriculum()
    {
        if (useCurriculum)
        {
            curriculumProgress = 0f;
            recentCheckpointSuccessRates.Clear();
            UpdateCurriculumSpawnBounds();
            Debug.Log($"[Curriculum] Initialized at {curriculumProgress:P0}. " +
                     $"Cube: [{currentCubeSpawnMin}] to [{currentCubeSpawnMax}], " +
                     $"DropZone: [{currentDropZoneSpawnMin}] to [{currentDropZoneSpawnMax}]");
        }
    }

    /// <summary>
    /// Gets the current curriculum progress (0 to 1).
    /// </summary>
    public float GetCurriculumProgress() => curriculumProgress;

    #endregion

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

        // Gripper position (normalized)
        float gripperPosition = 0f;
        if (leftGripper != null)
        {
            var drive = leftGripper.xDrive;
            gripperPosition = (drive.target - drive.lowerLimit) / (drive.upperLimit - drive.lowerLimit) * 2f - 1f;
        }
        currentObservations["GripperPosition"] = gripperPosition; // @obs: Normalized gripper position (-1 = fully closed, 1 = fully open)

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

        // Apply gripper control (7th action) via GripperController
        if (actions.ContinuousActions.Length > 6 && gripperController != null && leftGripper != null)
        {
            float gripperAction = actions.ContinuousActions[6];
            float gripperDelta = gripperAction * gripperActionSpeed * Time.fixedDeltaTime;

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

        // Joint controls
        if (continuousActionsOut.Length >= 6)
        {
            continuousActionsOut[0] = Input.GetKey(KeyCode.A) ? -1.0f : (Input.GetKey(KeyCode.D) ? 1.0f : 0.0f);
            continuousActionsOut[1] = Input.GetKey(KeyCode.W) ? 1.0f : (Input.GetKey(KeyCode.S) ? -1.0f : 0.0f);
            continuousActionsOut[2] = Input.GetKey(KeyCode.Q) ? 1.0f : (Input.GetKey(KeyCode.E) ? -1.0f : 0.0f);
            continuousActionsOut[3] = Input.GetKey(KeyCode.UpArrow) ? 1.0f : (Input.GetKey(KeyCode.DownArrow) ? -1.0f : 0.0f);
            continuousActionsOut[4] = Input.GetKey(KeyCode.LeftArrow) ? -1.0f : (Input.GetKey(KeyCode.RightArrow) ? 1.0f : 0.0f);
            continuousActionsOut[5] = Input.GetKey(KeyCode.Z) ? 1.0f : (Input.GetKey(KeyCode.X) ? -1.0f : 0.0f);
        }

        // Gripper control: G = close (negative), H = open (positive)
        if (continuousActionsOut.Length >= 7)
        {
            continuousActionsOut[6] = Input.GetKey(KeyCode.G) ? -1.0f : (Input.GetKey(KeyCode.H) ? 1.0f : 0.0f);
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
        // Get Y heights
        float cubeY = Application.isPlaying ? initialCubePosition.y : (targetCube != null ? targetCube.position.y : 1f);
        float dropZoneY = Application.isPlaying ? initialDropZonePosition.y : (dropZone != null ? dropZone.position.y : 1f);

        if (useCurriculum)
        {
            // Draw curriculum bounds relative to agent transform
            // Use current bounds if playing, otherwise use start bounds for editor preview
            Vector2 cubeMin = Application.isPlaying ? currentCubeSpawnMin : curriculumCubeStartMin;
            Vector2 cubeMax = Application.isPlaying ? currentCubeSpawnMax : curriculumCubeStartMax;
            Vector2 dzMin = Application.isPlaying ? currentDropZoneSpawnMin : curriculumDropZoneStartMin;
            Vector2 dzMax = Application.isPlaying ? currentDropZoneSpawnMax : curriculumDropZoneStartMax;

            // Draw END bounds first (faint, filled) - target difficulty
            DrawAgentRelativeSpawnArea(curriculumCubeEndMin, curriculumCubeEndMax, cubeY, new Color(0f, 1f, 0f, 0.2f), fillArea: true);
            DrawAgentRelativeSpawnArea(curriculumDropZoneEndMin, curriculumDropZoneEndMax, dropZoneY + 0.01f, new Color(0f, 0.5f, 1f, 0.2f), fillArea: true);

            // Draw current spawn area on top (bright outline only, no fill)
            DrawAgentRelativeSpawnArea(cubeMin, cubeMax, cubeY + 0.02f, new Color(0f, 1f, 0f, 1f), fillArea: false);
            DrawAgentRelativeSpawnArea(dzMin, dzMax, dropZoneY + 0.03f, new Color(0f, 0.5f, 1f, 1f), fillArea: false);
        }
        else if (randomizeSpawnPositions)
        {
            // Original behavior: shared spawn area
            DrawAgentRelativeSpawnArea(
                new Vector2(spawnAreaMin.x, spawnAreaMin.z),
                new Vector2(spawnAreaMax.x, spawnAreaMax.z),
                cubeY, new Color(0f, 1f, 0f, 0.3f));
            DrawAgentRelativeSpawnArea(
                new Vector2(spawnAreaMin.x, spawnAreaMin.z),
                new Vector2(spawnAreaMax.x, spawnAreaMax.z),
                dropZoneY + 0.01f, new Color(0f, 0.5f, 1f, 0.3f));
        }

        // Draw minimum distance circle at agent position
        Gizmos.color = new Color(1f, 0.5f, 0f, 0.5f);
        DrawWireCircle(transform.position + Vector3.up * cubeY, minCubeDropZoneDistance / 2f, 32);
    }

    /// <summary>
    /// Draws a spawn area relative to agent transform.
    /// </summary>
    private void DrawAgentRelativeSpawnArea(Vector2 boundsMin, Vector2 boundsMax, float yHeight, Color color, bool fillArea = true)
    {
        // Convert local bounds to world positions
        Vector3 bl = transform.TransformPoint(new Vector3(boundsMin.x, 0f, boundsMin.y));
        Vector3 br = transform.TransformPoint(new Vector3(boundsMax.x, 0f, boundsMin.y));
        Vector3 tl = transform.TransformPoint(new Vector3(boundsMin.x, 0f, boundsMax.y));
        Vector3 tr = transform.TransformPoint(new Vector3(boundsMax.x, 0f, boundsMax.y));

        bl.y = yHeight; br.y = yHeight; tl.y = yHeight; tr.y = yHeight;

        // Draw outline
        Gizmos.color = color;
        Gizmos.DrawLine(bl, br);
        Gizmos.DrawLine(br, tr);
        Gizmos.DrawLine(tr, tl);
        Gizmos.DrawLine(tl, bl);

        // Draw semi-transparent fill if requested and area has size
        float width = boundsMax.x - boundsMin.x;
        float depth = boundsMax.y - boundsMin.y;
        if (fillArea && width > 0.001f && depth > 0.001f)
        {
            Gizmos.color = new Color(color.r, color.g, color.b, 0.15f);
            Vector3 center = (bl + br + tl + tr) * 0.25f;
            Vector3 size = new Vector3(Vector3.Distance(bl, br), 0.01f, Vector3.Distance(bl, tl));

            Matrix4x4 oldMatrix = Gizmos.matrix;
            Gizmos.matrix = Matrix4x4.TRS(center, transform.rotation, Vector3.one);
            Gizmos.DrawCube(Vector3.zero, size);
            Gizmos.matrix = oldMatrix;
        }
    }

    private void DrawSpawnArea(Vector3 bl, Vector3 br, Vector3 tl, Vector3 tr, float y, Color color, string label)
    {
        bl.y = y; br.y = y; tl.y = y; tr.y = y;

        Gizmos.color = color;
        Gizmos.DrawLine(bl, br);
        Gizmos.DrawLine(br, tr);
        Gizmos.DrawLine(tr, tl);
        Gizmos.DrawLine(tl, bl);

        // Draw semi-transparent fill
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
