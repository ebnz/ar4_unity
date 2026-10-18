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

/// @task-name: T2
/// <summary>
/// RobotAgent for T2 with dynamic reward function evaluation via RewardFunctionManager
/// </summary>
/// <taskdescription>
/// T2 Task: Pick up a cube from a fixed location and transfer it to another fixed drop zone.
/// The agent controls the target position that the inverse kinematics system follows and gripper open/close actions.
/// Success is achieved when the cube is placed within the drop zone area.
/// </taskdescription>
/// <successcriteria>
/// Cube is positioned within 0.2 units of the drop zone center and gripper is open
/// </successcriteria>
/// <actionspace>
/// Continuous Actions (3): [x_delta, y_delta, z_delta] - Target position deltas
/// Discrete Actions (1): [gripper_action] where 0=no_action, 1=open_gripper, 2=close_gripper  
/// </actionspace>
public class RobotAgentT2 : Agent, IEurekaAgent
{
    [Header("References")]
    public RobotManager robotManager;
    public Transform dropZone;
    public Transform targetCube;  // Direct reference to cube instead of spawn location
    public GripperController gripperController;
    
    [Header("Settings")]
    private Vector3 smoothedAction;
    private bool gripperOpen = true;
    private ObservationNormalizer observationNormalizer;
    
    [Header("Task Objects")]
    private bool cubeIsGrasped = false;
    
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
    /// Distance threshold for cube grasping detection
    /// </summary>
    public float graspThreshold = 0.15f;
    
    /// <summary>
    /// Distance threshold for successful drop zone placement
    /// </summary>
    public float dropZoneThreshold = 0.2f;
    
    async void Start()
    {
        if (robotManager == null)
            robotManager = FindFirstObjectByType<RobotManager>();
            
        if (gripperController == null)
            gripperController = FindFirstObjectByType<GripperController>();
            
        if (dropZone != null)
            initialDropZonePosition = dropZone.position;
            
        if (targetCube != null)
            initialCubePosition = targetCube.position;
        else
        {
            Debug.LogError("Target cube not assigned to RobotAgentT2!");
            throw new System.InvalidOperationException("Target cube must be assigned to RobotAgentT2!");
        }

        statsLogger = FindFirstObjectByType<TrainingStatsLogger>();
        observationNormalizer = GetComponent<ObservationNormalizer>();

        // Wait for RewardFunctionManager async initialization to complete
        Debug.Log("RobotAgentT2 waiting for RewardFunctionManager initialization...");
        while (!RewardFunctionManager.IsInitialized)
        {
            await Task.Yield(); // Wait one frame
        }
        Debug.Log("RobotAgentT2: RewardFunctionManager ready, starting agent...");

        behaviorParameters = GetComponent<BehaviorParameters>();
    }
    
    
    public override void OnEpisodeBegin()
    {
        currentEpisodeSteps = 0;
        rewardsByType.Clear();
        cubeIsGrasped = false;
        gripperOpen = true;
        
        // Reset gripper controller to open state
        if (gripperController != null)
        {
            gripperController.OpenGrippers();
        }
        
        robotManager.ResetRobotInstantly();
        
        // Reset drop zone position
        dropZone.position = initialDropZonePosition;
        
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
        
        // Calculate initial distances
        if (targetCube != null)
        {
            previousDistanceToCube = Vector3.Distance(robotManager.target.position, targetCube.position);
            previousDistanceToDropZone = Vector3.Distance(targetCube.position, dropZone.position);
        }
    }
    
    public override void CollectObservations(VectorSensor sensor)
    {
        currentObservations.Clear();
        
            // End effector to cube vector
        Vector3 targetToCube = targetCube.position - robotManager.target.position;
        currentObservations["TargetToCube_X"] = targetToCube.x; // @obs: X component of vector from end effector to cube
        currentObservations["TargetToCube_Y"] = targetToCube.y; // @obs: Y component of vector from end effector to cube
        currentObservations["TargetToCube_Z"] = targetToCube.z; // @obs: Z component of vector from end effector to cube
        
        // Distance from end effector to cube
        float distanceToCube = Vector3.Distance(robotManager.target.position, targetCube.position);
        currentObservations["DistanceToCube"] = distanceToCube; // @obs: Euclidean distance from end effector to cube
        
        // Cube to drop zone vector
        Vector3 cubeToDropZone = dropZone.position - targetCube.position;
        currentObservations["CubeToDropZone_X"] = cubeToDropZone.x; // @obs: X component of vector from cube to drop zone
        currentObservations["CubeToDropZone_Y"] = cubeToDropZone.y; // @obs: Y component of vector from cube to drop zone
        currentObservations["CubeToDropZone_Z"] = cubeToDropZone.z; // @obs: Z component of vector from cube to drop zone
        
        // Distance from cube to drop zone
        float distanceToDropZone = Vector3.Distance(targetCube.position, dropZone.position);
        currentObservations["DistanceToDropZone"] = distanceToDropZone; // @obs: Euclidean distance from cube to drop zone
    
        
        // Gripper state - sync with actual controller if available
        if (gripperController != null)
        {
            // Consider gripper open if position is above a threshold (close to upper limit)
            float normalizedPosition = gripperController.currentPosition / gripperController.leftGripper.xDrive.upperLimit;
            gripperOpen = normalizedPosition > 0.5f; // Threshold for considering gripper "open"
        }
        currentObservations["GripperOpen"] = gripperOpen ? 1f : 0f; // @obs: 1 if gripper is open, 0 if closed
        
        // Cube grasped state
        currentObservations["CubeGrasped"] = cubeIsGrasped ? 1f : 0f; // @obs: 1 if cube is grasped, 0 if not

        // Collect raw observations
        float[] rawObs = new float[] {
            currentObservations["TargetToCube_X"],
            currentObservations["TargetToCube_Y"],
            currentObservations["TargetToCube_Z"],
            currentObservations["DistanceToCube"],
            currentObservations["CubeToDropZone_X"],
            currentObservations["CubeToDropZone_Y"],
            currentObservations["CubeToDropZone_Z"],
            currentObservations["DistanceToDropZone"],
            currentObservations["GripperOpen"],
            currentObservations["CubeGrasped"]
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
        if (actions.DiscreteActions[0] == 1)
        {
            // Open gripper (only if currently closed)
            gripperOpen = true;
            if (gripperController != null)
            {
                gripperController.OpenGrippers();
            }
        }
        else if (actions.DiscreteActions[0] == 2)
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
        
        // Clamp position
        if (TableTop != null)
        {
            Bounds tableBounds = TableTop.bounds;
            robotManager.target.position = new Vector3(
                Mathf.Clamp(robotManager.target.position.x, tableBounds.min.x, tableBounds.max.x),
                Mathf.Clamp(robotManager.target.position.y, tableBounds.min.y, tableBounds.max.y + 3f),
                Mathf.Clamp(robotManager.target.position.z, tableBounds.min.z, tableBounds.max.z)
            );
        }
        
        // Handle cube grasping/releasing
        UpdateCubeGrasping();
        
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
    
    private void UpdateCubeGrasping()
    {
        if (targetCube == null) return;
        
        float distanceToCube = currentObservations["DistanceToCube"];
        
        // Check if we should grasp the cube
        if (!cubeIsGrasped && !gripperOpen && distanceToCube < graspThreshold)
        {
            cubeIsGrasped = true;
            // Attach cube to end effector
            //targetCube.SetParent(robotManager.target);
            
            // Disable physics while grasped
            //Rigidbody cubeRb = targetCube.GetComponent<Rigidbody>();
            //if (cubeRb != null)
            //{
            //    cubeRb.isKinematic = true;
            //}
        }
        // Check if we should release the cube
        else if (cubeIsGrasped && gripperOpen)
        {
            cubeIsGrasped = false;
            // Detach cube from end effector
            //targetCube.SetParent(null);
            
            // Re-enable physics
            //Rigidbody cubeRb = targetCube.GetComponent<Rigidbody>();
            //if (cubeRb != null)
            //{
            //    cubeRb.isKinematic = false;
            //}
        }
    }
    
    private bool IsTaskComplete()
    {
        if (targetCube == null) return false;
        
        // Task is complete when cube is in drop zone and gripper is open (cube released)
        bool cubeInDropZone = currentObservations["DistanceToDropZone"] < dropZoneThreshold;
        bool gripperIsOpen = gripperOpen;
        bool cubeNotGrasped = !cubeIsGrasped;
        
        return cubeInDropZone && gripperIsOpen && cubeNotGrasped;
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
                ["previousDistanceToCube"] = previousDistanceToCube, // @context: Distance to cube in previous timestep
                ["previousDistanceToDropZone"] = previousDistanceToDropZone, // @context: Distance from cube to drop zone in previous timestep
                ["graspThreshold"] = graspThreshold, // @context: Distance threshold for grasping cube (0.15f)
                ["dropZoneThreshold"] = dropZoneThreshold, // @context: Distance threshold for successful placement (0.2f)
                ["episodeSteps"] = currentEpisodeSteps, // @context: Current number of steps in this episode
                ["maxSteps"] = ConfigManager.Instance.MaxStepsPerEpisode, // @context: Maximum steps per episode (2000)
                ["timeStep"] = Time.fixedDeltaTime, // @context: Fixed timestep duration (0.02f)
                ["targetPosition"] = robotManager.target.position, // @context: Current target position (Vector3)
                ["dropZonePosition"] = dropZone.position, // @context: Drop zone position (Vector3)
                ["cubePosition"] = targetCube != null ? targetCube.position : Vector3.zero, // @context: Current cube position (Vector3)
                ["cubeSpawnPosition"] = initialCubePosition, // @context: Initial cube spawn position (Vector3)
                ["gripperOpen"] = gripperOpen, // @context: Whether gripper is currently open (bool)
                ["cubeGrasped"] = cubeIsGrasped, // @context: Whether cube is currently grasped (bool)
                ["isSuccess"] = IsTaskComplete() // @context: Whether the task has been successfully completed (bool)
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
        continuousActionsOut[0] = Input.GetAxis("Horizontal");
        continuousActionsOut[1] = Input.GetKey(KeyCode.E) ? 1.0f : (Input.GetKey(KeyCode.Q) ? -1.0f : 0.0f);
        continuousActionsOut[2] = Input.GetAxis("Vertical");
        
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
    public bool IsCubeGrasped() => cubeIsGrasped;
    
    /// <summary>
    /// Check if gripper is currently open
    /// </summary>
    public bool IsGripperOpen() => gripperOpen;

    /// <summary>
    /// Called when a decision is requested (not on action repetitions)
    /// Increments episode step counter
    /// </summary>
    public void OnDecisionRequested()
    {
        currentEpisodeSteps++;
    }
}