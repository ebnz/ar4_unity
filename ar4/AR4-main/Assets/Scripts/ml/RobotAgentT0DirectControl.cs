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

/// @task-name: T0DirectControl
/// <summary>
/// RobotAgent for T0 with direct joint control (no IK) and dynamic reward function evaluation via RewardFunctionManager
/// </summary>
/// <taskdescription>
/// T0 Task: Move the robot end effector to reach a fixed target position.
/// The agent directly controls each joint angle to move the end effector to the goal.
/// Success is achieved when the end effector reaches within 0.1 units of the goal position.
/// </taskdescription>
/// <successcriteria>
/// Distance between end effector and goal position < 0.1 units
/// </successcriteria>
/// <actionspace>
/// Continuous Actions (6): [j1_delta, j2_delta, j3_delta, j4_delta, j5_delta, j6_delta] - Direct joint angle deltas
/// </actionspace>
public class RobotAgentT0DirectControl : Agent, IEurekaAgent
{
    [Header("References")]
    public ArticulationBody[] robotJoints;
    public Transform endEffector;
    public Transform goalPosition;

    [Header("Settings")]
    [Tooltip("Speed multiplier for joint movements")]
    public float jointActionSpeed = 50f;

    private ObservationNormalizer observationNormalizer;


    [Header("Reward Tracking")]
    private Dictionary<string, float> rewardsByType = new Dictionary<string, float>();

    private Vector3 initialGoalPosition;
    private int currentEpisodeSteps = 0;
    private float previousDistanceToGoal = 0f;
    private Dictionary<string, float> currentObservations = new Dictionary<string, float>();

    // Reference to stats logger for Eureka integration
    private TrainingStatsLogger statsLogger;

    private BehaviorParameters behaviorParameters;

    // Store initial joint positions for reset
    private float[] initialJointPositions;

    async void Start()
    {
        // Validate required references
        if (robotJoints == null || robotJoints.Length == 0)
        {
            Debug.LogError("RobotAgentT0DirectControl: robotJoints not assigned!");
        }

        if (endEffector == null)
        {
            Debug.LogError("RobotAgentT0DirectControl: endEffector not assigned!");
        }

        if (goalPosition != null)
            initialGoalPosition = goalPosition.position;

        // Store initial joint positions for reset
        initialJointPositions = new float[robotJoints.Length];
        for (int i = 0; i < robotJoints.Length; i++)
        {
            initialJointPositions[i] = robotJoints[i].xDrive.target;
        }

        statsLogger = FindFirstObjectByType<TrainingStatsLogger>();
        observationNormalizer = GetComponent<ObservationNormalizer>();

        // Wait for RewardFunctionManager async initialization to complete
        Debug.Log("RobotAgentT0DirectControl waiting for RewardFunctionManager initialization...");
        while (!RewardFunctionManager.IsInitialized)
        {
            await Task.Yield(); // Wait one frame
        }
        Debug.Log("RobotAgentT0DirectControl: RewardFunctionManager ready, starting agent...");

        behaviorParameters = GetComponent<BehaviorParameters>();
    }


    public override void OnEpisodeBegin()
    {
        currentEpisodeSteps = 0;
        rewardsByType.Clear();

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

        goalPosition.position = initialGoalPosition;

        previousDistanceToGoal = Vector3.Distance(endEffector.position, goalPosition.position);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        currentObservations.Clear();

        // End effector to goal vector
        Vector3 endEffectorToGoal = goalPosition.position - endEffector.position;
        currentObservations["EndEffectorToGoal_X"] = endEffectorToGoal.x; // @obs: X component of vector from end effector to goal position
        currentObservations["EndEffectorToGoal_Y"] = endEffectorToGoal.y; // @obs: Y component of vector from end effector to goal position
        currentObservations["EndEffectorToGoal_Z"] = endEffectorToGoal.z; // @obs: Z component of vector from end effector to goal position

        float distanceToGoal = Vector3.Distance(endEffector.position, goalPosition.position);
        currentObservations["DistanceToGoal"] = distanceToGoal; // @obs: Euclidean distance from end effector to goal position

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
            endEffectorToGoal.x,
            endEffectorToGoal.y,
            endEffectorToGoal.z,
            distanceToGoal
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
        // Apply direct joint control
        // Each action controls the delta for one joint
        for (int i = 0; i < robotJoints.Length && i < actions.ContinuousActions.Length; i++)
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

        // Execute dynamic reward function
        if (!behaviorParameters.IsInHeuristicMode() && RewardFunctionManager.IsInitialized)
        {
            ExecuteRewardFunction();
        }

        // Check if goal reached
        if (currentObservations["DistanceToGoal"] < ConfigManager.Instance.GoalThreshold)
        {
            EndDynamicEpisode();
        }

        previousDistanceToGoal = currentObservations["DistanceToGoal"];

        if (currentEpisodeSteps >= ConfigManager.Instance.MaxStepsPerEpisode)
        {
            AddRewardWithType("Timeout", -1.0f);
            EndDynamicEpisode();
        }
    }

    // Not an override; use a custom method name
    public void EndDynamicEpisode()
    {
        // Determine if episode was successful based on goal achievement
        bool isSuccess = currentObservations.ContainsKey("DistanceToGoal") &&
                         currentObservations["DistanceToGoal"] < ConfigManager.Instance.GoalThreshold;

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
                ["previousDistance"] = previousDistanceToGoal, // @context: Distance to goal in previous timestep
                ["goalThreshold"] = ConfigManager.Instance.GoalThreshold, // @context: Success threshold distance (0.1f)
                ["episodeSteps"] = currentEpisodeSteps, // @context: Current number of steps in this episode
                ["maxSteps"] = ConfigManager.Instance.MaxStepsPerEpisode, // @context: Maximum steps per episode (2000)
                ["timeStep"] = Time.fixedDeltaTime, // @context: Fixed timestep duration (0.02f)
                ["endEffectorPosition"] = endEffector.position, // @context: Current end effector position (Vector3)
                ["goalPosition"] = goalPosition.position // @context: Goal position (Vector3)
            };

            // Add joint positions to context
            for (int i = 0; i < robotJoints.Length; i++)
            {
                context[$"joint{i}Position"] = robotJoints[i].xDrive.target;
            }

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

        // Map keyboard to specific joints for manual control
        // Joint 0 (base rotation): A/D keys
        continuousActionsOut[0] = Input.GetKey(KeyCode.A) ? -1.0f : (Input.GetKey(KeyCode.D) ? 1.0f : 0.0f);

        // Joint 1 (shoulder): W/S keys
        continuousActionsOut[1] = Input.GetKey(KeyCode.W) ? 1.0f : (Input.GetKey(KeyCode.S) ? -1.0f : 0.0f);

        // Joint 2 (elbow): Q/E keys
        continuousActionsOut[2] = Input.GetKey(KeyCode.Q) ? 1.0f : (Input.GetKey(KeyCode.E) ? -1.0f : 0.0f);

        // Joint 3 (wrist pitch): Arrow Up/Down
        continuousActionsOut[3] = Input.GetKey(KeyCode.UpArrow) ? 1.0f : (Input.GetKey(KeyCode.DownArrow) ? -1.0f : 0.0f);

        // Joint 4 (wrist roll): Arrow Left/Right
        continuousActionsOut[4] = Input.GetKey(KeyCode.LeftArrow) ? -1.0f : (Input.GetKey(KeyCode.RightArrow) ? 1.0f : 0.0f);

        // Joint 5 (wrist yaw): Z/X keys
        continuousActionsOut[5] = Input.GetKey(KeyCode.Z) ? 1.0f : (Input.GetKey(KeyCode.X) ? -1.0f : 0.0f);
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
    /// Called when a decision is requested (not on action repetitions)
    /// Increments episode step counter
    /// </summary>
    public void OnDecisionRequested()
    {
        currentEpisodeSteps++;
    }
}
