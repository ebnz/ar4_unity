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

/// @task-name: T0Discrete
/// <summary>
/// RobotAgent for T0 with discrete actions and dynamic reward function evaluation via RewardFunctionManager
/// </summary>
/// <taskdescription>
/// T0 Task: Move the robot end effector to reach a fixed target position.
/// The agent controls the target position that the inverse kinematics system follows.
/// Success is achieved when the end effector reaches within 0.1 units of the goal position.
/// </taskdescription>
/// <successcriteria>
/// Distance between end effector and goal position < 0.1 units
/// </successcriteria>
/// <actionspace>
/// Discrete Actions (3 branches with 3 actions each):
/// - Branch 0 (X-axis): [0=left, 1=stay, 2=right]
/// - Branch 1 (Y-axis): [0=down, 1=stay, 2=up]
/// - Branch 2 (Z-axis): [0=backward, 1=stay, 2=forward]
/// </actionspace>
public class RobotAgentT0Discrete : Agent, IEurekaAgent
{
    [Header("References")]
    public RobotManager robotManager;
    public Transform goalPosition;

    [Header("Settings")]
    private Vector3 smoothedAction;
    private ObservationNormalizer observationNormalizer;


    [Header("Reward Tracking")]
    private Dictionary<string, float> rewardsByType = new Dictionary<string, float>();

    private Vector3 initialGoalPosition;
    private int currentEpisodeSteps = 0;
    private float previousDistanceToGoal = 0f;
    private Dictionary<string, float> currentObservations = new Dictionary<string, float>();

    public BoxCollider TableTop;

    // Reference to stats logger for Eureka integration
    private TrainingStatsLogger statsLogger;

    private BehaviorParameters behaviorParameters;

    async void Start()
    {
        if (robotManager == null)
            robotManager = FindFirstObjectByType<RobotManager>();

        if (goalPosition != null)
            initialGoalPosition = goalPosition.position;

        statsLogger = FindFirstObjectByType<TrainingStatsLogger>();
        observationNormalizer = GetComponent<ObservationNormalizer>();

        // Wait for RewardFunctionManager async initialization to complete
        Debug.Log("RobotAgentT0Discrete waiting for RewardFunctionManager initialization...");
        while (!RewardFunctionManager.IsInitialized)
        {
            await Task.Yield(); // Wait one frame
        }
        Debug.Log("RobotAgentT0Discrete: RewardFunctionManager ready, starting agent...");

        behaviorParameters = GetComponent<BehaviorParameters>();
    }


    public override void OnEpisodeBegin()
    {
        currentEpisodeSteps = 0;
        rewardsByType.Clear();

        robotManager.ResetRobotInstantly();
        goalPosition.position = initialGoalPosition;

        previousDistanceToGoal = Vector3.Distance(robotManager.target.position, goalPosition.position);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        currentObservations.Clear();

        Vector3 targetToGoal = goalPosition.position - robotManager.target.position;
        currentObservations["TargetToGoal_X"] = targetToGoal.x; // @obs: X component of vector from current target to goal position
        currentObservations["TargetToGoal_Y"] = targetToGoal.y; // @obs: Y component of vector from current target to goal position
        currentObservations["TargetToGoal_Z"] = targetToGoal.z; // @obs: Z component of vector from current target to goal position

        float distanceToGoal = Vector3.Distance(robotManager.target.position, goalPosition.position);
        currentObservations["DistanceToGoal"] = distanceToGoal; // @obs: Euclidean distance from end effector to goal position

        // Collect raw observations
        float[] rawObs = new float[] {
            targetToGoal.x,
            targetToGoal.y,
            targetToGoal.z,
            distanceToGoal
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
        // Convert discrete actions to continuous values
        // Each branch has 3 actions: [negative, stay, positive] -> maps to [-1, 0, 1]
        Vector3 targetDeltaAction = new Vector3(
            actions.DiscreteActions[0] - 1,  // X-axis: 0->-1, 1->0, 2->1
            actions.DiscreteActions[1] - 1,  // Y-axis: 0->-1, 1->0, 2->1
            actions.DiscreteActions[2] - 1   // Z-axis: 0->-1, 1->0, 2->1
        );

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
                ["targetPosition"] = robotManager.target.position, // @context: Current target position (Vector3)
                ["goalPosition"] = goalPosition.position // @context: Goal position (Vector3)
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
        var discreteActionsOut = actionsOut.DiscreteActions;

        // X-axis (Horizontal): Left=0, Stay=1, Right=2
        float horizontal = Input.GetAxis("Horizontal");
        discreteActionsOut[0] = horizontal < -0.1f ? 0 : (horizontal > 0.1f ? 2 : 1);

        // Y-axis (E/Q keys): Down=0, Stay=1, Up=2
        discreteActionsOut[1] = Input.GetKey(KeyCode.Q) ? 0 : (Input.GetKey(KeyCode.E) ? 2 : 1);

        // Z-axis (Vertical): Backward=0, Stay=1, Forward=2
        float vertical = Input.GetAxis("Vertical");
        discreteActionsOut[2] = vertical < -0.1f ? 0 : (vertical > 0.1f ? 2 : 1);
    }

    /*void OnDestroy()
    {
        // Final stats logging handled by TrainingStatsLogger
        if (statsLogger != null && ConfigManager.Instance.UseFileBasedReward)
        {
            statsLogger.WriteEurekaResults();
        }
    }*/

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
