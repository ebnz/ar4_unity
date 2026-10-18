/*
 * RobotAgent.cs
 * 
 * This file implements a Machine Learning Agent for controlling a robotic arm
 * in a simple target reaching task using Unity ML-Agents framework. The agent learns
 * to control a robotic arm to reach a fixed target position.
 */

using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using TMPro;
using System.Collections.Generic;

/// <summary>
/// <taskdescription>
/// RobotAgent handles the ML training process for a robotic arm,
/// implementing target reaching behavior using reinforcement learning.
/// </taskdescription>
/// <successcriteria>
/// Reach target within 0.1 units distance
/// </successcriteria>
/// </summary>
public class RobotAgent : Agent
{
    [Header("References")]
    /// <summary>
    /// Reference to the RobotManager component.
    /// </summary>
    public RobotManager robotManager;
    public Transform goalPosition; 
    
    [Header("Settings")]
    private float actionSpeed = 0.5f;
    private float goalThreshold = 0.1f;
    private float actionSmoothing = 0.3f;
    private Vector3 smoothedAction;
    
    [Header("Reward Tracking")]
    private Dictionary<string, float> rewardsByType = new Dictionary<string, float>();
    
    private Vector3 initialGoalPosition;
    private int currentEpisodeSteps = 0;
    private int maxStepsPerEpisode = 2000;
    private float previousDistanceToGoal = 0f;

    // Add observation tracking
    private Dictionary<string, float> currentObservations = new Dictionary<string, float>();

    public delegate void EpisodeEndHandler(RobotAgent agent, int finalStepCount);
    public event EpisodeEndHandler OnEpisodeEnd;
    public BoxCollider TableTop;

    void Start()
    {
        if (robotManager == null)
            robotManager = FindFirstObjectByType<RobotManager>();
            
        if (goalPosition == null)
            Debug.LogError("Goal position not assigned to RobotAgent!");
            
        if (goalPosition != null)
            initialGoalPosition = goalPosition.position;
    }

    private void ResetRewardTracking()
    {
        rewardsByType.Clear();
    }
    
    private void AddObservation(string name, float value, VectorSensor sensor)
    {
        currentObservations[name] = value;
        sensor.AddObservation(value);
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

    public Dictionary<string, float> GetRewardsByType()
    {
        return new Dictionary<string, float>(rewardsByType);
    }

    public override void OnEpisodeBegin()
    {
        currentEpisodeSteps = 0;
        
        // Reset reward tracking for the new episode
        ResetRewardTracking();
        
        // Reset the robot position instantly
        robotManager.ResetRobotInstantly();
        
        // Set goal position to fixed location
        goalPosition.position = initialGoalPosition;

        // Calculate initial distance
        previousDistanceToGoal = Vector3.Distance(robotManager.target.position, goalPosition.position);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        currentObservations.Clear();
        
        // Robot base to end effector (gripper[0]) - helps understand robot configuration
        // Vector3 relativeBaseToGripper = robotManager.gripper[0].position - robotManager.robotJoints[0].transform.position;
        // AddObservation("RelativeBaseToGripper_X", relativeBaseToGripper.x, sensor);
        //AddObservation("RelativeBaseToGripper_Y", relativeBaseToGripper.y, sensor);
        //AddObservation("RelativeBaseToGripper_Z", relativeBaseToGripper.z, sensor);

        // Target to gripper[0] - actual vs desired position
        //Vector3 targetToGripper = robotManager.target.position - robotManager.gripper[0].position;
        //AddObservation("TargetToGripper_X", targetToGripper.x, sensor);
        //AddObservation("TargetToGripper_Y", targetToGripper.y, sensor);
        //AddObservation("TargetToGripper_Z", targetToGripper.z, sensor);

        // Target to goal - main objective
        Vector3 targetToGoal = goalPosition.position - robotManager.target.position;
        AddObservation("TargetToGoal_X", targetToGoal.x, sensor);
        AddObservation("TargetToGoal_Y", targetToGoal.y, sensor);
        AddObservation("TargetToGoal_Z", targetToGoal.z, sensor);

        // Distance to goal
        float distanceToGoal = Vector3.Distance(robotManager.target.position, goalPosition.position);
        AddObservation("DistanceToGoal", distanceToGoal, sensor);
    }
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        /* APPLY ACTIONS */

        // Get continuous actions with smoothing
        Vector3 targetDeltaAction = new Vector3(
            actions.ContinuousActions[0],
            actions.ContinuousActions[1],
            actions.ContinuousActions[2]
        );

        // Apply the actions to move the robot
        // Smooth the action to prevent jerky movements
        smoothedAction = Vector3.Lerp(smoothedAction, targetDeltaAction, actionSmoothing);
        
        // Apply movement to the target position
        Vector3 targetDelta = smoothedAction * actionSpeed * Time.fixedDeltaTime;
        robotManager.target.position += targetDelta;

        // Clamp target position within the TableTop bounds
        if (TableTop != null)
        {
            Bounds tableBounds = TableTop.bounds;

            float xMin = tableBounds.min.x;
            float xMax = tableBounds.max.x;
            float zMin = tableBounds.min.z;
            float zMax = tableBounds.max.z;
            float yMin = tableBounds.min.y;
            float yMax = tableBounds.max.y + 3f; // Allow movement above the table

            robotManager.target.position = new Vector3(
                Mathf.Clamp(robotManager.target.position.x, xMin, xMax),
                Mathf.Clamp(robotManager.target.position.y, yMin, yMax),
                Mathf.Clamp(robotManager.target.position.z, zMin, zMax)
            );
        }

        /* END APPLY ACTIONS */

        /* REWARD LOGIC */
        
        // Calculate distance to goal
        float currentDistanceToGoal = Vector3.Distance(robotManager.target.position, goalPosition.position);
        float distanceDelta = previousDistanceToGoal - currentDistanceToGoal;
        AddRewardWithType("Moving closer", distanceDelta * 50f);
        previousDistanceToGoal = currentDistanceToGoal;

        // Proximity reward - increases as we get closer
        //float proximityReward = 1.0f / (1.0f + currentDistanceToGoal);
        //AddRewardWithType("Proximity", proximityReward * 0.1f);
        
        // Small time penalty to encourage efficiency
        AddRewardWithType("TimePenalty", -0.001f);

        // Check if goal reached
        if (currentDistanceToGoal < goalThreshold)
        {
            AddRewardWithType("GoalReached", 10.0f);
            EndEpisode();
        }

        /* END REWARD LOGIC */

        currentEpisodeSteps++;

        // End episode if max steps reached
        if (currentEpisodeSteps >= maxStepsPerEpisode)
        {
            AddRewardWithType("Timeout", -1.0f);
            EndEpisode();
        }
    }
    
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        
        // Manual control for testing
        continuousActionsOut[0] = Input.GetAxis("Horizontal"); // X axis
        continuousActionsOut[1] = Input.GetKey(KeyCode.E) ? 1.0f : (Input.GetKey(KeyCode.Q) ? -1.0f : 0.0f); // Y axis
        continuousActionsOut[2] = Input.GetAxis("Vertical"); // Z axis
    }

    private void Update() {
        if(Input.GetKeyDown(KeyCode.R))
        {
            OnEpisodeBegin();
        }
    }

    // Draw a debugging box around the clamped area
    void OnDrawGizmosSelected()
    {
        if (robotManager == null || robotManager.robotJoints == null || robotManager.robotJoints.Length == 0)
            return;

        if (TableTop == null)
            return;

        // Use the TableTop bounds as the base area
        Bounds tableBounds = TableTop.bounds;

        float xMin = tableBounds.min.x;
        float xMax = tableBounds.max.x;
        float zMin = tableBounds.min.z;
        float zMax = tableBounds.max.z;
        float yMin = tableBounds.min.y;
        float yMax = tableBounds.max.y + 3f; // Add 3 units above the table

        Vector3 center = new Vector3((xMin + xMax) / 2f, (yMin + yMax) / 2f, (zMin + zMax) / 2f);
        Vector3 size = new Vector3(xMax - xMin, yMax - yMin, zMax - zMin);

        Gizmos.color = Color.red;
        Gizmos.DrawWireCube(center, size);
    }

    // Make these public or create accessor methods
    public int GetCurrentEpisodeSteps() => currentEpisodeSteps;
    public int GetMaxStepsPerEpisode() => maxStepsPerEpisode;

    // Can't override EndEpisode since it's not marked as virtual in the base class
    // Instead, we'll create our own method and call it when needed
    public new void EndEpisode()
    {
        OnEpisodeEnd?.Invoke(this, currentEpisodeSteps);
        base.EndEpisode();
    }
}