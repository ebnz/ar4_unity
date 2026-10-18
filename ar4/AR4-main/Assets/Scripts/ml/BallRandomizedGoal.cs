using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class BallRandomizedGoalAgent : Agent
{
    public Rigidbody ballRigidbody;
    public Transform goalTransform;
    public float forceMultiplier = 10f;

    private Vector3 initialPosition;
    private Vector3 goalInitialPosition;
    private float previousDistanceToGoal;
    private float closerRewardScale = 50f;
    private float goalRewardScale = 10f;

    public override void Initialize()
    {
        initialPosition = ballRigidbody.position;
        goalInitialPosition = goalTransform.position;
    }

    public override void OnEpisodeBegin()
    {
        // Reset ball and goal positions
        ballRigidbody.linearVelocity = Vector3.zero;
        ballRigidbody.angularVelocity = Vector3.zero;
        ballRigidbody.position = initialPosition;

        // Randomize goal position within a certain range, but keep at least 4 units from the player
        float range = 7.0f; // Adjust as needed
        float minDistance = 4.0f;
        Vector3 randomOffset;
        Vector3 candidatePosition;
        int maxTries = 100;
        int tries = 0;
        do
        {
            randomOffset = new Vector3(
                Random.Range(-range, range),
                0f,
                Random.Range(-range, range)
            );
            candidatePosition = initialPosition + randomOffset;
            candidatePosition = new Vector3(candidatePosition.x, 0.5f, candidatePosition.z);
            tries++;
        }
        while (Vector3.Distance(initialPosition, candidatePosition) < minDistance && tries < maxTries);

        goalTransform.position = candidatePosition;

        // Initialize previous distance to goal
        previousDistanceToGoal = Vector3.Distance(ballRigidbody.position, goalTransform.position);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Ball position relative to goal
        sensor.AddObservation(ballRigidbody.position - goalTransform.position);
        // Ball velocity
        sensor.AddObservation(ballRigidbody.linearVelocity);
        // Goal position relative to initial position
        sensor.AddObservation(goalTransform.position - goalInitialPosition);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Actions: [0]=X force, [1]=Z force
        float forceX = actions.ContinuousActions[0];
        float forceZ = actions.ContinuousActions[1];

        ballRigidbody.AddForce(new Vector3(forceX, 0, forceZ) * forceMultiplier);

        // Reward for getting closer to the goal
        float distanceToGoal = Vector3.Distance(ballRigidbody.position, goalTransform.position);
        float distanceDelta = previousDistanceToGoal - distanceToGoal;
        AddReward(distanceDelta * closerRewardScale);
        previousDistanceToGoal = distanceToGoal;

        if (distanceToGoal < 1.0f)
        {
            SetReward(goalRewardScale);
            EndEpisode();
        }

        // Small negative reward for each step to encourage faster solutions
        AddReward(-0.001f);

        // End episode if ball falls off platform
        if (ballRigidbody.position.y < -1)
        {
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");
        continuousActionsOut[0] = horizontal;
        continuousActionsOut[1] = vertical;
    }
}