
// Default T0 reward function - move to target task
// Extract observations
float currentDistanceToGoal = observations["DistanceToGoal"];
float distanceDelta = context.previousDistance - currentDistanceToGoal;

// Moving closer reward
rewards["MovingCloser"] = distanceDelta * 50f;

// Time penalty
rewards["TimePenalty"] = -0.001f;

// Goal reached bonus
if (currentDistanceToGoal < context.goalThreshold)
{
    rewards["GoalReached"] = 10.0f;
}