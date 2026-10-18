using UnityEngine;
using Unity.MLAgents;

/// <summary>
/// Custom DecisionRequester that tracks actual decision requests
/// Replaces the standard DecisionRequester component
/// </summary>
public class DecisionRequesterWithLogging : DecisionRequester
{
    private TrainingStatsLogger statsLogger;
    private IEurekaAgent agent;

    void Start()
    {
        statsLogger = FindFirstObjectByType<TrainingStatsLogger>();
        agent = GetComponent<IEurekaAgent>();

        if (agent == null)
        {
            Debug.LogWarning("DecisionRequesterWithLogging: No IEurekaAgent found on this GameObject");
        }
    }

    protected override bool ShouldRequestDecision(DecisionRequestContext context)
    {
        // Check if base class says we should request a decision
        bool shouldRequest = base.ShouldRequestDecision(context);

        // If we're requesting a decision, notify the agent and log it
        if (shouldRequest && agent != null)
        {
            agent.OnDecisionRequested();

            if (statsLogger != null)
            {
                statsLogger.OnDynamicAgentDecision(agent);
            }
        }

        return shouldRequest;
    }
}
