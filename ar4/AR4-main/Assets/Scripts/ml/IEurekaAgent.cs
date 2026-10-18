using System.Collections.Generic;

/// <summary>
/// Interface for agents that can be tracked by the TrainingStatsLogger for Eureka training
/// </summary>
public interface IEurekaAgent
{
    /// <summary>
    /// Get current observations for external tracking
    /// </summary>
    Dictionary<string, float> GetCurrentObservations();
    
    /// <summary>
    /// Get current episode step count
    /// </summary>
    int GetCurrentEpisodeSteps();
    
    /// <summary>
    /// Get maximum steps per episode
    /// </summary>
    int GetMaxStepsPerEpisode();
    
    /// <summary>
    /// Get rewards by type for this episode
    /// </summary>
    Dictionary<string, float> GetRewardsByType();
    
    /// <summary>
    /// Get the agent's name for identification
    /// </summary>
    string name { get; }

    /// <summary>
    /// Called when a decision is requested (not on action repetitions)
    /// </summary>
    void OnDecisionRequested();
}