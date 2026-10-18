using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System;
using System.Text;

[Serializable]
public class RewardComponentStat
{
    public float mean;
    public float std;
    public float min;
    public float max;
}

[Serializable]
public class ObservationStat
{
    public float mean;
    public float std;
    public float min;
    public float max;
}

[Serializable]
public class RewardComponentEntry
{
    public string name;
    public float value; // For individual episodes only
}

[Serializable]
public class RewardComponentAggregated
{
    public string name;
    public RewardComponentStat stats; // For aggregated checkpoints only
}

[Serializable]
public class ObservationStatEntry
{
    public string name;
    public ObservationStat stats;
}

[Serializable]
public class EpisodeData
{
    public int episode_number;
    public int timestep;
    public float reward;
    public int length;
    public bool success;
    public List<RewardComponentEntry> reward_components; // Will only have 'value' populated
    public List<ObservationStatEntry> observations;
}

[Serializable]
public class EpisodeStats
{
    public float mean;
    public float std;
    public float min;
    public float max;
}

[Serializable]
public class AggregatedCheckpoint
{
    public int checkpoint_number;
    public int step_number;
    public int num_episodes;
    public float success_rate;
    public float curriculum_progress;  // 0 to 1, -1 if curriculum disabled
    public EpisodeStats episode_reward_stats;
    public EpisodeStats episode_length_stats;
    public List<RewardComponentAggregated> reward_components;
    public List<ObservationStatEntry> observations;
}

[Serializable]
public class TrainingResults
{
    public int run_id;
    public string reward_path;
    public List<EpisodeData> episodes = new List<EpisodeData>();
}

[Serializable]
public class AggregatedResults
{
    public int run_id;
    public string reward_path;
    public List<AggregatedCheckpoint> checkpoints = new List<AggregatedCheckpoint>();
}

public class TrainingStatsLogger : MonoBehaviour
{
    [Header("Logging Settings")]
    public bool writeResultsToFile = true;
    public int batchSize = 20; // For individual episodes: save every N episodes

    private TrainingResults results = new TrainingResults();
    private AggregatedResults aggregatedResults = new AggregatedResults();
    private int totalStepsAllAgents = 0;
    private int episodesSinceLastSave = 0;
    private int checkpointNumber = 0;
    private bool useAggregation;
    private int aggregationStepInterval;
    private List<IEurekaAgent> trackedDynamicAgents = new List<IEurekaAgent>();

    // Temporary storage for current episode observations
    private Dictionary<IEurekaAgent, Dictionary<string, List<float>>> currentEpisodeObservations =
        new Dictionary<IEurekaAgent, Dictionary<string, List<float>>>();

    // Aggregation mode: buffer for episodes since last checkpoint
    private List<float> checkpointEpisodeRewards = new List<float>();
    private List<int> checkpointEpisodeLengths = new List<int>();
    private List<bool> checkpointEpisodeSuccesses = new List<bool>();
    private Dictionary<string, List<float>> checkpointRewardComponents = new Dictionary<string, List<float>>();
    private Dictionary<string, List<float>> checkpointObservations = new Dictionary<string, List<float>>();

    void Start()
    {
        // Find all agents
        var allAgents = FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None);
        trackedDynamicAgents.AddRange(allAgents.OfType<IEurekaAgent>());

        if (trackedDynamicAgents.Count == 0)
        {
            Debug.LogError("No IEurekaAgent found for logging!");
            enabled = false;
            return;
        }

        // Initialize observation tracking
        foreach (var agent in trackedDynamicAgents)
        {
            currentEpisodeObservations[agent] = new Dictionary<string, List<float>>();
        }

        // Set run ID and reward path
        int runId = ConfigManager.Instance.IsInitialized ? ConfigManager.Instance.RunId : 0;
        string rewardPath = ConfigManager.Instance.IsInitialized ? ConfigManager.Instance.RewardPath : "";

        results.run_id = runId;
        results.reward_path = rewardPath;
        aggregatedResults.run_id = runId;
        aggregatedResults.reward_path = rewardPath;

        // Read logging mode and interval from ConfigManager
        useAggregation = ConfigManager.Instance.UseAggregation;
        aggregationStepInterval = ConfigManager.Instance.StatsAggregationInterval;

        string mode = useAggregation ? $"aggregated (checkpoint every {aggregationStepInterval} steps)" : $"individual episodes (save every {batchSize} episodes)";
        Debug.Log($"Training Stats Logger initialized. Tracking {trackedDynamicAgents.Count} agents in {mode} mode");
    }

    void FixedUpdate()
    {
        // Collect observations during episode
        foreach (var agent in trackedDynamicAgents)
        {
            if (agent == null) continue;

            // Collect observations
            var obs = agent.GetCurrentObservations();
            foreach (var kvp in obs)
            {
                if (!currentEpisodeObservations[agent].ContainsKey(kvp.Key))
                {
                    currentEpisodeObservations[agent][kvp.Key] = new List<float>();
                }
                currentEpisodeObservations[agent][kvp.Key].Add(kvp.Value);
            }
        }
    }
    
    public void OnDynamicAgentDecision(IEurekaAgent agent)
    {
        totalStepsAllAgents++;
        // In aggregation mode, check if it's time for a checkpoint
        if (useAggregation && trackedDynamicAgents.Count > 0)
        {
            // Check if we've crossed the checkpoint threshold (only check on first agent to avoid multiple triggers)
            if (agent == trackedDynamicAgents[0] && totalStepsAllAgents > aggregationStepInterval * (checkpointNumber + 1))
            {
                SaveAggregatedCheckpoint();
            }
        }
    }

    public void OnDynamicAgentEpisodeEnd(IEurekaAgent agent, float episodeReward, int episodeLength, Dictionary<string, float> rewardComponents, bool isSuccess)
    {
        if (useAggregation)
        {
            // Aggregation mode: buffer episode data for checkpoint
            checkpointEpisodeRewards.Add(episodeReward);
            checkpointEpisodeLengths.Add(episodeLength);
            checkpointEpisodeSuccesses.Add(isSuccess);

            // Aggregate final cumulative reward component values
            foreach (var kvp in rewardComponents)
            {
                if (!checkpointRewardComponents.ContainsKey(kvp.Key))
                {
                    checkpointRewardComponents[kvp.Key] = new List<float>();
                }
                checkpointRewardComponents[kvp.Key].Add(kvp.Value);
            }

            // Aggregate observations (get mean from current episode)
            if (currentEpisodeObservations.ContainsKey(agent))
            {
                foreach (var kvp in currentEpisodeObservations[agent])
                {
                    if (kvp.Value.Count > 0)
                    {
                        if (!checkpointObservations.ContainsKey(kvp.Key))
                        {
                            checkpointObservations[kvp.Key] = new List<float>();
                        }
                        checkpointObservations[kvp.Key].Add(kvp.Value.Average());
                    }
                }
                currentEpisodeObservations[agent].Clear();
            }
        }
        else
        {
            // Individual episode mode: store final cumulative values and observation stats
            var episodeObservationStats = new List<ObservationStatEntry>();
            if (currentEpisodeObservations.ContainsKey(agent))
            {
                foreach (var kvp in currentEpisodeObservations[agent])
                {
                    var values = kvp.Value;
                    if (values.Count > 0)
                    {
                        episodeObservationStats.Add(new ObservationStatEntry
                        {
                            name = kvp.Key,
                            stats = new ObservationStat
                            {
                                mean = values.Average(),
                                std = CalculateStandardDeviation(values),
                                min = values.Min(),
                                max = values.Max()
                            }
                        });
                    }
                }
                currentEpisodeObservations[agent].Clear();
            }

            // Store final cumulative reward component values
            var episodeRewardComponents = new List<RewardComponentEntry>();
            foreach (var kvp in rewardComponents)
            {
                episodeRewardComponents.Add(new RewardComponentEntry
                {
                    name = kvp.Key,
                    value = kvp.Value // Final cumulative value
                });
            }

            // Create episode entry
            var episode = new EpisodeData
            {
                episode_number = results.episodes.Count,
                timestep = totalStepsAllAgents / trackedDynamicAgents.Count,
                reward = episodeReward,
                length = episodeLength,
                success = isSuccess,
                reward_components = episodeRewardComponents,
                observations = episodeObservationStats
            };

            results.episodes.Add(episode);
            episodesSinceLastSave++;

            // Save in batches
            if (episodesSinceLastSave >= batchSize)
            {
                SaveResults();
                episodesSinceLastSave = 0;
            }
        }
    }

    private float CalculateStandardDeviation(List<float> values)
    {
        if (values.Count <= 1) return 0f;

        float mean = values.Average();
        float sumOfSquaredDifferences = values.Sum(x => (x - mean) * (x - mean));
        return Mathf.Sqrt(sumOfSquaredDifferences / (values.Count - 1));
    }

    private void SaveAggregatedCheckpoint()
    {
        if (!writeResultsToFile || checkpointEpisodeRewards.Count == 0) return;

        // Don't overwrite error results - runtime errors should be preserved for Python to detect
        if (RewardFunctionManager.HasRuntimeError)
        {
            Debug.LogWarning($"Skipping checkpoint save due to runtime error in reward function: {RewardFunctionManager.LastRuntimeError}");
            // Clear buffers anyway to avoid memory buildup
            checkpointEpisodeRewards.Clear();
            checkpointEpisodeLengths.Clear();
            checkpointEpisodeSuccesses.Clear();
            checkpointRewardComponents.Clear();
            checkpointObservations.Clear();
            return;
        }

        try
        {
            checkpointNumber++;
            int environmentSteps = totalStepsAllAgents;

            // Calculate success rate
            float successRate = checkpointEpisodeSuccesses.Count > 0
                ? (float)checkpointEpisodeSuccesses.Count(s => s) / checkpointEpisodeSuccesses.Count
                : 0f;

            // Calculate episode reward stats
            var rewardStats = new EpisodeStats
            {
                mean = checkpointEpisodeRewards.Average(),
                std = CalculateStandardDeviation(checkpointEpisodeRewards),
                min = checkpointEpisodeRewards.Min(),
                max = checkpointEpisodeRewards.Max()
            };

            // Calculate episode length stats
            var lengthFloats = checkpointEpisodeLengths.Select(x => (float)x).ToList();
            var lengthStats = new EpisodeStats
            {
                mean = lengthFloats.Average(),
                std = CalculateStandardDeviation(lengthFloats),
                min = lengthFloats.Min(),
                max = lengthFloats.Max()
            };

            // Calculate reward component stats (across episodes)
            // Fill in 0 for episodes that don't have a particular component
            int numEpisodes = checkpointEpisodeRewards.Count;
            var rewardComponentEntries = new List<RewardComponentAggregated>();
            foreach (var kvp in checkpointRewardComponents)
            {
                // Pad with zeros for episodes that didn't have this component
                var allValues = new List<float>(kvp.Value);
                while (allValues.Count < numEpisodes)
                {
                    allValues.Add(0f);
                }

                rewardComponentEntries.Add(new RewardComponentAggregated
                {
                    name = kvp.Key,
                    stats = new RewardComponentStat
                    {
                        mean = allValues.Average(),
                        std = CalculateStandardDeviation(allValues),
                        min = allValues.Min(),
                        max = allValues.Max()
                    }
                });
            }

            // Calculate observation stats (across episodes)
            var observationEntries = new List<ObservationStatEntry>();
            foreach (var kvp in checkpointObservations)
            {
                observationEntries.Add(new ObservationStatEntry
                {
                    name = kvp.Key,
                    stats = new ObservationStat
                    {
                        mean = kvp.Value.Average(),
                        std = CalculateStandardDeviation(kvp.Value),
                        min = kvp.Value.Min(),
                        max = kvp.Value.Max()
                    }
                });
            }

            // Get curriculum progress from first curriculum-enabled agent (if any)
            float curriculumProgress = GetCurriculumProgress();

            // Create checkpoint
            var checkpoint = new AggregatedCheckpoint
            {
                checkpoint_number = checkpointNumber,
                step_number = environmentSteps,
                num_episodes = checkpointEpisodeRewards.Count,
                success_rate = successRate,
                curriculum_progress = curriculumProgress,
                episode_reward_stats = rewardStats,
                episode_length_stats = lengthStats,
                reward_components = rewardComponentEntries,
                observations = observationEntries
            };

            aggregatedResults.checkpoints.Add(checkpoint);

            // Save to file
            string fullPath = ConfigManager.Instance.GetResultsFilePath();
            string json = JsonUtility.ToJson(aggregatedResults, true);
            File.WriteAllText(fullPath, json);

            string progressStr = curriculumProgress >= 0 ? $", curriculum={curriculumProgress:P0}" : "";
            Debug.Log($"Saved checkpoint {checkpointNumber} at step {environmentSteps} ({checkpointEpisodeRewards.Count} episodes, success={successRate:P0}{progressStr}) to {fullPath}");

            // Notify agents for curriculum learning
            NotifyAgentsOfCheckpoint(successRate);

            // Clear checkpoint buffers
            checkpointEpisodeRewards.Clear();
            checkpointEpisodeLengths.Clear();
            checkpointEpisodeSuccesses.Clear();
            checkpointRewardComponents.Clear();
            checkpointObservations.Clear();
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to save aggregated checkpoint: {e.Message}");
        }
    }

    private void SaveResults()
    {
        if (!writeResultsToFile) return;

        // Don't overwrite error results - runtime errors should be preserved for Python to detect
        if (RewardFunctionManager.HasRuntimeError)
        {
            Debug.LogWarning($"Skipping results save due to runtime error in reward function: {RewardFunctionManager.LastRuntimeError}");
            return;
        }

        try
        {
            string fullPath = ConfigManager.Instance.GetResultsFilePath();
            string json = JsonUtility.ToJson(results, true);
            File.WriteAllText(fullPath, json);

            Debug.Log($"Saved {results.episodes.Count} episodes to {fullPath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to save results: {e.Message}");
        }
    }

    /// <summary>
    /// Gets curriculum progress from the first curriculum-enabled agent.
    /// Returns -1 if no curriculum agent is found or curriculum is disabled.
    /// </summary>
    private float GetCurriculumProgress()
    {
        foreach (var agent in trackedDynamicAgents)
        {
            if (agent is RobotAgentTPickAndPlaceDirectControlRandomizedSpawn curriculumAgent
                && curriculumAgent.useCurriculum)
            {
                return curriculumAgent.GetCurriculumProgress();
            }
        }
        return -1f;
    }

    /// <summary>
    /// Notifies all tracked agents of a checkpoint save for curriculum learning.
    /// </summary>
    private void NotifyAgentsOfCheckpoint(float successRate)
    {
        foreach (var agent in trackedDynamicAgents)
        {
            // Check if agent supports curriculum learning via OnCheckpointSaved method
            if (agent is RobotAgentTPickAndPlaceDirectControlRandomizedSpawn curriculumAgent)
            {
                curriculumAgent.OnCheckpointSaved(successRate);
            }
        }
    }

    void OnApplicationQuit()
    {
        // Don't overwrite error results on quit
        if (RewardFunctionManager.HasRuntimeError)
        {
            Debug.LogWarning("Skipping final save on quit - preserving runtime error result");
            return;
        }

        if (useAggregation)
        {
            // Save any remaining checkpoint data on quit
            if (checkpointEpisodeRewards.Count > 0)
            {
                SaveAggregatedCheckpoint();
            }
        }
        else
        {
            // Save any remaining episodes on quit
            if (episodesSinceLastSave > 0)
            {
                SaveResults();
            }
        }
    }
}