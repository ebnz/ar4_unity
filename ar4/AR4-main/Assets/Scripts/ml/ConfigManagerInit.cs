using UnityEngine;

/// <summary>
/// MonoBehaviour to initialize the ConfigManager at scene start.
/// Attach this to a GameObject in your scene to ensure configuration is loaded.
/// </summary>
public class ConfigManagerInit : MonoBehaviour
{
    [Header("Training Settings")]
    [Tooltip("Maximum steps per episode (in decisions, not actions)")]
    public int maxEpisodeSteps = 200;

    [Tooltip("Speed multiplier for robot actions")]
    public float actionSpeed = 0.5f;

    [Tooltip("Distance threshold to consider goal reached")]
    public float goalThreshold = 0.1f;

    [Tooltip("Action smoothing factor (0-1)")]
    [Range(0f, 1f)]
    public float actionSmoothing = 0.3f;

    [Header("Stats Logging")]
    [Tooltip("Use aggregated checkpoints (true) or individual episodes (false)")]
    public bool useAggregation = true;

    [ConditionalHide("useAggregation", hideWhenTrue: false)]
    [Tooltip("Stats aggregation interval for TrainingStatsLogger (in steps)")]
    public int statsAggregationInterval = 5000;

    [Header("Configuration Mode")]
    [Tooltip("Use Python training config if available (enable during training, disable for inference)")]
    public bool usePythonConfig = false;

    [Header("UI Settings")]
    [Tooltip("Automatically create UI display component for configuration")]
    public bool createDisplayUI = true;

    void Awake()
    {
        if (!ConfigManager.Instance.IsInitialized)
        {
            LoadConfiguration();
        }

        // Create display UI if requested
        if (createDisplayUI)
        {
            CreateDisplayUI();
        }

        Debug.Log("Max Steps Per Episode: " + ConfigManager.Instance.MaxStepsPerEpisode);
    }

    private void LoadConfiguration()
    {
        var config = ConfigManager.Instance;

        // Priority 1: Parse command-line arguments (for Eureka builds)
        if (ParseCommandLineArguments(config))
        {
            // Use inspector values for settings not provided via command-line
            config.ActionSpeed = actionSpeed;
            config.GoalThreshold = goalThreshold;
            config.ActionSmoothing = actionSmoothing;

            Debug.Log("Configuration loaded from command-line arguments (Build/Eureka mode)");
            config.Initialize();
            return;
        }

        // Priority 2: Try to load Python config if enabled (for Editor mode)
        if (usePythonConfig && TryLoadPythonConfig(out EditorTrainingConfig pythonConfig))
        {
            // Python config found - use it
            config.RunId = pythonConfig.run_id;
            config.MaxStepsPerEpisode = pythonConfig.max_episode_steps;
            config.UseAggregation = pythonConfig.use_aggregation;
            config.StatsAggregationInterval = pythonConfig.stats_aggregation_interval > 0 ? pythonConfig.stats_aggregation_interval : statsAggregationInterval;
            config.OutputFilePath = System.IO.Path.Combine(pythonConfig.results_dir, pythonConfig.result_filename);
            config.RewardPath = "";
            config.MLAgentsPort = 5004;

            // Use inspector values for settings not in Python config
            config.ActionSpeed = actionSpeed;
            config.GoalThreshold = goalThreshold;
            config.ActionSmoothing = actionSmoothing;

            // Export reward function to Python's rewards directory
            ExportRewardFunction(pythonConfig.rewards_dir);

            Debug.Log("Configuration loaded from Python training config (Editor mode)");
        }
        else
        {
            // Priority 3: Use inspector values
            config.RewardPath = "";
            config.RunId = 0;
            config.MaxStepsPerEpisode = maxEpisodeSteps;
            config.UseAggregation = useAggregation;
            config.StatsAggregationInterval = statsAggregationInterval;
            config.OutputFilePath = "";
            config.MLAgentsPort = 5004;
            config.ActionSpeed = actionSpeed;
            config.GoalThreshold = goalThreshold;
            config.ActionSmoothing = actionSmoothing;

            Debug.Log("Configuration loaded from Inspector values");
        }

        config.Initialize();
    }

    private bool ParseCommandLineArguments(ConfigManager config)
    {
        string[] args = System.Environment.GetCommandLineArgs();
        bool foundAnyArg = false;

        // Set defaults
        config.RewardPath = "";
        config.RunId = 0;
        config.MaxStepsPerEpisode = maxEpisodeSteps;
        config.UseAggregation = useAggregation;
        config.StatsAggregationInterval = statsAggregationInterval;
        config.OutputFilePath = "";
        config.MLAgentsPort = 5004;

        // Parse arguments
        for (int i = 0; i < args.Length - 1; i++)
        {
            string arg = args[i].ToLower();
            string value = args[i + 1];

            switch (arg)
            {
                case "--reward-path":
                    config.RewardPath = value;
                    foundAnyArg = true;
                    Debug.Log($"[CmdLine] reward-path = {value}");
                    break;

                case "--run-id":
                    if (int.TryParse(value, out int runId))
                    {
                        config.RunId = runId;
                        foundAnyArg = true;
                        Debug.Log($"[CmdLine] run-id = {runId}");
                    }
                    break;

                case "--max-training-steps":
                    // Ignored - Python handles this
                    break;

                case "--max-episode-steps":
                    if (int.TryParse(value, out int maxSteps))
                    {
                        config.MaxStepsPerEpisode = maxSteps;
                        foundAnyArg = true;
                        Debug.Log($"[CmdLine] max-episode-steps = {maxSteps}");
                    }
                    break;

                case "--output-file":
                    config.OutputFilePath = value;
                    foundAnyArg = true;
                    Debug.Log($"[CmdLine] output-file = {value}");
                    break;

                case "--mlagents-port":
                    if (int.TryParse(value, out int port))
                    {
                        config.MLAgentsPort = port;
                        foundAnyArg = true;
                        Debug.Log($"[CmdLine] mlagents-port = {port}");
                    }
                    break;

                case "--use-aggregation":
                    if (bool.TryParse(value, out bool useAgg))
                    {
                        config.UseAggregation = useAgg;
                        foundAnyArg = true;
                        Debug.Log($"[CmdLine] use-aggregation = {useAgg}");
                    }
                    break;

                case "--stats-aggregation-interval":
                    if (int.TryParse(value, out int statsInterval))
                    {
                        config.StatsAggregationInterval = statsInterval;
                        foundAnyArg = true;
                        Debug.Log($"[CmdLine] stats-aggregation-interval = {statsInterval}");
                    }
                    break;
            }
        }

        return foundAnyArg;
    }

    private bool TryLoadPythonConfig(out EditorTrainingConfig config)
    {
        config = null;

        string configPath = System.IO.Path.Combine(UnityEngine.Application.streamingAssetsPath, "python_training_config.json");

        if (!System.IO.File.Exists(configPath))
        {
            return false;
        }

        try
        {
            string json = System.IO.File.ReadAllText(configPath);
            config = JsonUtility.FromJson<EditorTrainingConfig>(json);
            return true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to parse Python config: {e.Message}");
            return false;
        }
    }

    private void ExportRewardFunction(string rewardsDir)
    {
        try
        {
            string rewardCode = null;

            // Try to get reward code from RewardFunctionManagerInit
            RewardFunctionManagerInit rewardInit = FindFirstObjectByType<RewardFunctionManagerInit>();
            if (rewardInit != null)
            {
                // Check if using file-based reward
                if (rewardInit.rewardFunctionFile != null)
                {
                    rewardCode = rewardInit.rewardFunctionFile.text;
                }
                // Fallback to inspector reward code
                else if (rewardInit.useInspectorReward)
                {
                    rewardCode = rewardInit.defaultRewardFunctionCode;
                }
            }

            // Fallback: try RewardFunctionManager
            if (string.IsNullOrEmpty(rewardCode))
            {
                rewardCode = RewardFunctionManager.GetCurrentRewardFunctionCode();
            }

            if (string.IsNullOrEmpty(rewardCode))
            {
                // Create marker file
                string markerPath = System.IO.Path.Combine(rewardsDir, "using_builtin_reward.txt");
                System.IO.File.WriteAllText(markerPath, "Using reward function from Unity agent script.");
                return;
            }

            string filename = $"reward_editor_{System.DateTime.Now:yyyyMMdd_HHmmss}.cs";
            string fullPath = System.IO.Path.Combine(rewardsDir, filename);
            System.IO.File.WriteAllText(fullPath, rewardCode);

            Debug.Log($"Exported reward function to: {fullPath}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to export reward function: {e.Message}");
        }
    }
    
    /// <summary>
    /// Create the configuration display UI component
    /// </summary>
    private void CreateDisplayUI()
    {
        try
        {
            // Check if display already exists
            ConfigManagerDisplay existingDisplay = FindFirstObjectByType<ConfigManagerDisplay>();
            if (existingDisplay != null)
            {
                Debug.Log("ConfigManagerDisplay already exists, skipping creation.");
                return;
            }
            
            // Create a new GameObject for the display
            GameObject displayGO = new GameObject("ConfigManagerDisplay");
            displayGO.AddComponent<ConfigManagerDisplay>();
            
            Debug.Log("Created ConfigManagerDisplay UI component (Press F2 to toggle)");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to create ConfigManagerDisplay: {e.Message}");
        }
    }
}

/// <summary>
/// Configuration data structure for Unity Editor mode (loaded from Python)
/// </summary>
[System.Serializable]
public class EditorTrainingConfig
{
    public int run_id;
    public string rewards_dir;
    public string results_dir;
    public string result_filename;
    public int max_episode_steps;
    public bool use_aggregation;
    public int stats_aggregation_interval;
}
