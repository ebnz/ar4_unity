using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

/// <summary>
/// Manages configuration for Eureka training, including command line argument parsing
/// Provides centralized configuration handling for ML agents
/// </summary>
public class ConfigManager
{
    private static ConfigManager instance;
    public static ConfigManager Instance
    {
        get
        {
            if (instance == null)
            {
                instance = new ConfigManager();
            }
            return instance;
        }
    }

    // Configuration properties - all public and settable
    public string RewardPath { get; set; } = "";
    public int RunId { get; set; } = 0;
    public int MaxStepsPerEpisode { get; set; } = 200;
    public string OutputFilePath { get; set; } = "";
    public bool UseFileBasedReward { get; set; } = false;
    public int MLAgentsPort { get; set; } = 5004;
    public float ActionSpeed { get; set; } = 0.5f;
    public float GoalThreshold { get; set; } = 0.1f;
    public float ActionSmoothing { get; set; } = 0.3f;
    public bool UseAggregation { get; set; } = true;
    public int StatsAggregationInterval { get; set; } = 5000;

    // Tracking
    public bool IsInitialized { get; private set; } = false;

    private ConfigManager()
    {
        // Private constructor for singleton
    }

    /// <summary>
    /// Mark configuration as initialized and log current values
    /// Call this after setting all config properties
    /// </summary>
    public void Initialize()
    {
        IsInitialized = true;
        UseFileBasedReward = !string.IsNullOrEmpty(RewardPath);
        LogConfiguration();
    }

    /// <summary>
    /// Log current configuration to console
    /// </summary>
    private void LogConfiguration()
    {
        Debug.Log($"ConfigManager initialized:");
        Debug.Log($"  Reward Path: {RewardPath}");
        Debug.Log($"  Run ID: {RunId}");
        Debug.Log($"  Max Steps: {MaxStepsPerEpisode}");
        Debug.Log($"  Output File: {OutputFilePath}");
        Debug.Log($"  ML-Agents Port: {MLAgentsPort}");
        Debug.Log($"  Use File-Based Reward: {UseFileBasedReward}");
        Debug.Log($"  Action Speed: {ActionSpeed}");
        Debug.Log($"  Goal Threshold: {GoalThreshold}");
        Debug.Log($"  Action Smoothing: {ActionSmoothing}");
    }

    /// <summary>
    /// Get configuration as a dictionary for easy access
    /// </summary>
    public Dictionary<string, object> GetConfigDictionary()
    {
        return new Dictionary<string, object>
        {
            ["RewardPath"] = RewardPath,
            ["RunId"] = RunId,
            ["MaxStepsPerEpisode"] = MaxStepsPerEpisode,
            ["OutputFilePath"] = OutputFilePath,
            ["MLAgentsPort"] = MLAgentsPort,
            ["UseFileBasedReward"] = UseFileBasedReward,
            ["ActionSpeed"] = ActionSpeed,
            ["GoalThreshold"] = GoalThreshold,
            ["ActionSmoothing"] = ActionSmoothing,
        };
    }

    /// <summary>
    /// Update a configuration value dynamically
    /// </summary>
    public void SetConfigValue(string key, object value)
    {
        switch (key.ToLower())
        {
            case "rewardpath":
                RewardPath = value.ToString();
                UseFileBasedReward = !string.IsNullOrEmpty(RewardPath);
                break;
            case "runid":
                if (int.TryParse(value.ToString(), out int runId))
                    RunId = runId;
                break;
            case "maxepisode-steps": // New, clearer key
            case "maxstepsper Episode":
            case "maxsteps":
                if (int.TryParse(value.ToString(), out int maxSteps))
                    MaxStepsPerEpisode = maxSteps;
                break;
            case "outputfilepath":
            case "outputfile":
                OutputFilePath = value.ToString();
                break;
            case "mlagentsport":
            case "port":
                if (int.TryParse(value.ToString(), out int port))
                    MLAgentsPort = port;
                break;
            case "actionspeed":
                if (float.TryParse(value.ToString(), out float actionSpeed))
                    ActionSpeed = actionSpeed;
                break;
            case "goalthreshold":
                if (float.TryParse(value.ToString(), out float goalThreshold))
                    GoalThreshold = goalThreshold;
                break;
            case "actionsmoothing":
                if (float.TryParse(value.ToString(), out float actionSmoothing))
                    ActionSmoothing = actionSmoothing;
                break;
            default:
                Debug.LogWarning($"Unknown config key: {key}");
                break;
        }
    }

    /// <summary>
    /// Get the full path for the results JSON file
    /// </summary>
    public string GetResultsFilePath()
    {
        // If OutputFilePath is already an absolute path (from Eureka), use it directly
        if (!string.IsNullOrEmpty(OutputFilePath) && System.IO.Path.IsPathRooted(OutputFilePath))
        {
            // Ensure the directory exists
            string directory = System.IO.Path.GetDirectoryName(OutputFilePath);
            if (!string.IsNullOrEmpty(directory))
            {
                System.IO.Directory.CreateDirectory(directory);
            }
            return OutputFilePath;
        }

        // Otherwise, construct path in persistent data (for Editor mode)
        string resultsDir = System.IO.Path.Combine(Application.persistentDataPath, "EurekaRewards", "results");
        System.IO.Directory.CreateDirectory(resultsDir);

        string fileName = !string.IsNullOrEmpty(OutputFilePath)
            ? OutputFilePath
            : $"results_{RunId}.json";

        return System.IO.Path.Combine(resultsDir, fileName);
    }

    /// <summary>
    /// Validate that required configuration is present
    /// </summary>
    public bool ValidateConfiguration()
    {
        if (!IsInitialized)
        {
            Debug.LogError("ConfigManager not initialized!");
            return false;
        }

        if (UseFileBasedReward && string.IsNullOrEmpty(RewardPath))
        {
            Debug.LogError("File-based reward enabled but no reward path provided!");
            return false;
        }

        if (MaxStepsPerEpisode <= 0)
        {
            Debug.LogError("Max steps per episode must be positive!");
            return false;
        }

        return true;
    }
}