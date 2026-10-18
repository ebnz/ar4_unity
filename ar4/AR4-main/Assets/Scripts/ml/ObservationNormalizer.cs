using UnityEngine;
using Unity.MLAgents.Policies;
using System.IO;
using System;

/// <summary>
/// VecNormalize statistics loaded from JSON (exported from Python training)
/// </summary>
[Serializable]
public class VecNormalizeStats
{
    public float[] mean;
    public float[] var;
    public float epsilon;
    public float clip_obs;
}

/// <summary>
/// Applies VecNormalize observation normalization for ONNX inference
/// Only active when BehaviorParameters is set to InferenceOnly mode
/// </summary>
public class ObservationNormalizer : MonoBehaviour
{
    [Header("Normalization Stats File")]
    [Tooltip("Drag the vecnorm_stats.json file here (preferred method)")]
    public TextAsset statsFile;

    [Tooltip("Alternative: Path to vecnorm_stats.json file (relative to StreamingAssets or absolute). Only used if statsFile is not assigned.")]
    public string statsFilePath = "";

    private VecNormalizeStats stats;
    private bool isActive = false;
    private BehaviorParameters behaviorParams;

    void Start()
    {
        behaviorParams = GetComponent<BehaviorParameters>();

        if (behaviorParams == null)
        {
            Debug.LogError("ObservationNormalizer requires BehaviorParameters component!");
            enabled = false;
            return;
        }

        // Only load and activate normalization in inference mode
        if (behaviorParams.BehaviorType == BehaviorType.InferenceOnly)
        {
            LoadNormalizationStats();
        }
        else
        {
            Debug.Log("ObservationNormalizer: Training mode detected, normalization disabled (handled by Python)");
        }
    }

    private void LoadNormalizationStats()
    {
        try
        {
            string json = null;

            // Priority 1: Use assigned TextAsset
            if (statsFile != null)
            {
                json = statsFile.text;
                Debug.Log($"Loading VecNormalize stats from assigned TextAsset: {statsFile.name}");
            }
            else if (!string.IsNullOrEmpty(statsFilePath))
            {
                // Priority 2: Try StreamingAssets first, then absolute path
                string fullPath = Path.Combine(Application.streamingAssetsPath, statsFilePath);
                if (!File.Exists(fullPath))
                {
                    fullPath = statsFilePath;
                }

                if (!File.Exists(fullPath))
                {
                    Debug.LogWarning($"VecNormalize stats file not found: {fullPath}. Normalization will be disabled.");
                    return;
                }

                json = File.ReadAllText(fullPath);
            }
            else
            {
                Debug.LogWarning("No VecNormalize stats file assigned. Normalization will be disabled.");
                return;
            }

            stats = JsonUtility.FromJson<VecNormalizeStats>(json);

            if (stats.mean == null || stats.var == null)
            {
                Debug.LogError("Failed to parse VecNormalize stats JSON");
                return;
            }

            isActive = true;
            Debug.Log($"✅ VecNormalize stats loaded: {stats.mean.Length} dimensions, epsilon={stats.epsilon}, clip={stats.clip_obs}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to load VecNormalize stats: {e.Message}");
        }
    }

    /// <summary>
    /// Normalize observations using loaded VecNormalize statistics
    /// Formula: clamp((obs - mean) / sqrt(var + epsilon), -clip_obs, clip_obs)
    /// </summary>
    public float[] Normalize(float[] observations)
    {
        if (!isActive || stats == null)
        {
            // Pass through unchanged if normalization not active
            return observations;
        }

        if (observations.Length != stats.mean.Length)
        {
            Debug.LogError($"Observation dimension mismatch! Expected {stats.mean.Length}, got {observations.Length}");
            return observations;
        }

        float[] normalized = new float[observations.Length];

        for (int i = 0; i < observations.Length; i++)
        {
            float std = Mathf.Sqrt(stats.var[i] + stats.epsilon);
            float normalizedValue = (observations[i] - stats.mean[i]) / std;
            normalized[i] = Mathf.Clamp(normalizedValue, -stats.clip_obs, stats.clip_obs);
        }

        return normalized;
    }

    /// <summary>
    /// Check if normalization is currently active
    /// </summary>
    public bool IsActive => isActive;
}
