using UnityEngine;
using System.Threading.Tasks;

/// <summary>
/// MonoBehaviour to initialize RewardFunctionManager with inspector-editable reward functions.
/// Attach this to a GameObject in your scene for easy reward function prototyping and testing.
/// Uses Unity's Awake/Start execution order to ensure initialization before agents start.
/// </summary>
public class RewardFunctionManagerInit : MonoBehaviour
{
    [Header("Default Reward Function")]
    [Tooltip("Drag and drop a text file containing the reward function code")]
    public TextAsset rewardFunctionFile;

    [TextArea(15, 30)]
    [Tooltip("Alternative: Enter reward function code directly (used if no file is assigned above)")]
    public string defaultRewardFunctionCode = @"
// Default T0 reward function - move to target task
// Extract observations
float currentDistanceToGoal = observations[""DistanceToGoal""];
float previousDistance = (float)context[""previousDistance""];
float distanceDelta = previousDistance - currentDistanceToGoal;

// Moving closer reward
rewards[""MovingCloser""] = distanceDelta * 50f;

// Time penalty
rewards[""TimePenalty""] = -0.001f;

// Goal reached bonus
float goalThreshold = (float)context[""goalThreshold""];
if (currentDistanceToGoal < goalThreshold)
{
    rewards[""GoalReached""] = 10.0f;
}

// Example: Access Vector3 context variables
// Vector3 targetPos = (Vector3)context[""targetPosition""];
// Vector3 goalPos = (Vector3)context[""goalPosition""];
// float directDistance = Vector3.Distance(targetPos, goalPos);

// Note: 'rewards' dictionary and total calculation are auto-generated
// Note: RewardResult return is auto-generated
";
    
    [Header("Settings")]
    [Tooltip("Use the reward function defined in the inspector above")]
    public bool useInspectorReward = true;
    
    [Tooltip("Enable runtime recompilation for testing (inspector only)")]
    public bool enableRuntimeRecompilation = true;
    
    [Tooltip("Automatically create UI display component for reward function")]
    public bool createDisplayUI = true;

    async void Awake()
    {
        // This runs BEFORE any Start() methods in the scene
        // Async compilation will complete before agents start
        Debug.Log("RewardFunctionManagerInit.Awake() called - starting async initialization...");
        await InitializeRewardManagerAsync();
        
        // Create display UI if requested
        if (createDisplayUI)
        {
            CreateDisplayUI();
        }
    }
    
    private async Task InitializeRewardManagerAsync()
    {
        try
        {
            // ConfigManager should be initialized by ConfigManagerInit
            // (no need to initialize here)

            // Determine which reward code to use
            // Priority: 1) File asset, 2) Inspector code, 3) null (use config default)
            string rewardCode = null;
            if (rewardFunctionFile != null)
            {
                rewardCode = rewardFunctionFile.text;
                Debug.Log($"Using reward function from file: {rewardFunctionFile.name}");
            }
            else if (useInspectorReward)
            {
                rewardCode = defaultRewardFunctionCode;
                Debug.Log("Using reward function from inspector text area");
            }
            
            // Initialize the reward function manager with proper async compilation
            bool success = await RewardFunctionManager.Initialize(
                ConfigManager.Instance.RewardPath,
                rewardCode,
                ConfigManager.Instance.UseFileBasedReward
            );
            
            if (success)
            {
                Debug.Log($"RewardFunctionManager initialized successfully. Using {GetRewardSource()}");
            }
            else
            {
                Debug.LogError("Failed to initialize RewardFunctionManager!");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error initializing RewardFunctionManager: {e.Message}");
        }
    }
    
    private string GetRewardSource()
    {
        if (ConfigManager.Instance.UseFileBasedReward && !string.IsNullOrEmpty(ConfigManager.Instance.RewardPath))
        {
            return $"file-based reward (ID: {ConfigManager.Instance.RewardPath})";
        }
        else if (rewardFunctionFile != null)
        {
            return $"reward file asset: {rewardFunctionFile.name}";
        }
        else if (useInspectorReward)
        {
            return "inspector reward function";
        }
        else
        {
            return "built-in default reward";
        }
    }
    
    /// <summary>
    /// Recompile the reward function at runtime (for testing purposes)
    /// Works with both file asset and inspector code
    /// </summary>
    [ContextMenu("Recompile Reward Function")]
    public async void RecompileReward()
    {
        if (!enableRuntimeRecompilation)
        {
            Debug.LogWarning("Runtime recompilation is disabled. Enable it in the inspector to use this feature.");
            return;
        }
        
        string codeToCompile = null;

        if (rewardFunctionFile != null)
        {
            codeToCompile = rewardFunctionFile.text;
            Debug.Log($"Recompiling reward function from file: {rewardFunctionFile.name}");
        }
        else if (useInspectorReward && !string.IsNullOrEmpty(defaultRewardFunctionCode))
        {
            codeToCompile = defaultRewardFunctionCode;
            Debug.Log("Recompiling reward function from inspector text area");
        }

        if (string.IsNullOrEmpty(codeToCompile))
        {
            Debug.LogWarning("No reward function code available. Assign a file or enable 'Use Inspector Reward' with code.");
            return;
        }

        try
        {
            Debug.Log("Recompiling reward function...");
            bool success = await RewardFunctionManager.RecompileRewardFunction(codeToCompile);
            
            if (success)
            {
                Debug.Log(" Reward function recompiled successfully!");
            }
            else
            {
                Debug.LogError(" Failed to recompile reward function. Check console for compilation errors.");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error recompiling reward function: {e.Message}");
        }
    }
    
    /// <summary>
    /// Validate the reward function syntax without compiling
    /// </summary>
    [ContextMenu("Validate Reward Function")]
    public void ValidateRewardFunction()
    {
        if (string.IsNullOrEmpty(defaultRewardFunctionCode))
        {
            Debug.LogWarning("No reward function code to validate.");
            return;
        }
        
        // Basic syntax validation
        var requiredElements = new string[]
        {
            "observations[",
            "context[",
            "rewards["
        };
        
        var missingElements = new System.Collections.Generic.List<string>();
        
        foreach (var element in requiredElements)
        {
            if (!defaultRewardFunctionCode.Contains(element))
            {
                missingElements.Add(element);
            }
        }
        
        if (missingElements.Count == 0)
        {
            Debug.Log(" Reward function syntax validation passed!");
        }
        else
        {
            Debug.LogWarning($"� Potential issues found. Missing elements: {string.Join(", ", missingElements)}");
        }
    }
    
    void OnValidate()
    {
        // Update tooltip based on current settings
        if (Application.isPlaying && useInspectorReward && enableRuntimeRecompilation)
        {
            // Could add real-time validation here if needed
        }
    }
    
    /// <summary>
    /// Create the reward function display UI component
    /// </summary>
    private void CreateDisplayUI()
    {
        try
        {
            // Check if display already exists
            RewardFunctionDisplay existingDisplay = FindFirstObjectByType<RewardFunctionDisplay>();
            if (existingDisplay != null)
            {
                Debug.Log("RewardFunctionDisplay already exists, skipping creation.");
                return;
            }
            
            // Create a new GameObject for the display
            GameObject displayGO = new GameObject("RewardFunctionDisplay");
            displayGO.AddComponent<RewardFunctionDisplay>();
            
            Debug.Log("Created RewardFunctionDisplay UI component (Press F1 to toggle)");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to create RewardFunctionDisplay: {e.Message}");
        }
    }
}