using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using System.Linq;
using UnityEngine;
using Microsoft.CodeAnalysis.CSharp.Scripting;
using Microsoft.CodeAnalysis.Scripting;

/// <summary>
/// Manages dynamic reward function loading and execution for Eureka-style reward generation
/// Static class for consistent architecture with ConfigManager
/// </summary>
public static class RewardFunctionManager
{

    // Current reward function delegate - updated to return RewardResult
    public delegate RewardResult RewardFunction(
        Dictionary<string, float> observations, 
        Dictionary<string, object> context
    );
    
    // Static fields instead of instance fields
    private static RewardFunction currentRewardFunction;
    private static string currentRewardCode;
    private static Dictionary<string, float> rewardComponents = new Dictionary<string, float>();
    private static bool isInitialized = false;
    private static string lastCompilationError = "";
    private static bool hasRuntimeError = false;
    private static string lastRuntimeError = "";

    // Thread safety
    private static readonly object lockObject = new object();

    // Static properties
    public static bool IsInitialized => isInitialized;
    public static string LastCompilationError => lastCompilationError;
    public static bool HasRuntimeError => hasRuntimeError;
    public static string LastRuntimeError => lastRuntimeError;
    
    
    
    
    /// <summary>
    /// Initialize the reward function manager with specified or default reward code
    /// Priority: File-based -> Fallback code -> Built-in default
    /// </summary>
    public static async Task<bool> Initialize(string rewardPath, string fallbackRewardCode, bool useFileBasedReward)
    {
        lock (lockObject)
        {
            if (isInitialized)
            {
                Debug.Log("RewardFunctionManager already initialized, skipping...");
                return true;
            }
        }
        
        try
        {
            Debug.Log($"Initializing RewardFunctionManager: rewardPath={rewardPath}, useFileBasedReward={useFileBasedReward}");
            
            string rewardCode = null;
            
            // Priority 1: File-based reward (if rewardPath provided)
            if (useFileBasedReward && !string.IsNullOrEmpty(rewardPath))
            {
                Debug.Log($"Loading reward function from file: {rewardPath}");
                rewardCode = LoadRewardFunctionFromFile(rewardPath);
            }
            // Priority 2: Fallback reward code (from inspector or agent)
            else if (!string.IsNullOrEmpty(fallbackRewardCode))
            {
                Debug.Log("Using fallback reward function");
                rewardCode = fallbackRewardCode;
            }
            // Priority 3: No built-in default - require fallback
            else
            {
                Debug.LogError("No reward function provided! RewardFunctionManagerInit should provide a fallback.");
                lock (lockObject)
                {
                    isInitialized = false;
                }
                return false;
            }
            
            // Compile the reward function
            bool success = await CompileRewardFunctionFromCode(rewardCode);
            lock (lockObject)
            {
                isInitialized = success;
            }
            return success;
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to initialize reward function: {e.Message}");
            lock (lockObject)
            {
                isInitialized = false;
            }
            return false;
        }
    }
    
    
    /// <summary>
    /// Load a reward function from C# code string
    /// </summary>
    private static async Task LoadRewardFunctionFromCode(string code, int sampleId = 0)
    {
        try
        {
            // Create script options with necessary references
            var scriptOptions = ScriptOptions.Default
                .WithReferences(
                    typeof(UnityEngine.Object).Assembly,
                    typeof(System.Math).Assembly,
                    typeof(Dictionary<,>).Assembly,
                    typeof(RewardFunctionManager).Assembly // Add current assembly
                )
                .WithImports(
                    "System",
                    "System.Collections.Generic",
                    "UnityEngine"
                );
                
            // Wrap the code in a lambda with auto-generated boilerplate
            string wrappedCode = $@"
                (Dictionary<string, float> observations, Dictionary<string, object> context) =>
                {{
                    var rewards = new Dictionary<string, float>();
                    float totalReward = 0f; // Optional override
                    
                    {code}
                    
                    // Auto-calculate total if not explicitly set
                    if (totalReward == 0f) {{
                        foreach (var r in rewards.Values) totalReward += r;
                    }}
                    
                    return new RewardFunctionManager.RewardResult {{ Total = totalReward, Components = rewards }};
                }}
            ";

            Debug.Log(wrappedCode);
            
            // Compile and execute
            var result = await CSharpScript.EvaluateAsync<RewardFunction>(
                wrappedCode, 
                scriptOptions
            );
            
            if (result != null)
            {
                currentRewardFunction = result;
                currentRewardCode = code;
                Debug.Log($"Successfully loaded reward function {sampleId}");
            }
        }
        catch (Exception e)
        {
            lastCompilationError = e.Message;
            Debug.LogError($"Failed to load reward function: {e.Message}");
            // No fallback - let initialization fail
            currentRewardFunction = null;
            throw; // Re-throw the exception so the caller can handle it
        }
    }
    
    /// <summary>
    /// Handles reward function compilation failures by logging and exiting
    /// </summary>
    private static void HandleCompilationError(string error, int runId)
    {
        lastCompilationError = error;
        Debug.LogError($"Reward function compilation failed: {error}");

        // Create and save structured error result for Eureka dashboard
        var errorResult = CreateCompilationErrorResult(error, runId);
        SaveCompilationErrorResult(errorResult);

        Debug.Log($"Compilation error result saved for run {runId}");

        // Do not quit the application. Let Python detect the error file and handle the retry.
        // Application.Quit(1);
    }

    /// <summary>
    /// Handles runtime errors in reward function execution (e.g., missing dictionary keys)
    /// </summary>
    private static void HandleRuntimeError(string error, int runId)
    {
        hasRuntimeError = true;
        lastRuntimeError = error;
        Debug.LogError($"Runtime error in reward function: {error}");

        // Create and save structured error result (reuse compilation error structure with different type)
        var errorResult = CreateRuntimeErrorResult(error, runId);
        SaveCompilationErrorResult(errorResult);

        Debug.Log($"Runtime error result saved for run {runId}");

        // Quit application after saving error - Python will detect the error and retry
        // This prevents running the full training with 0 rewards
        Debug.Log("Quitting application due to runtime error...");
        Application.Quit(1);
    }

    /// <summary>
    /// Create a runtime error result
    /// </summary>
    private static CompilationErrorResult CreateRuntimeErrorResult(string errorMessage, int runId)
    {
        var errorResult = new CompilationErrorResult
        {
            type = "runtime_error",
            error_message = errorMessage,
            run_id = runId,
            reward_path = ConfigManager.Instance.RewardPath
        };

        return errorResult;
    }

    /// <summary>
    /// Load reward function from file and return the code
    /// </summary>
    private static string LoadRewardFunctionFromFile(string rewardPath)
    {
        try
        {
            string filePath;
            if (Path.IsPathRooted(rewardPath))
            {
                // Use the full path directly
                filePath = rewardPath;
            }
            else
            {
                // Create the path to the reward function file
                string rewardsDir = Path.Combine(Application.persistentDataPath, "EurekaRewards", "rewards");
                filePath = Path.Combine(rewardsDir, rewardPath);
            }

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"Reward function file not found: {filePath}");
            }

            string content = File.ReadAllText(filePath);
            Debug.Log($"Loaded reward function from: {filePath}");
            return content;
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to load reward function from file: {e.Message}");
            throw;
        }
    }
    
    
    /// <summary>
    /// Compile reward function from code string
    /// </summary>
    private static async Task<bool> CompileRewardFunctionFromCode(string code)
    {
        try
        {
            currentRewardCode = code;
            
            // Await the async compilation directly. This ensures that if an exception is thrown,
            // the catch block executes on the main Unity thread, making Application.Quit() safe to call.
            await LoadRewardFunctionFromCode(code, ConfigManager.Instance.RunId);
            
            return currentRewardFunction != null;
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to compile reward function: {e.Message}");
            HandleCompilationError(e.Message, ConfigManager.Instance.RunId);
            return false;
        }
    }
    
    /// <summary>
    /// Execute the current reward function
    /// </summary>
    public static (float totalReward, Dictionary<string, float> components) ComputeReward(
        Dictionary<string, float> observations,
        Dictionary<string, object> context)
    {
        if (currentRewardFunction != null)
        {
            try
            {
                var result = currentRewardFunction(observations, context);
                return (result.Total, result.Components);
            }
            catch (KeyNotFoundException e)
            {
                // Handle missing dictionary key as a runtime error (same as compilation error)
                if (!hasRuntimeError) // Only report once
                {
                    HandleRuntimeError($"Missing key in reward function: {e.Message}", ConfigManager.Instance.RunId);
                }
                return (0f, new Dictionary<string, float>());
            }
            catch (Exception e)
            {
                // Check if it's a dictionary key error wrapped in another exception
                if (e.InnerException is KeyNotFoundException keyEx)
                {
                    if (!hasRuntimeError)
                    {
                        HandleRuntimeError($"Missing key in reward function: {keyEx.Message}", ConfigManager.Instance.RunId);
                    }
                    return (0f, new Dictionary<string, float>());
                }
                Debug.LogError($"Error executing reward function: {e.Message}");
                return (0f, new Dictionary<string, float>());
            }
        }

        // No built-in default - RewardFunctionManagerInit should provide fallback
        Debug.LogError("No reward function loaded! Make sure RewardFunctionManagerInit is properly configured.");
        return (0f, new Dictionary<string, float>());
    }
    
    /// <summary>
    /// Execute reward function with async signature for compatibility with RobotAgentDynamic
    /// </summary>
    public static Task<RewardResult> ExecuteRewardFunction(Dictionary<string, float> observations, Dictionary<string, object> context)
    {
        // Current implementation is synchronous, so we wrap it in a completed task
        try
        {
            var (totalReward, components) = ComputeReward(observations, context);
            return Task.FromResult(new RewardResult
            {
                Total = totalReward,
                Components = components
            });
        }
        catch (Exception e)
        {
            Debug.LogError($"Error executing reward function: {e.Message}");
            return Task.FromResult(new RewardResult
            {
                Total = 0f,
                Components = new Dictionary<string, float>()
            });
        }
    }
    
    /// <summary>
    /// Result structure for reward function execution
    /// </summary>
    public class RewardResult
    {
        public float Total { get; set; }
        public Dictionary<string, float> Components { get; set; }
    }
    
    
    /// <summary>
    /// Recompile the current reward function (for runtime testing)
    /// </summary>
    public static async Task<bool> RecompileRewardFunction(string newCode)
    {
        try
        {
            Debug.Log("Recompiling reward function...");
            bool success = await CompileRewardFunctionFromCode(newCode);
            
            if (success)
            {
                Debug.Log("Reward function recompiled successfully");
            }
            
            return success;
        }
        catch (Exception e)
        {
            Debug.LogError($"Error recompiling reward function: {e.Message}");
            return false;
        }
    }
    
    /// <summary>
    /// Get the current reward components for logging
    /// </summary>
    public static Dictionary<string, float> GetRewardComponents()
    {
        return new Dictionary<string, float>(rewardComponents);
    }
    
    /// <summary>
    /// Get the current reward function code for display
    /// </summary>
    public static string GetCurrentRewardFunctionCode()
    {
        return currentRewardCode ?? "";
    }
    
    /// <summary>
    /// Get the last compilation error for display
    /// </summary>
    public static string GetLastCompilationError()
    {
        return lastCompilationError;
    }

    /// <summary>
    /// Get the last runtime error for display
    /// </summary>
    public static string GetLastRuntimeError()
    {
        return lastRuntimeError;
    }

    /// <summary>
    /// Create a compilation error result
    /// </summary>
    private static CompilationErrorResult CreateCompilationErrorResult(string errorMessage, int runId)
    {
        var errorResult = new CompilationErrorResult
        {
            type = "compilation_error",
            error_message = errorMessage,
            run_id = runId,
            reward_path = ConfigManager.Instance.RewardPath
        };

        return errorResult;
    }

    /// <summary>
    /// Save compilation error result to JSON file
    /// </summary>
    private static void SaveCompilationErrorResult(CompilationErrorResult errorResult)
    {
        try
        {
            string fullPath = ConfigManager.Instance.GetResultsFilePath();

            // Create a results list with just the error result
            var resultsList = new CompilationErrorResultsList();
            resultsList.results.Add(errorResult);

            string json = JsonUtility.ToJson(resultsList, true);
            File.WriteAllText(fullPath, json);

            Debug.Log($"Compilation error result saved to: {fullPath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to save compilation error result: {e.Message}");
        }
    }
}

/// <summary>
/// Minimal structure for compilation error reporting
/// </summary>
[Serializable]
public class CompilationErrorResult
{
    public string type;
    public string error_message;
    public int run_id;
    public string reward_path;
}

/// <summary>
/// Wrapper for compilation error results list (matches Python expected format)
/// </summary>
[Serializable]
public class CompilationErrorResultsList
{
    public List<CompilationErrorResult> results = new List<CompilationErrorResult>();
}

/// <summary>
/// Example of how to integrate with RobotAgent
/// </summary>
public static class RobotAgentRewardIntegration
{
    public static void Example()
    {
        // In RobotAgent.OnActionReceived:
        /*
        var context = new RewardFunctionManager.RewardContext
        {
            previousDistance = previousDistanceToGoal,
            timeStep = Time.fixedDeltaTime,
            episodeSteps = currentEpisodeSteps,
            robotBase = robotManager.robotJoints[0].transform,
            endEffector = robotManager.gripper[0],
            target = robotManager.target,
            goal = goalPosition,
            goalThreshold = goalThreshold
        };
        
        var (totalReward, components) = RewardFunctionManager.Instance.ComputeReward(
            GetCurrentObservations(),
            smoothedAction,
            context
        );
        
        // Add reward with components
        foreach (var comp in components)
        {
            AddRewardWithType(comp.Key, comp.Value);
        }
        */
    }
}