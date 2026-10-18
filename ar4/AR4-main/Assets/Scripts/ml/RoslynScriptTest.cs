using System;
using System.Threading.Tasks;
using UnityEngine;
using Microsoft.CodeAnalysis.CSharp.Scripting;
using Microsoft.CodeAnalysis.Scripting;
using System.Collections.Generic;

public class RoslynScriptTest : MonoBehaviour
{
    async void Start()
    {
        Debug.Log("=== Roslyn Script Test ===");
        
        await TestSimpleExpression();
        await TestWithParameters();
        await TestUnityTypes();
        await TestRewardFunction();
    }
    
    async Task TestSimpleExpression()
    {
        Debug.Log("\n--- Test 1: Simple Expression ---");
        try
        {
            int result = await CSharpScript.EvaluateAsync<int>("1 + 2");
            Debug.Log($"1 + 2 = {result}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error: {e.Message}");
        }
    }
    
    async Task TestWithParameters()
    {
        Debug.Log("\n--- Test 2: With Parameters ---");
        try
        {
            var globals = new Globals { X = 5, Y = 3 };
            int result = await CSharpScript.EvaluateAsync<int>("X + Y", globals: globals);
            Debug.Log($"X + Y = {result} (where X={globals.X}, Y={globals.Y})");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error: {e.Message}");
        }
    }
    
    async Task TestUnityTypes()
    {
        Debug.Log("\n--- Test 3: Unity Math ---");
        try
        {
            var options = ScriptOptions.Default
                .WithReferences(typeof(UnityEngine.Mathf).Assembly)
                .WithImports("UnityEngine");
                
            float result = await CSharpScript.EvaluateAsync<float>(
                "Mathf.Sqrt(9f) + Mathf.Sin(Mathf.PI / 2f)", 
                options);
                
            Debug.Log($"Sqrt(9) + Sin(PI/2) = {result}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error: {e.Message}");
        }
    }
    
    async Task TestRewardFunction()
    {
        Debug.Log("\n--- Test 4: Reward Function ---");
        try
        {
            var options = ScriptOptions.Default
                .WithReferences(
                    typeof(UnityEngine.Mathf).Assembly,
                    typeof(Dictionary<,>).Assembly
                )
                .WithImports("UnityEngine", "System.Collections.Generic");
            
            // Define reward function code
            string code = @"
                var distance = observations[""distance""];
                var reward = Mathf.Exp(-2f * distance);
                var success = distance < 0.1f ? 10f : 0f;
                return reward + success;
            ";
            
            // Test with different observations
            var observations1 = new Dictionary<string, float> { { "distance", 0.5f } };
            var globals1 = new RewardGlobals { observations = observations1 };
            
            float reward1 = await CSharpScript.EvaluateAsync<float>(code, options, globals1);
            Debug.Log($"Reward for distance=0.5: {reward1}");
            
            var observations2 = new Dictionary<string, float> { { "distance", 0.05f } };
            var globals2 = new RewardGlobals { observations = observations2 };
            
            float reward2 = await CSharpScript.EvaluateAsync<float>(code, options, globals2);
            Debug.Log($"Reward for distance=0.05: {reward2}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error: {e.Message}");
        }
    }
    
    public class Globals
    {
        public int X { get; set; }
        public int Y { get; set; }
    }
    
    public class RewardGlobals
    {
        public Dictionary<string, float> observations { get; set; }
    }
}