using System;
using System.Collections.Generic;
using UnityEngine;
using System.Linq.Expressions;
using System.Reflection;

public class UnityExpressionEvaluator : MonoBehaviour
{
    [Header("Test Expressions")]
    public float x = 5;
    public float y = 3;
    
    void Start()
    {
        Debug.Log("=== Unity Expression Evaluator ===");
        
        // Test 1: Simple math expression
        TestSimpleMath();
        
        // Test 2: Using Unity's Mathf
        TestUnityMath();
        
        // Test 3: Dynamic delegate creation
        TestDynamicDelegate();
    }
    
    void TestSimpleMath()
    {
        Debug.Log("\n--- Test 1: Simple Math ---");
        
        // Create expression: x + y
        var xParam = Expression.Parameter(typeof(float), "x");
        var yParam = Expression.Parameter(typeof(float), "y");
        var body = Expression.Add(xParam, yParam);
        
        var lambda = Expression.Lambda<Func<float, float, float>>(body, xParam, yParam);
        var compiled = lambda.Compile();
        
        float result = compiled(x, y);
        Debug.Log($"{x} + {y} = {result}");
    }
    
    void TestUnityMath()
    {
        Debug.Log("\n--- Test 2: Unity Math ---");
        
        // Create expression: Mathf.Sqrt(x * x + y * y)
        var xParam = Expression.Parameter(typeof(float), "x");
        var yParam = Expression.Parameter(typeof(float), "y");
        
        // x * x
        var xSquared = Expression.Multiply(xParam, xParam);
        // y * y  
        var ySquared = Expression.Multiply(yParam, yParam);
        // x*x + y*y
        var sum = Expression.Add(xSquared, ySquared);
        
        // Mathf.Sqrt
        var sqrtMethod = typeof(Mathf).GetMethod("Sqrt");
        var sqrtCall = Expression.Call(sqrtMethod, sum);
        
        var lambda = Expression.Lambda<Func<float, float, float>>(sqrtCall, xParam, yParam);
        var compiled = lambda.Compile();
        
        float result = compiled(x, y);
        Debug.Log($"Distance from origin: {result}");
    }
    
    void TestDynamicDelegate()
    {
        Debug.Log("\n--- Test 3: Dynamic Delegate ---");
        
        // Build a reward-like function dynamically
        var observations = new Dictionary<string, float>
        {
            { "distance", 0.5f },
            { "velocity", 1.2f }
        };
        
        // Create a simple reward function: exp(-distance) + velocity * 0.1
        var dictParam = Expression.Parameter(typeof(Dictionary<string, float>), "obs");
        
        // obs["distance"]
        var distanceKey = Expression.Constant("distance");
        var getItem = typeof(Dictionary<string, float>).GetMethod("get_Item");
        var distance = Expression.Call(dictParam, getItem, distanceKey);
        
        // -distance
        var negDistance = Expression.Negate(distance);
        
        // Mathf.Exp(-distance)
        var expMethod = typeof(Mathf).GetMethod("Exp");
        var expCall = Expression.Call(expMethod, negDistance);
        
        // obs["velocity"] * 0.1f
        var velocityKey = Expression.Constant("velocity");
        var velocity = Expression.Call(dictParam, getItem, velocityKey);
        var factor = Expression.Constant(0.1f);
        var velocityTerm = Expression.Multiply(velocity, factor);
        
        // exp(-distance) + velocity * 0.1
        var total = Expression.Add(expCall, velocityTerm);
        
        var lambda = Expression.Lambda<Func<Dictionary<string, float>, float>>(total, dictParam);
        var rewardFunc = lambda.Compile();
        
        float reward = rewardFunc(observations);
        Debug.Log($"Reward calculation: exp(-{observations["distance"]}) + {observations["velocity"]} * 0.1 = {reward}");
    }
    
    [ContextMenu("Run Tests")]
    void RunTests()
    {
        TestSimpleMath();
        TestUnityMath();
        TestDynamicDelegate();
    }
}