using UnityEngine;
using TMPro;
using System.Collections.Generic;
using System.Linq;

public class RobotAgentVisualizer : MonoBehaviour
{
    [Header("References")]
    public MonoBehaviour robotAgentComponent;
    
    [Header("UI Settings")]
    public Vector3 panelOffset = new Vector3(0f, 1.5f, 0f);
    public float panelWidth = 1.5f;
    public float panelHeight = 1.0f;
    public Color backgroundColor = new Color(0, 0, 0, 0.8f);
    public Color textColor = Color.white;
    public float updateInterval = 0.1f;

    // UI Elements
    private GameObject visualizerPanel;
    private TextMeshPro stepsText;
    private TextMeshPro rewardsText;
    private float timeSinceLastUpdate = 0f;
    
    // Cache the agent interface
    private IEurekaAgent robotAgent;

    void Start()
    {
        // Try to get from assigned component first
        if (robotAgentComponent != null && robotAgentComponent is IEurekaAgent)
        {
            robotAgent = robotAgentComponent as IEurekaAgent;
        }
        else
        {
            // Find any IEurekaAgent in the scene
            var allAgents = FindObjectsByType<MonoBehaviour>(FindObjectsSortMode.None);
            robotAgent = allAgents.OfType<IEurekaAgent>().FirstOrDefault();
        }
            
        if (robotAgent == null)
        {
            Debug.LogError("No IEurekaAgent found for visualization!");
            enabled = false;
            return;
        }
        
        CreateVisualizerPanel();
    }

    void Update()
    {
        timeSinceLastUpdate += Time.deltaTime;
        
        if (timeSinceLastUpdate >= updateInterval)
        {
            UpdateVisualizer();
            timeSinceLastUpdate = 0f;
        }
    }
    
    private void CreateVisualizerPanel()
    {
        // Create parent object
        visualizerPanel = new GameObject("VisualizerPanel");
        visualizerPanel.transform.parent = transform;
        visualizerPanel.transform.localPosition = panelOffset;
        visualizerPanel.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
        
        // Add background quad
        GameObject background = GameObject.CreatePrimitive(PrimitiveType.Quad);
        background.transform.parent = visualizerPanel.transform;
        background.transform.localPosition = Vector3.zero;
        background.transform.localScale = new Vector3(panelWidth, panelHeight, 1f);
        
        // Set background material
        Renderer backgroundRenderer = background.GetComponent<Renderer>();
        Material mat = new Material(Shader.Find("Standard"));
        mat.color = backgroundColor;
        backgroundRenderer.material = mat;
        
        // Create text elements
        stepsText = CreateTextElement("StepsText", new Vector3(0f, panelHeight * 0.4f, -0.01f), 0.08f);
        rewardsText = CreateTextElement("RewardsText", new Vector3(0f, -panelHeight * 0.2f, -0.01f), 0.08f);
    }
    
    private TextMeshPro CreateTextElement(string name, Vector3 localPosition, float fontSize)
    {
        GameObject textObj = new GameObject(name);
        textObj.transform.parent = visualizerPanel.transform;
        textObj.transform.localPosition = localPosition;
        textObj.transform.localRotation = Quaternion.identity;
        
        TextMeshPro tmp = textObj.AddComponent<TextMeshPro>();
        tmp.alignment = TextAlignmentOptions.Center;
        tmp.fontSize = fontSize * 10;
        tmp.color = textColor;
        tmp.text = name;
        
        return tmp;
    }
    
    private void UpdateVisualizer()
    {
        if (robotAgent == null) return;
        
        // Update steps
        stepsText.text = $"Steps: {robotAgent.GetCurrentEpisodeSteps()} / {robotAgent.GetMaxStepsPerEpisode()}";

        // Get detailed rewards by type
        Dictionary<string, float> detailedRewards = robotAgent.GetRewardsByType();
        
        // Calculate total from dictionary values
        float totalReward = 0f;
        foreach (var reward in detailedRewards.Values)
        {
            totalReward += reward;
        }
        
        // Build reward text
        System.Text.StringBuilder rewardText = new System.Text.StringBuilder();
        rewardText.AppendLine($"Total Reward: {totalReward:F2}");
        
        // Add each reward type
        foreach (var reward in detailedRewards)
        {
            if (reward.Value != 0)
            {
                rewardText.AppendLine($"{reward.Key}: {reward.Value:F2}");
            }
        }
        
        rewardsText.text = rewardText.ToString();
    }
} 