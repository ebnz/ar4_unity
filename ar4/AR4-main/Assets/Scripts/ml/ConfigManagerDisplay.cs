using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Text;

public class ConfigManagerDisplay : MonoBehaviour
{
    [Header("UI Components")]
    private GameObject displayPanel;
    private TextMeshProUGUI configText;
    private TextMeshProUGUI titleText;
    private ScrollRect scrollRect;
    
    [Header("Settings")]
    [SerializeField] private KeyCode toggleKey = KeyCode.F2;
    [SerializeField] private bool showOnStart = true;
    [SerializeField] private Color backgroundColor = new Color(0.1f, 0.1f, 0.1f, 0.9f);
    [SerializeField] private Color textColor = new Color(0.9f, 0.9f, 0.9f, 1f);
    [SerializeField] private int fontSize = 14;
    
    private Canvas canvas;
    private bool isVisible = false;
    
    void Awake()
    {
        CreateUIComponents();
        UpdateVisibility();
    }
    
    void Start()
    {
        isVisible = showOnStart;
        UpdateVisibility();
        UpdateConfigDisplay();
    }
    
    void Update()
    {
        if (Input.GetKeyDown(toggleKey))
        {
            ToggleVisibility();
        }
    }
    
    private void CreateUIComponents()
    {
        // Create Canvas if it doesn't exist
        canvas = FindFirstObjectByType<Canvas>();
        if (canvas == null)
        {
            GameObject canvasGO = new GameObject("UICanvas");
            canvas = canvasGO.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvasGO.AddComponent<CanvasScaler>();
            canvasGO.AddComponent<GraphicRaycaster>();
        }
        
        // Create main panel
        displayPanel = new GameObject("ConfigManagerDisplay");
        displayPanel.transform.SetParent(canvas.transform, false);
        
        RectTransform panelRect = displayPanel.AddComponent<RectTransform>();
        panelRect.anchorMin = new Vector2(0.6f, 0.1f);
        panelRect.anchorMax = new Vector2(0.95f, 0.9f);
        panelRect.offsetMin = Vector2.zero;
        panelRect.offsetMax = Vector2.zero;
        
        // Add background
        Image background = displayPanel.AddComponent<Image>();
        background.color = backgroundColor;
        
        // Create vertical layout
        VerticalLayoutGroup layoutGroup = displayPanel.AddComponent<VerticalLayoutGroup>();
        layoutGroup.padding = new RectOffset(10, 10, 10, 10);
        layoutGroup.spacing = 10;
        layoutGroup.childControlHeight = true;
        layoutGroup.childControlWidth = true;
        layoutGroup.childForceExpandHeight = false;
        layoutGroup.childForceExpandWidth = true;
        
        // Create title
        GameObject titleGO = new GameObject("Title");
        titleGO.transform.SetParent(displayPanel.transform, false);
        titleText = titleGO.AddComponent<TextMeshProUGUI>();
        titleText.text = "Configuration (Press " + toggleKey.ToString() + " to toggle)";
        titleText.fontSize = fontSize + 2;
        titleText.fontStyle = FontStyles.Bold;
        titleText.color = textColor;
        titleText.alignment = TextAlignmentOptions.Center;
        
        LayoutElement titleLayout = titleGO.AddComponent<LayoutElement>();
        titleLayout.preferredHeight = 30;
        titleLayout.flexibleHeight = 0;
        
        // Create scroll view
        GameObject scrollViewGO = new GameObject("ScrollView");
        scrollViewGO.transform.SetParent(displayPanel.transform, false);
        scrollRect = scrollViewGO.AddComponent<ScrollRect>();
        
        LayoutElement scrollLayout = scrollViewGO.AddComponent<LayoutElement>();
        scrollLayout.flexibleHeight = 1;
        scrollLayout.minHeight = 0;
        
        // Create viewport
        GameObject viewportGO = new GameObject("Viewport");
        viewportGO.transform.SetParent(scrollViewGO.transform, false);
        RectTransform viewportRect = viewportGO.AddComponent<RectTransform>();
        viewportRect.anchorMin = Vector2.zero;
        viewportRect.anchorMax = Vector2.one;
        viewportRect.sizeDelta = Vector2.zero;
        viewportRect.anchoredPosition = Vector2.zero;
        
        viewportGO.AddComponent<RectMask2D>();
        scrollRect.viewport = viewportRect;
        
        // Create content
        GameObject contentGO = new GameObject("Content");
        contentGO.transform.SetParent(viewportGO.transform, false);
        RectTransform contentRect = contentGO.AddComponent<RectTransform>();
        contentRect.anchorMin = new Vector2(0, 1);
        contentRect.anchorMax = new Vector2(1, 1);
        contentRect.pivot = new Vector2(0.5f, 1);
        contentRect.anchoredPosition = Vector2.zero;
        
        // Add vertical layout group to content for better spacing
        VerticalLayoutGroup contentLayout = contentGO.AddComponent<VerticalLayoutGroup>();
        contentLayout.padding = new RectOffset(10, 10, 10, 10);
        contentLayout.spacing = 5;
        contentLayout.childControlHeight = false;
        contentLayout.childControlWidth = true;
        contentLayout.childForceExpandHeight = false;
        contentLayout.childForceExpandWidth = true;
        
        ContentSizeFitter contentFitter = contentGO.AddComponent<ContentSizeFitter>();
        contentFitter.horizontalFit = ContentSizeFitter.FitMode.PreferredSize;
        contentFitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;
        
        scrollRect.content = contentRect;
        
        // Create config text
        GameObject configGO = new GameObject("ConfigText");
        configGO.transform.SetParent(contentGO.transform, false);
        configText = configGO.AddComponent<TextMeshProUGUI>();
        configText.fontSize = fontSize;
        configText.color = textColor;
        configText.alignment = TextAlignmentOptions.TopLeft;
        configText.font = Resources.Load<TMP_FontAsset>("Fonts & Materials/LiberationSans SDF");
        configText.textWrappingMode = TMPro.TextWrappingModes.Normal;
        configText.overflowMode = TextOverflowModes.Overflow;
        
        // Add layout element to config text with proper sizing
        LayoutElement configLayout = configGO.AddComponent<LayoutElement>();
        configLayout.preferredWidth = -1;
        configLayout.flexibleHeight = 1;

        RectTransform scrollRectTransform = scrollViewGO.GetComponent<RectTransform>();
        scrollRectTransform.anchorMin = new Vector2(0, 0);
        scrollRectTransform.anchorMax = new Vector2(1, 1);
        scrollRectTransform.offsetMin = Vector2.zero;
        scrollRectTransform.offsetMax = Vector2.zero;
        
        // Add ContentSizeFitter to the text for proper height calculation
        ContentSizeFitter textFitter = configGO.AddComponent<ContentSizeFitter>();
        textFitter.horizontalFit = ContentSizeFitter.FitMode.Unconstrained;
        textFitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;
        
        // Configure scrollbar
        GameObject scrollbarGO = new GameObject("Scrollbar");
        scrollbarGO.transform.SetParent(scrollViewGO.transform, false);
        Scrollbar scrollbar = scrollbarGO.AddComponent<Scrollbar>();
        scrollbar.direction = Scrollbar.Direction.BottomToTop;
        
        RectTransform scrollbarRect = scrollbarGO.GetComponent<RectTransform>();
        scrollbarRect.anchorMin = new Vector2(1, 0);
        scrollbarRect.anchorMax = new Vector2(1, 1);
        scrollbarRect.pivot = new Vector2(1, 0.5f);
        scrollbarRect.sizeDelta = new Vector2(20, 0);
        scrollbarRect.anchoredPosition = new Vector2(-5, 0);
        
        Image scrollbarBg = scrollbarGO.AddComponent<Image>();
        scrollbarBg.color = new Color(0.2f, 0.2f, 0.2f, 0.8f);
        
        GameObject handleGO = new GameObject("Handle");
        handleGO.transform.SetParent(scrollbarGO.transform, false);
        RectTransform handleRect = handleGO.AddComponent<RectTransform>();
        handleRect.sizeDelta = new Vector2(20, 20);
        
        Image handleImage = handleGO.AddComponent<Image>();
        handleImage.color = new Color(0.5f, 0.5f, 0.5f, 1f);
        
        scrollbar.targetGraphic = handleImage;
        scrollbar.handleRect = handleRect;
        scrollRect.verticalScrollbar = scrollbar;
        scrollRect.verticalScrollbarVisibility = ScrollRect.ScrollbarVisibility.AutoHideAndExpandViewport;
        scrollRect.verticalScrollbarSpacing = -3;
        
        // Set initial scroll position to top-left
        scrollRect.horizontalNormalizedPosition = 0f;
        scrollRect.verticalNormalizedPosition = 1f;
    }
    
    public void UpdateConfigDisplay()
    {
        if (configText == null) return;
        
        StringBuilder sb = new StringBuilder();
        
        if (ConfigManager.Instance.IsInitialized)
        {
            sb.AppendLine("<color=#4ECDC4>=== Configuration Settings ===</color>\n");
            
            // Core settings
            sb.AppendLine("<color=#95E1D3>Core Settings:</color>");
            sb.AppendLine($"  Run ID: {ConfigManager.Instance.RunId}");
            sb.AppendLine($"  ML-Agents Port: {ConfigManager.Instance.MLAgentsPort}");
            sb.AppendLine($"  Max Steps Per Episode: {ConfigManager.Instance.MaxStepsPerEpisode}");
            sb.AppendLine($"  Output File Path: {ConfigManager.Instance.OutputFilePath}");
            sb.AppendLine();
            
            // Reward settings
            sb.AppendLine("<color=#95E1D3>Reward Function:</color>");
            sb.AppendLine($"  Reward Path: {ConfigManager.Instance.RewardPath}");
            sb.AppendLine($"  Use File-Based: {ConfigManager.Instance.UseFileBasedReward}");
            sb.AppendLine();
            
            // Action & behavior settings
            sb.AppendLine("<color=#95E1D3>Action Settings:</color>");
            sb.AppendLine($"  Action Speed: {ConfigManager.Instance.ActionSpeed}");
            sb.AppendLine($"  Goal Threshold: {ConfigManager.Instance.GoalThreshold}");
            sb.AppendLine($"  Action Smoothing: {ConfigManager.Instance.ActionSmoothing}");
            
            configText.text = sb.ToString();
        }
        else
        {
            configText.text = "<color=#FF6B6B>ConfigManager not initialized</color>";
        }
        
        // Force layout rebuild to ensure proper sizing
        StartCoroutine(RefreshLayoutNextFrame());
        
        // Reset scroll position to top-left after layout rebuild
        StartCoroutine(ResetScrollPositionNextFrame());
    }
    
    private System.Collections.IEnumerator RefreshLayoutNextFrame()
    {
        yield return null; // Wait one frame
        if (scrollRect != null && scrollRect.content != null)
        {
            UnityEngine.UI.LayoutRebuilder.ForceRebuildLayoutImmediate(scrollRect.content);
        }
    }
    
    private System.Collections.IEnumerator ResetScrollPositionNextFrame()
    {
        yield return null; // Wait one frame for layout to complete
        yield return null; // Wait another frame to ensure everything is settled
        
        if (scrollRect != null)
        {
            // Force scroll to top-left corner
            scrollRect.normalizedPosition = new Vector2(0f, 1f);
        }
    }
    
    public void ToggleVisibility()
    {
        isVisible = !isVisible;
        UpdateVisibility();
        
        if (isVisible)
        {
            UpdateConfigDisplay();
        }
    }
    
    private void UpdateVisibility()
    {
        if (displayPanel != null)
        {
            displayPanel.SetActive(isVisible);
        }
    }
    
    public void SetVisibility(bool visible)
    {
        isVisible = visible;
        UpdateVisibility();
    }
    
    void OnDestroy()
    {
        if (displayPanel != null)
        {
            Destroy(displayPanel);
        }
    }
}