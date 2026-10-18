using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class RewardFunctionDisplay : MonoBehaviour
{
    [Header("UI Components")]
    private GameObject displayPanel;
    private TextMeshProUGUI codeText;
    private TextMeshProUGUI titleText;
    private ScrollRect scrollRect;
    
    [Header("Settings")]
    [SerializeField] private KeyCode toggleKey = KeyCode.F1;
    [SerializeField] private bool showOnStart = true;
    [SerializeField] private Color backgroundColor = new Color(0.1f, 0.1f, 0.1f, 0.9f);
    [SerializeField] private Color textColor = new Color(0.9f, 0.9f, 0.9f, 1f);
    [SerializeField] private int fontSize = 14;
    
    private Canvas canvas;
    private bool isVisible = false;
    private bool hasShownRuntimeError = false;

    void Awake()
    {
        CreateUIComponents();
        UpdateVisibility();
    }
    
    void Start()
    {
        isVisible = showOnStart;
        UpdateVisibility();
        UpdateRewardFunctionDisplay();
    }
    
    void Update()
    {
        if (Input.GetKeyDown(toggleKey))
        {
            ToggleVisibility();
        }

        // Check for runtime errors and update display if one occurred
        if (!hasShownRuntimeError && RewardFunctionManager.HasRuntimeError)
        {
            hasShownRuntimeError = true;
            UpdateRewardFunctionDisplay();
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
        displayPanel = new GameObject("RewardFunctionDisplay");
        displayPanel.transform.SetParent(canvas.transform, false);
        
        RectTransform panelRect = displayPanel.AddComponent<RectTransform>();
        panelRect.anchorMin = new Vector2(0.05f, 0.1f);
        panelRect.anchorMax = new Vector2(0.4f, 0.9f);
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
        titleText.text = "Reward Function (Press " + toggleKey.ToString() + " to toggle)";
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
        
        // Create code text
        GameObject codeGO = new GameObject("CodeText");
        codeGO.transform.SetParent(contentGO.transform, false);
        codeText = codeGO.AddComponent<TextMeshProUGUI>();
        codeText.fontSize = fontSize;
        codeText.color = textColor;
        codeText.alignment = TextAlignmentOptions.TopLeft;
        codeText.font = Resources.Load<TMP_FontAsset>("Fonts & Materials/LiberationSans SDF");
        codeText.textWrappingMode = TMPro.TextWrappingModes.Normal;
        codeText.overflowMode = TextOverflowModes.Overflow;
        
        // Add layout element to code text with proper sizing
        LayoutElement codeLayout = codeGO.AddComponent<LayoutElement>();
        codeLayout.preferredWidth = -1;
        codeLayout.flexibleHeight = 1;

        RectTransform scrollRectTransform = scrollViewGO.GetComponent<RectTransform>();
        scrollRectTransform.anchorMin = new Vector2(0, 0);
        scrollRectTransform.anchorMax = new Vector2(1, 1);
        scrollRectTransform.offsetMin = Vector2.zero;
        scrollRectTransform.offsetMax = Vector2.zero;
        
        // Add ContentSizeFitter to the text for proper height calculation
        ContentSizeFitter textFitter = codeGO.AddComponent<ContentSizeFitter>();
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
    
    public void UpdateRewardFunctionDisplay()
    {
        if (codeText == null) return;
        
        string rewardCode = RewardFunctionManager.GetCurrentRewardFunctionCode();
        string compilationError = RewardFunctionManager.GetLastCompilationError();
        string runtimeError = RewardFunctionManager.GetLastRuntimeError();

        if (!string.IsNullOrEmpty(compilationError))
        {
            codeText.text = $"<color=#FF6B6B>COMPILATION FAILED:\n{compilationError}</color>\n\n<color=#888888>--- CODE ---</color>\n{rewardCode}";
        }
        else if (!string.IsNullOrEmpty(runtimeError))
        {
            codeText.text = $"<color=#FF6B6B>RUNTIME ERROR:\n{runtimeError}</color>\n\n<color=#888888>--- CODE ---</color>\n{rewardCode}";
        }
        else if (!string.IsNullOrEmpty(rewardCode))
        {
            codeText.text = rewardCode;
        }
        else
        {
            codeText.text = "<color=#FF6B6B>No reward function loaded</color>";
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
            UpdateRewardFunctionDisplay();
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