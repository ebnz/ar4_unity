using UnityEngine;

/// <summary>
/// Simple visualizer for detected AprilTags
/// Shows a cube with RGB axes
/// </summary>
public class TagVisualizer : MonoBehaviour
{
    public Color tagColor = Color.green;
    public float size = 0.2f;
    public int tagID = -1;

    void Awake()
    {
    }
}
