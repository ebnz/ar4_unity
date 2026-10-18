using System.IO;
using UnityEngine;

public class FrameCapture : MonoBehaviour
{
    public Camera captureCamera; // Camera used for capturing
    public int imageWidth = 1080;
    public int imageHeight = 1080;
    public string folderPath = "CapturedFrames";

    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private int frameCount;//

    void Start()
    {
        // Create folder if it doesn't exist
        if (!Directory.Exists(folderPath))
        {
            Directory.CreateDirectory(folderPath);
        }

        // Set up RenderTexture
        renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        captureCamera.targetTexture = renderTexture;

        // Create a Texture2D to store the captured image
        texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
    }

    void OnDestroy()
    {
        // Clean up
        captureCamera.targetTexture = null;
        renderTexture.Release();
        Destroy(renderTexture);
        Destroy(texture2D);
    }

    void Update()
    {
        // Render the camera's view to the RenderTexture
        captureCamera.Render();

        // Read the RenderTexture into the Texture2D
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture2D.Apply();
        RenderTexture.active = null;

        // Save the frame as a PNG file
        byte[] bytes = texture2D.EncodeToPNG();
        string fileName = $"{folderPath}/frame_{frameCount:D6}.png";
        File.WriteAllBytes(fileName, bytes);

        frameCount++;
    }
}