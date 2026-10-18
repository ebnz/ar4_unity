using System.IO;
using UnityEngine;

public class CameraCapture : MonoBehaviour
{
    private int counter = 0;

    public Camera leftCamera;
    public Camera rightCamera;

    public int imageWidth = 1080;
    public int imageHeight = 1080;

    public string saveFolder = "Screenshots";

    /// <summary>
    /// Saves the images from the left and right camera.
    /// </summary>
    public void CaptureAndSave()
    {
        CaptureCamera(leftCamera, $"Screenshots/LeftCamera_{counter}.png");
        CaptureCamera(rightCamera, $"Screenshots/RightCamera_{counter}.png");
    }

    /// <summary>
    /// Capture the scene from the camera.
    /// </summary>
    void CaptureCamera(Camera cam, string savePath)
    {
        // Create a temporary RenderTexture
        RenderTexture renderTexture = new RenderTexture(imageWidth, imageHeight, 24);
        cam.targetTexture = renderTexture;
        cam.Render();

        // Set up a Texture2D to copy the RenderTexture
        RenderTexture.active = renderTexture;
        Texture2D texture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        texture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        texture.Apply();

        // Save to disk
        byte[] bytes = texture.EncodeToPNG(); // Change to EncodeToJPG() for JPG
        File.WriteAllBytes(savePath, bytes);

        // Cleanup
        cam.targetTexture = null;
        RenderTexture.active = null;
        Destroy(renderTexture);
        Destroy(texture);

        Debug.Log($"Saved image to: {savePath}");
        counter += 1;
    }

    void Start()
    {
        // Ensure the save directory exists
        if (!Directory.Exists(saveFolder))
        {
            Directory.CreateDirectory(saveFolder);
        }

        CaptureAndSave();
    }

    void Update()
    {
        // Check if the "E" key was pressed
        if (Input.GetKeyDown(KeyCode.E))
        {
            CaptureAndSave();
        }
    }
}
