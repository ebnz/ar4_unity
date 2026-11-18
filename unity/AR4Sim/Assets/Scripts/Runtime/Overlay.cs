using System;
using System.Collections;
using System.IO;
using UnityEngine;

namespace AR4Sim
{
    /// <summary>
    /// On-screen overlay "Group | Task | date time | t" and a Screenshot button.
    /// The button hides itself during the capture, so the saved image shows the scene and the
    /// overlay text only. Files go to Screenshots/ next to the player (Build/Screenshots) or,
    /// in the Editor, next to the project folder (unity/AR4Sim/Screenshots).
    /// </summary>
    public class Overlay : MonoBehaviour
    {
        public string group = "-";
        public string task = "-";
        [Tooltip("Simulation time shown in the overlay [s]; negative hides it")]
        public double simTime = -1;

        bool capturing;
        string message;
        float messageUntil;
        GUIStyle textStyle, buttonStyle;

        public static string ScreenshotDir => Path.Combine(Path.GetDirectoryName(Application.dataPath) ?? ".", "Screenshots");

        public string Text
        {
            get
            {
                var t = $"Group {group}  |  Task {task}  |  {DateTime.Now:yyyy-MM-dd HH:mm:ss}";
                return simTime >= 0 ? t + $"  |  t = {simTime:0.0} s" : t;
            }
        }

        void OnGUI()
        {
            int fs = Mathf.Max(12, Screen.height / 36);
            if (textStyle == null || textStyle.fontSize != fs)
            {
                textStyle = new GUIStyle(GUI.skin.box) { fontSize = fs, alignment = TextAnchor.MiddleLeft, padding = new RectOffset(fs / 2, fs / 2, fs / 4, fs / 4) };
                textStyle.normal.textColor = Color.white;
                buttonStyle = new GUIStyle(GUI.skin.button) { fontSize = fs };
            }
            var content = new GUIContent(Text);
            var size = textStyle.CalcSize(content);
            GUI.Box(new Rect(10, 10, size.x, size.y), content, textStyle);

            if (capturing) return;
            var btn = new Rect(Screen.width - 10 - 8 * fs, 10, 8 * fs, size.y);
            if (GUI.Button(btn, "Screenshot", buttonStyle)) StartCoroutine(SaveScreenshot());
            if (message != null && Time.unscaledTime < messageUntil)
            {
                var m = new GUIContent(message);
                var ms = textStyle.CalcSize(m);
                GUI.Box(new Rect(10, Screen.height - 10 - ms.y, ms.x, ms.y), m, textStyle);
            }
        }

        /// <summary>Capture the screen (scene + overlay text, without the button) at the end of a frame.</summary>
        public IEnumerator Capture(Action<Texture2D> done)
        {
            if (Application.isBatchMode)
                throw new InvalidOperationException("screenshot needs a window (player started with -batchmode)");
            capturing = true;
            yield return null;                         // one frame without the button
            yield return new WaitForEndOfFrame();
            var tex = ScreenCapture.CaptureScreenshotAsTexture();
            capturing = false;
            done(tex);
        }

        IEnumerator SaveScreenshot()
        {
            Texture2D tex = null;
            yield return Capture(t => tex = t);
            try
            {
                Directory.CreateDirectory(ScreenshotDir);
                string safeTask = string.Join("_", task.Split(Path.GetInvalidFileNameChars()));
                string path = Path.Combine(ScreenshotDir, $"{safeTask}_{DateTime.Now:yyyyMMdd_HHmmss}.png");
                File.WriteAllBytes(path, tex.EncodeToPNG());
                message = $"saved {path}";
                Debug.Log($"[AR4 overlay] {message}");
            }
            catch (Exception e)
            {
                message = $"screenshot failed: {e.Message}";
                Debug.LogError($"[AR4 overlay] {message}");
            }
            finally
            {
                Destroy(tex);
            }
            messageUntil = Time.unscaledTime + 5f;
        }
    }
}
