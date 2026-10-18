using UnityEngine;

public class RotateAutomatically : MonoBehaviour
{
    [Header("Rotation Speed (degrees per second)")]
    public float xVelocity = 0f;
    public float yVelocity = 0f;
    public float zVelocity = 0f;

    [Header("Rotation Toggle")]
    public bool isRotating = true;

    void Update()
    {
        if (!isRotating) return;

        float x = xVelocity * Time.deltaTime;
        float y = yVelocity * Time.deltaTime;
        float z = zVelocity * Time.deltaTime;

        transform.Rotate(x, y, z, Space.Self);
    }
}
