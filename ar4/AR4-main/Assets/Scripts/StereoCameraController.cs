using UnityEngine;

public class StereoCameraController : MonoBehaviour
{
    public Transform leftCamera;
    public Transform rightCamera;
    public float interaxialDistance = 0.064f;

    void LateUpdate()
    {
        leftCamera.localPosition = new Vector3(-interaxialDistance / 2, 0, 0);
        rightCamera.localPosition = new Vector3(interaxialDistance / 2, 0, 0);
    }
}
