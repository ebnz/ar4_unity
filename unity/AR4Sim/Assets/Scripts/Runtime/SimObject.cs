using UnityEngine;

namespace AR4Sim
{
    /// <summary>A rigid body the bridge exposes by name (get_objects / set_object_pose / reset).</summary>
    [RequireComponent(typeof(Rigidbody))]
    public class SimObject : MonoBehaviour
    {
        Vector3 initialPos;
        Quaternion initialRot;
        public Rigidbody Body { get; private set; }

        void Awake()
        {
            Body = GetComponent<Rigidbody>();
            initialPos = transform.position;
            initialRot = transform.rotation;
        }

        public void SetWorldPose(Vector3 pos, Quaternion rot)
        {
            Body.position = pos;
            Body.rotation = rot;
            transform.SetPositionAndRotation(pos, rot);
            Body.linearVelocity = Vector3.zero;
            Body.angularVelocity = Vector3.zero;
        }

        public void ResetPose() => SetWorldPose(initialPos, initialRot);
    }
}
