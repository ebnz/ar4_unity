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
            if (!Body.isKinematic)
            {
                Body.linearVelocity = Vector3.zero;
                Body.angularVelocity = Vector3.zero;
            }
        }

        public void ResetPose() => SetWorldPose(initialPos, initialRot);

        /// <summary>Kinematic objects are only moved by SetWorldPose (viewer mode); others by physics.</summary>
        public void SetKinematic(bool kinematic) => Body.isKinematic = kinematic;

        // Viewer mode: smooth motion to a target pose within `duration` seconds. Either linear/slerp
        // (free cube) or rigidly attached to a moving transform (cube held in the gripper).
        Vector3 fromPos, toPos, relPos;
        Quaternion fromRot, toRot, relRot;
        Transform follow;
        float t0, duration;

        public bool Animating => duration > 0f;
        public (Vector3 pos, Quaternion rot) TargetPose => duration > 0f ? (toPos, toRot) : (transform.position, transform.rotation);

        /// <summary>Move to the world pose (pos, rot) within duration seconds. If follow is given, the
        /// object keeps its target pose relative to follow's target pose (followPos, followRot) meanwhile.</summary>
        public void AnimateTo(Vector3 pos, Quaternion rot, float duration, Transform follow = null,
                              Vector3 followPos = default, Quaternion followRot = default)
        {
            fromPos = transform.position;
            fromRot = transform.rotation;
            toPos = pos;
            toRot = rot;
            this.follow = follow;
            if (follow != null)
            {
                relRot = Quaternion.Inverse(followRot) * rot;
                relPos = Quaternion.Inverse(followRot) * (pos - followPos);
            }
            t0 = Time.unscaledTime;
            this.duration = Mathf.Max(0f, duration);
            if (this.duration <= 0f) Finish();
        }

        public void Finish()
        {
            duration = 0f;
            follow = null;
            SetWorldPose(toPos, toRot);
        }

        void LateUpdate()   // after RobotView.Update, so a held cube follows the gripper of this frame
        {
            if (duration <= 0f) return;
            float a = Mathf.Clamp01((Time.unscaledTime - t0) / duration);
            if (a >= 1f) { Finish(); return; }
            if (follow != null)
                SetWorldPose(follow.TransformPoint(Vector3.zero) + follow.rotation * relPos, follow.rotation * relRot);
            else
                SetWorldPose(Vector3.Lerp(fromPos, toPos, a), Quaternion.Slerp(fromRot, toRot, a));
        }

        public void SetVisible(bool visible)
        {
            foreach (var r in GetComponentsInChildren<Renderer>()) r.enabled = visible;
        }
    }
}
