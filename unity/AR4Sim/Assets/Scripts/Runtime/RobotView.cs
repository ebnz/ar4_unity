using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace AR4Sim
{
    /// <summary>
    /// Physics-free copy of the robot's visuals, posed directly from joint values (viewer mode).
    ///
    /// The bridge uses it to show states computed by an external kinematic simulator. No
    /// Physics.Simulate is needed, so contacts, collision filtering and drive dynamics play no role:
    /// what Python sends is exactly what is shown. The articulated robot is hidden meanwhile.
    /// Link poses follow the ArticulationBody joint frames (parent anchor * joint motion * anchor^-1).
    /// </summary>
    public class RobotView : MonoBehaviour
    {
        class Link
        {
            public string Name;
            public Transform View, Parent;
            public ArticulationJointType Type;
            public Vector3 ParentAnchorPos, AnchorPos;
            public Quaternion ParentAnchorRot, AnchorRot;
        }

        readonly List<Link> links = new List<Link>();   // parents before children
        Renderer[] realRenderers;
        Transform viewEe;
        AR4Controller robot;

        public float[] Q { get; private set; } = new float[6];
        public float Gripper { get; private set; }
        public bool Active { get; private set; }

        public static RobotView Create(AR4Controller robot)
        {
            var view = new GameObject("AR4_View").AddComponent<RobotView>();
            view.Build(robot);
            view.qTo = new float[6];
            view.gTo = robot.GripperMax;
            view.SetActive(false);
            return view;
        }

        void Build(AR4Controller ar4)
        {
            robot = ar4;
            realRenderers = robot.GetComponentsInChildren<Renderer>(true).Where(r => r.enabled).ToArray();
            var bodies = robot.GetComponentsInChildren<ArticulationBody>(true);  // depth-first: parents first
            var viewOf = new Dictionary<ArticulationBody, Transform>();
            foreach (var b in bodies)
            {
                var go = new GameObject("view_" + b.name);
                var parentBody = b.isRoot ? null : b.transform.parent.GetComponentInParent<ArticulationBody>();
                go.transform.SetParent(parentBody != null ? viewOf[parentBody] : transform, false);
                go.transform.SetPositionAndRotation(b.transform.position, b.transform.rotation);
                viewOf[b] = go.transform;
                if (!b.isRoot)
                    links.Add(new Link
                    {
                        Name = b.name, View = go.transform, Parent = viewOf[parentBody], Type = b.jointType,
                        ParentAnchorPos = b.parentAnchorPosition, ParentAnchorRot = b.parentAnchorRotation,
                        AnchorPos = b.anchorPosition, AnchorRot = b.anchorRotation,
                    });
                if (b.name == "ee_link") viewEe = go.transform;
            }
            // Copy the visible meshes (not the colliders) onto the view links.
            foreach (var r in realRenderers.OfType<MeshRenderer>())
            {
                var mf = r.GetComponent<MeshFilter>();
                var owner = r.GetComponentInParent<ArticulationBody>();
                if (mf == null || mf.sharedMesh == null || owner == null) continue;
                var go = new GameObject(r.name);
                go.transform.SetParent(viewOf[owner], false);
                go.transform.SetPositionAndRotation(r.transform.position, r.transform.rotation);
                var ls = r.transform.lossyScale;
                go.transform.localScale = ls;   // view links have unit scale
                go.AddComponent<MeshFilter>().sharedMesh = mf.sharedMesh;
                go.AddComponent<MeshRenderer>().sharedMaterials = r.sharedMaterials;
            }
        }

        public void SetActive(bool on)
        {
            Active = on;
            gameObject.SetActive(on);
            foreach (var r in realRenderers) r.enabled = !on;
        }

        // Animation: the view moves from the pose shown when a new state arrives (from) to the new
        // state (to) within `duration` seconds, linearly in joint space. Q / Gripper are the target.
        float[] qFrom = new float[6], qTo = new float[6];
        float gFrom, gTo, t0, duration;

        float Alpha => duration <= 0 ? 1f : Mathf.Clamp01((Time.unscaledTime - t0) / duration);
        public bool Animating => Active && duration > 0 && Alpha < 1f;

        /// <summary>Set the target state: arm joints q [rad], jaw opening per side [m]. With
        /// duration > 0 the view moves there smoothly from the pose it shows now.</summary>
        public void SetState(float[] q, float gripper, float duration = 0f)
        {
            ShownPose(out var qNow, out var gNow);
            qFrom = qNow;
            gFrom = gNow;
            qTo = robot.Clamp(q);
            gTo = Mathf.Clamp(gripper, robot.GripperMin, robot.GripperMax);
            Q = (float[])qTo.Clone();
            Gripper = gTo;
            t0 = Time.unscaledTime;
            this.duration = Mathf.Max(0f, duration);
            if (this.duration <= 0f) Apply(qTo, gTo);
        }

        /// <summary>Jump to the end of the current animation.</summary>
        public void Finish()
        {
            duration = 0f;
            Apply(qTo, gTo);
        }

        void Update()
        {
            if (duration <= 0f) return;
            float a = Alpha;
            ShownPose(out var q, out var g);
            Apply(q, g);
            if (a >= 1f) duration = 0f;
        }

        void ShownPose(out float[] q, out float g)
        {
            float a = Alpha;
            q = new float[6];
            for (int i = 0; i < 6; i++) q[i] = Mathf.Lerp(qFrom[i], qTo[i], a);
            g = Mathf.Lerp(gFrom, gTo, a);
        }

        void Apply(float[] q, float gripper)
        {
            foreach (var l in links)
            {
                int arm = System.Array.IndexOf(ArmLinkNames, l.Name);
                Quaternion jr = Quaternion.identity;
                Vector3 jt = Vector3.zero;
                if (l.Type == ArticulationJointType.RevoluteJoint && arm >= 0)
                    jr = Quaternion.AngleAxis(q[arm] * Mathf.Rad2Deg, Vector3.right);
                else if (l.Type == ArticulationJointType.PrismaticJoint)
                    jt = Vector3.right * gripper;
                // child pose in parent = P * J * A^-1, with P = parent anchor, A = anchor (in child)
                Quaternion rot = l.ParentAnchorRot * jr * Quaternion.Inverse(l.AnchorRot);
                Vector3 pos = l.ParentAnchorPos + l.ParentAnchorRot * jt - rot * l.AnchorPos;
                l.View.localPosition = pos;
                l.View.localRotation = rot;
            }
        }

        /// <summary>Evaluate f with the view posed at the target state (restores the shown pose).</summary>
        T AtTarget<T>(System.Func<T> f)
        {
            if (duration <= 0f) return f();
            ShownPose(out var q, out var g);
            Apply(qTo, gTo);
            var result = f();
            Apply(q, g);
            return result;
        }

        /// <summary>The view's ee_link (moves with the shown pose).</summary>
        public Transform Ee => viewEe;

        /// <summary>World pose of ee_link at the target state.</summary>
        public (Vector3 pos, Quaternion rot) TargetEeWorld => AtTarget(() => (viewEe.position, viewEe.rotation));

        static readonly string[] ArmLinkNames = { "link_1", "link_2", "link_3", "link_4", "link_5", "link_6" };

        /// <summary>Flange pose of the target state (in base_link).</summary>
        public (Vector3 pos, Quaternion rot) FlangePose => AtTarget(() => robot.PoseInBase(viewEe.position, viewEe.rotation));

        /// <summary>TCP pose of the target state (in base_link).</summary>
        public (Vector3 pos, Quaternion rot) TcpPose => AtTarget(() =>
        {
            Vector3 zWorld = viewEe.rotation * new Vector3(0, 0, 1).Ros2Unity();
            return robot.PoseInBase(viewEe.position + zWorld * robot.tcpOffset, viewEe.rotation);
        });
    }
}
