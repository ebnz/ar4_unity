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

        /// <summary>Pose the view: arm joints q [rad], jaw opening per side [m].</summary>
        public void SetState(float[] q, float gripper)
        {
            Q = robot.Clamp(q);
            Gripper = Mathf.Clamp(gripper, robot.GripperMin, robot.GripperMax);
            foreach (var l in links)
            {
                int arm = System.Array.IndexOf(ArmLinkNames, l.Name);
                Quaternion jr = Quaternion.identity;
                Vector3 jt = Vector3.zero;
                if (l.Type == ArticulationJointType.RevoluteJoint && arm >= 0)
                    jr = Quaternion.AngleAxis(Q[arm] * Mathf.Rad2Deg, Vector3.right);
                else if (l.Type == ArticulationJointType.PrismaticJoint)
                    jt = Vector3.right * Gripper;
                // child pose in parent = P * J * A^-1, with P = parent anchor, A = anchor (in child)
                Quaternion rot = l.ParentAnchorRot * jr * Quaternion.Inverse(l.AnchorRot);
                Vector3 pos = l.ParentAnchorPos + l.ParentAnchorRot * jt - rot * l.AnchorPos;
                l.View.localPosition = pos;
                l.View.localRotation = rot;
            }
        }

        static readonly string[] ArmLinkNames = { "link_1", "link_2", "link_3", "link_4", "link_5", "link_6" };

        public (Vector3 pos, Quaternion rot) FlangePose => robot.PoseInBase(viewEe.position, viewEe.rotation);

        public (Vector3 pos, Quaternion rot) TcpPose
        {
            get
            {
                Vector3 zWorld = viewEe.rotation * new Vector3(0, 0, 1).Ros2Unity();
                return robot.PoseInBase(viewEe.position + zWorld * robot.tcpOffset, viewEe.rotation);
            }
        }
    }
}
