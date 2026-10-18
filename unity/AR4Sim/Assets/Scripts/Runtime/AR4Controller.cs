using System;
using System.Collections.Generic;
using System.Linq;

using UnityEngine;

namespace AR4Sim
{
    /// <summary>
    /// Joint-level controller for the imported AR4 ArticulationBody chain.
    /// All public quantities use URDF/ROS conventions: radians, meters, right-handed
    /// frames expressed in base_link. Physics is advanced only via <see cref="Step"/>.
    /// </summary>
    public class AR4Controller : MonoBehaviour
    {
        public static readonly string[] ArmJointNames = { "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6" };
        static readonly string[] ArmLinkNames = { "link_1", "link_2", "link_3", "link_4", "link_5", "link_6" };
        const string Jaw1Link = "gripper_jaw1_link", Jaw2Link = "gripper_jaw2_link";

        [Header("Arm drives")]
        public float armStiffness = 200000f;
        public float armDamping = 2000f;
        public float armForceLimit = 1000f;

        [Tooltip("Joint velocity limits [rad/s]; the scene builder copies them from the URDF")]
        public float[] maxJointVelocity = { 1.0472f, 1.0472f, 1.0472f, 1.0472f, 1.0472f, 1.0472f };

        [Header("Gripper drives")]
        public float gripperStiffness = 10000f;
        public float gripperDamping = 100f;
        [Tooltip("Squeeze force per jaw [N]. Above ~10 N the jaws push through light objects (PhysX contact limits).")]
        public float gripperForceLimit = 5f;
        [Tooltip("Jaw speed [m/s]")]
        public float gripperMaxVelocity = 0.03f;

        [Tooltip("TCP offset from ee_link along its z axis [m] (centre of the finger pads)")]
        public float tcpOffset = 0.055f;

        ArticulationBody root;
        Transform baseLink, eeLink;
        ArticulationBody[] arm;
        ArticulationBody jaw1, jaw2;

        public float[] Lower { get; private set; }
        public float[] Upper { get; private set; }
        public float[] MaxVelocity { get; private set; }
        public float GripperMin { get; private set; }
        public float GripperMax { get; private set; }

        // Commanded goal, and the velocity-limited setpoint that is actually sent to the drives.
        float[] goal, setpoint;
        float speedScale = 1f;
        float gripperGoal, gripperSetpoint;

        public bool HasGripper => jaw1 != null;

        void Awake()
        {
            var bodies = GetComponentsInChildren<ArticulationBody>(true);
            ArticulationBody Find(string n) => bodies.FirstOrDefault(b => b.name == n);

            root = bodies.First(b => b.isRoot);
            root.immovable = true;
            baseLink = Find("base_link").transform;
            eeLink = Find("ee_link").transform;
            arm = ArmLinkNames.Select(n => Find(n) ?? throw new Exception($"link {n} not found")).ToArray();
            jaw1 = Find(Jaw1Link);
            jaw2 = Find(Jaw2Link);

            // The importer's keyboard controller would fight us for the drives (the scene builder
            // already removes it; this covers hand-imported robots).
            foreach (var c in GetComponentsInChildren<Unity.Robotics.UrdfImporter.Control.Controller>(true))
                DestroyImmediate(c);

            Lower = arm.Select(b => b.xDrive.lowerLimit * Mathf.Deg2Rad).ToArray();
            Upper = arm.Select(b => b.xDrive.upperLimit * Mathf.Deg2Rad).ToArray();
            MaxVelocity = (float[])maxJointVelocity.Clone();
            foreach (var b in arm)
            {
                // The importer puts the URDF joint velocity limit here, but PhysX applies it to the
                // body's world angular velocity. We limit the setpoint instead (see Step).
                b.maxAngularVelocity = 50f;
                b.maxJointVelocity = 50f;
                ConfigureDrive(b, armStiffness, armDamping, armForceLimit);
            }

            if (HasGripper)
            {
                GripperMin = jaw1.xDrive.lowerLimit;
                GripperMax = jaw1.xDrive.upperLimit;
                ConfigureDrive(jaw1, gripperStiffness, gripperDamping, gripperForceLimit);
                ConfigureDrive(jaw2, gripperStiffness, gripperDamping, gripperForceLimit);
                // Help the fingers hold on to things.
                var grip = new PhysicsMaterial("ar4_finger") { staticFriction = 1.2f, dynamicFriction = 1.0f, frictionCombine = PhysicsMaterialCombine.Maximum };
                foreach (var c in jaw1.GetComponentsInChildren<Collider>().Concat(jaw2.GetComponentsInChildren<Collider>()))
                    c.sharedMaterial = grip;
            }

            // URDF links without <inertial> (ee_link) get Unity's defaults: 1 kg and 1 kg m^2,
            // a few hundred times the real wrist. Make them effectively massless instead.
            foreach (var b in bodies)
                if (!b.isRoot && b.automaticInertiaTensor && !b.GetComponentsInChildren<Collider>().Any(c => c.attachedArticulationBody == b))
                {
                    b.mass = 1e-3f;
                    b.automaticInertiaTensor = false;
                    b.inertiaTensor = Vector3.one * 1e-7f;
                    b.inertiaTensorRotation = Quaternion.identity;
                }

            foreach (var b in bodies) b.solverIterations = 32;
            IgnoreSelfCollisions();

            goal = new float[6];
            setpoint = new float[6];
            Reset(new float[6], HasGripper ? GripperMax : 0f);
        }

        static void ConfigureDrive(ArticulationBody b, float k, float d, float f)
        {
            var drive = b.xDrive;
            drive.stiffness = k;
            drive.damping = d;
            drive.forceLimit = f;
            drive.driveType = ArticulationDriveType.Force;
            b.xDrive = drive;
        }

        // A stepper-driven arm doesn't care about its own convex hulls overlapping;
        // only the outside world should collide with the robot.
        void IgnoreSelfCollisions()
        {
            var cols = GetComponentsInChildren<Collider>(true);
            for (int i = 0; i < cols.Length; i++)
                for (int j = i + 1; j < cols.Length; j++)
                    Physics.IgnoreCollision(cols[i], cols[j]);
        }

        /// <summary>Re-apply drive gains after changing the public fields at runtime.</summary>
        public void ApplyDriveParams()
        {
            foreach (var b in arm) ConfigureDrive(b, armStiffness, armDamping, armForceLimit);
            if (HasGripper)
            {
                ConfigureDrive(jaw1, gripperStiffness, gripperDamping, gripperForceLimit);
                ConfigureDrive(jaw2, gripperStiffness, gripperDamping, gripperForceLimit);
            }
        }

        public float[] Clamp(float[] q) => q.Select((v, i) => Mathf.Clamp(v, Lower[i], Upper[i])).ToArray();

        public void SetJointTargets(float[] q, float speed)
        {
            if (q.Length != 6) throw new ArgumentException("q must have 6 elements");
            goal = Clamp(q);
            speedScale = Mathf.Clamp(speed, 1e-3f, 1f);
        }

        public void SetGripper(float position)
        {
            if (!HasGripper) throw new InvalidOperationException("robot has no gripper");
            gripperGoal = Mathf.Clamp(position, GripperMin, GripperMax);
        }

        /// <summary>Teleport the robot to q (no dynamics) and hold it there.</summary>
        public void Reset(float[] q, float gripper)
        {
            q = Clamp(q);
            Array.Copy(q, goal, 6);
            Array.Copy(q, setpoint, 6);
            for (int i = 0; i < 6; i++)
            {
                arm[i].jointPosition = new ArticulationReducedSpace(q[i]);
                arm[i].jointVelocity = new ArticulationReducedSpace(0f);
                SetDriveTarget(arm[i], q[i] * Mathf.Rad2Deg, 0f);
            }
            if (HasGripper)
            {
                gripperGoal = gripperSetpoint = Mathf.Clamp(gripper, GripperMin, GripperMax);
                foreach (var j in new[] { jaw1, jaw2 })
                {
                    j.jointPosition = new ArticulationReducedSpace(gripperSetpoint);
                    j.jointVelocity = new ArticulationReducedSpace(0f);
                    SetDriveTarget(j, gripperSetpoint, 0f);
                }
            }
        }

        static void SetDriveTarget(ArticulationBody b, float target, float targetVelocity)
        {
            var drive = b.xDrive;
            drive.target = target;
            drive.targetVelocity = targetVelocity;
            b.xDrive = drive;
        }

        /// <summary>Advance the setpoints by dt (velocity-limited) and simulate one physics step.</summary>
        public void Step(float dt)
        {
            for (int i = 0; i < 6; i++)
            {
                float vmax = MaxVelocity[i] * speedScale;
                float delta = Mathf.Clamp(goal[i] - setpoint[i], -vmax * dt, vmax * dt);
                setpoint[i] += delta;
                SetDriveTarget(arm[i], setpoint[i] * Mathf.Rad2Deg, delta / dt * Mathf.Rad2Deg);
            }
            if (HasGripper)
            {
                float delta = Mathf.Clamp(gripperGoal - gripperSetpoint, -gripperMaxVelocity * dt, gripperMaxVelocity * dt);
                gripperSetpoint += delta;
                // Mimic joint: Unity articulations have no mimic support, so drive both jaws.
                SetDriveTarget(jaw1, gripperSetpoint, delta / dt);
                SetDriveTarget(jaw2, gripperSetpoint, delta / dt);
            }
            Physics.Simulate(dt);
        }

        public float[] JointPositions => arm.Select(b => b.jointPosition[0]).ToArray();
        public float[] JointVelocities => arm.Select(b => b.jointVelocity[0]).ToArray();
        public float[] JointEfforts => arm.Select(b => b.driveForce.dofCount > 0 ? b.driveForce[0] : 0f).ToArray();
        public float[] Goal => (float[])goal.Clone();
        public float GripperPosition => HasGripper ? 0.5f * (jaw1.jointPosition[0] + jaw2.jointPosition[0]) : 0f;
        public float GripperGoal => gripperGoal;

        /// <summary>True when the setpoint reached the goal and the joints settled within tol [rad].</summary>
        public bool AtGoal(float tol)
        {
            var q = JointPositions;
            for (int i = 0; i < 6; i++)
                if (Mathf.Abs(setpoint[i] - goal[i]) > 1e-6f || Mathf.Abs(q[i] - goal[i]) > tol) return false;
            return true;
        }

        public bool GripperAtGoal => !HasGripper || Mathf.Abs(gripperSetpoint - gripperGoal) < 1e-6f;

        /// <summary>Pose of a Unity transform expressed in the ROS base_link frame.</summary>
        public (Vector3 pos, Quaternion rot) PoseInBase(Vector3 worldPos, Quaternion worldRot)
        {
            Vector3 p = baseLink.InverseTransformPoint(worldPos).Unity2Ros();
            Quaternion r = (Quaternion.Inverse(baseLink.rotation) * worldRot).Unity2Ros();
            return (p, r);
        }

        public (Vector3 pos, Quaternion rot) FlangePose => PoseInBase(eeLink.position, eeLink.rotation);

        public (Vector3 pos, Quaternion rot) TcpPose
        {
            get
            {
                // ee_link's ROS +z axis expressed as a Unity world direction.
                Vector3 zWorld = eeLink.rotation * new Vector3(0, 0, 1).Ros2Unity();
                return PoseInBase(eeLink.position + zWorld * tcpOffset, eeLink.rotation);
            }
        }

        public Transform BaseLink => baseLink;
    }
}
