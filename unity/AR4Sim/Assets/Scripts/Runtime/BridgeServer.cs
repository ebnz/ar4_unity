using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Newtonsoft.Json.Linq;

using UnityEngine;
using Debug = UnityEngine.Debug;

namespace AR4Sim
{
    /// <summary>
    /// Lockstep TCP bridge. One JSON object per line in each direction:
    ///   -> {"id": 1, "cmd": "step", "n": 10}
    ///   <- {"id": 1, "ok": true, "state": {...}}
    /// Physics only advances on step / step_until, so the simulation waits for the client.
    /// Socket I/O runs on background threads; commands execute on the Unity main thread.
    /// </summary>
    public class BridgeServer : MonoBehaviour
    {
        public const string ProtocolVersion = "1.0";

        public AR4Controller robot;
        public int port = 5005;
        [Tooltip("Physics time step [s]")]
        public float dt = 0.01f;
        [Tooltip("Max wall time per frame spent serving requests [ms]")]
        public float serviceBudgetMs = 12f;

        class Client
        {
            public TcpClient Tcp;
            public StreamWriter Writer;
            public readonly object Lock = new object();

            public void Send(string line)
            {
                lock (Lock)
                {
                    try { Writer.Write(line); Writer.Write('\n'); Writer.Flush(); }
                    catch (Exception) { /* client went away; reader thread cleans up */ }
                }
            }
        }

        readonly BlockingCollection<(Client client, string line)> inbox = new BlockingCollection<(Client, string)>();
        TcpListener listener;
        Thread acceptThread;
        volatile bool running;
        long stepCount;
        SimObject[] objects;

        void Awake()
        {
            ParseCommandLine();
            Physics.simulationMode = SimulationMode.Script;
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = 60;
            Application.runInBackground = true;
        }

        void Start()
        {
            if (robot == null) robot = FindAnyObjectByType<AR4Controller>();
            objects = FindObjectsByType<SimObject>(FindObjectsSortMode.InstanceID).OrderBy(o => o.name).ToArray();
            // Let the scene settle once so objects rest on the ground before the first client connects.
            for (int i = 0; i < 50; i++) robot.Step(dt);
            stepCount = 0;

            listener = new TcpListener(IPAddress.Loopback, port);
            listener.Start();
            running = true;
            acceptThread = new Thread(AcceptLoop) { IsBackground = true, Name = "BridgeAccept" };
            acceptThread.Start();
            Debug.Log($"[AR4 bridge] listening on 127.0.0.1:{port}, dt={dt}s (lockstep)");
        }

        void ParseCommandLine()
        {
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == "-port") port = int.Parse(args[i + 1]);
                if (args[i] == "-dt") dt = float.Parse(args[i + 1], System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        void AcceptLoop()
        {
            while (running)
            {
                TcpClient tcp;
                try { tcp = listener.AcceptTcpClient(); }
                catch (Exception) { break; }
                tcp.NoDelay = true;
                var stream = tcp.GetStream();
                var client = new Client { Tcp = tcp, Writer = new StreamWriter(stream, new UTF8Encoding(false)) };
                new Thread(() => ReadLoop(client, stream)) { IsBackground = true, Name = "BridgeClient" }.Start();
            }
        }

        void ReadLoop(Client client, NetworkStream stream)
        {
            Debug.Log($"[AR4 bridge] client connected: {client.Tcp.Client.RemoteEndPoint}");
            try
            {
                using var reader = new StreamReader(stream, Encoding.UTF8);
                string line;
                while (running && (line = reader.ReadLine()) != null)
                    if (line.Length > 0) inbox.Add((client, line));
            }
            catch (Exception) { }
            client.Tcp.Close();
            Debug.Log("[AR4 bridge] client disconnected");
        }

        void Update()
        {
            var budget = Stopwatch.StartNew();
            while (budget.ElapsedMilliseconds < serviceBudgetMs)
            {
                int wait = Math.Max(0, (int)(serviceBudgetMs - budget.ElapsedMilliseconds));
                if (!inbox.TryTake(out var msg, wait)) break;
                msg.client.Send(Handle(msg.line));
            }
        }

        void OnDestroy()
        {
            running = false;
            listener?.Stop();
        }

        string Handle(string line)
        {
            JToken id = null;
            try
            {
                var req = JObject.Parse(line);
                id = req["id"];
                var reply = Dispatch((string)req["cmd"] ?? throw new ArgumentException("missing 'cmd'"), req);
                reply["id"] = id;
                reply["ok"] = true;
                return reply.ToString(Newtonsoft.Json.Formatting.None);
            }
            catch (Exception e)
            {
                return new JObject { ["id"] = id, ["ok"] = false, ["error"] = e.Message }.ToString(Newtonsoft.Json.Formatting.None);
            }
        }

        JObject Dispatch(string cmd, JObject req)
        {
            switch (cmd)
            {
                case "ping":
                    return new JObject { ["version"] = ProtocolVersion };

                case "get_info":
                    return new JObject
                    {
                        ["version"] = ProtocolVersion,
                        ["dt"] = dt,
                        ["joint_names"] = new JArray(AR4Controller.ArmJointNames),
                        ["lower"] = new JArray(robot.Lower),
                        ["upper"] = new JArray(robot.Upper),
                        ["max_velocity"] = new JArray(robot.MaxVelocity),
                        ["has_gripper"] = robot.HasGripper,
                        ["gripper_range"] = new JArray(robot.GripperMin, robot.GripperMax),
                        ["tcp_offset"] = robot.tcpOffset,
                        ["objects"] = new JArray(objects.Select(o => o.name)),
                    };

                case "get_state":
                    return new JObject { ["state"] = State() };

                case "set_joint_targets":
                    robot.SetJointTargets(FloatArray(req, "q", 6), (float?)req["speed"] ?? 1f);
                    return new JObject { ["goal"] = new JArray(robot.Goal) };

                case "set_gripper":
                    robot.SetGripper(Required<float>(req, "position"));
                    return new JObject { ["goal"] = robot.GripperGoal };

                case "step":
                {
                    int n = (int?)req["n"] ?? 1;
                    if (n < 0) throw new ArgumentException("n must be >= 0");
                    for (int i = 0; i < n; i++) DoStep();
                    return new JObject { ["state"] = State() };
                }

                case "step_until":
                {
                    // Step until the arm (and gripper) reached their goals, or max_steps elapsed.
                    int maxSteps = (int?)req["max_steps"] ?? 1000;
                    float tol = (float?)req["tol"] ?? 0.002f;
                    int settle = (int?)req["settle_steps"] ?? 5;
                    int steps = 0, settled = 0;
                    while (steps < maxSteps && settled < settle)
                    {
                        DoStep();
                        steps++;
                        settled = robot.AtGoal(tol) && robot.GripperAtGoal ? settled + 1 : 0;
                    }
                    return new JObject { ["reached"] = settled >= settle, ["steps"] = steps, ["state"] = State() };
                }

                case "reset":
                {
                    var q = req["q"] != null ? FloatArray(req, "q", 6) : new float[6];
                    float g = (float?)req["gripper"] ?? robot.GripperMax;
                    robot.Reset(q, g);
                    if ((bool?)req["reset_objects"] ?? true)
                        foreach (var o in objects) o.ResetPose();
                    stepCount = 0;
                    // One step so transforms and contacts reflect the teleported state.
                    robot.Step(dt);
                    return new JObject { ["state"] = State() };
                }

                case "get_objects":
                    return new JObject { ["objects"] = Objects() };

                case "set_object_pose":
                {
                    string name = Required<string>(req, "name");
                    var obj = objects.FirstOrDefault(o => o.name == name) ?? throw new ArgumentException($"unknown object '{name}'");
                    var p = FloatArray(req, "position", 3);
                    var q = req["orientation"] != null ? FloatArray(req, "orientation", 4) : new[] { 0f, 0f, 0f, 1f };
                    var baseT = robot.BaseLink;
                    obj.SetWorldPose(baseT.TransformPoint(new Vector3(p[0], p[1], p[2]).Ros2Unity()),
                                     baseT.rotation * new Quaternion(q[0], q[1], q[2], q[3]).Ros2Unity());
                    return new JObject();
                }

                case "set_params":
                {
                    // Tune the controller at runtime; returns the current values.
                    var fields = new (string key, Func<float> get, Action<float> set)[]
                    {
                        ("arm_stiffness", () => robot.armStiffness, v => robot.armStiffness = v),
                        ("arm_damping", () => robot.armDamping, v => robot.armDamping = v),
                        ("arm_force_limit", () => robot.armForceLimit, v => robot.armForceLimit = v),
                        ("gripper_stiffness", () => robot.gripperStiffness, v => robot.gripperStiffness = v),
                        ("gripper_damping", () => robot.gripperDamping, v => robot.gripperDamping = v),
                        ("gripper_force_limit", () => robot.gripperForceLimit, v => robot.gripperForceLimit = v),
                        ("gripper_max_velocity", () => robot.gripperMaxVelocity, v => robot.gripperMaxVelocity = v),
                    };
                    var known = fields.Select(f => f.key).ToHashSet();
                    foreach (var prop in req.Properties())
                        if (prop.Name != "id" && prop.Name != "cmd" && !known.Contains(prop.Name))
                            throw new ArgumentException($"unknown param '{prop.Name}' (known: {string.Join(", ", known)})");
                    foreach (var f in fields)
                        if (req[f.key] != null) f.set((float)req[f.key]);
                    robot.ApplyDriveParams();
                    var reply = new JObject();
                    foreach (var f in fields) reply[f.key] = f.get();
                    return new JObject { ["params"] = reply };
                }

                case "get_bodies":
                {
                    // Diagnostics: mass properties and drive settings of every articulation link.
                    var arr = new JArray();
                    foreach (var b in robot.GetComponentsInChildren<ArticulationBody>())
                        arr.Add(new JObject
                        {
                            ["name"] = b.name,
                            ["joint_type"] = b.jointType.ToString(),
                            ["mass"] = b.mass,
                            ["inertia"] = new JArray(b.inertiaTensor.x, b.inertiaTensor.y, b.inertiaTensor.z),
                            ["auto_inertia"] = b.automaticInertiaTensor,
                            ["stiffness"] = b.xDrive.stiffness,
                            ["damping"] = b.xDrive.damping,
                            ["force_limit"] = b.xDrive.forceLimit,
                            ["joint_friction"] = b.jointFriction,
                            ["angular_damping"] = b.angularDamping,
                            ["colliders"] = b.GetComponentsInChildren<Collider>().Count(c => c.attachedArticulationBody == b),
                        });
                    return new JObject { ["bodies"] = arr };
                }

                case "get_image":
                    return new JObject { ["png"] = Convert.ToBase64String(RenderPng(req)) };

                default:
                    throw new ArgumentException($"unknown cmd '{cmd}'");
            }
        }

        /// <summary>Render the main camera, or a temporary view from eye -> target (ROS base frame).</summary>
        byte[] RenderPng(JObject req)
        {
            if (SystemInfo.graphicsDeviceType == UnityEngine.Rendering.GraphicsDeviceType.Null)
                throw new InvalidOperationException("no graphics device (started with -nographics)");
            int w = (int?)req["width"] ?? 640, h = (int?)req["height"] ?? 480;
            var main = Camera.main;
            var cam = main;
            if (req["eye"] != null)
            {
                cam = new GameObject("BridgeCapture").AddComponent<Camera>();
                cam.CopyFrom(main);
                var eye = FloatArray(req, "eye", 3);
                var target = req["target"] != null ? FloatArray(req, "target", 3) : new[] { 0f, -0.3f, 0.1f };
                var baseT = robot.BaseLink;
                cam.transform.position = baseT.TransformPoint(new Vector3(eye[0], eye[1], eye[2]).Ros2Unity());
                cam.transform.LookAt(baseT.TransformPoint(new Vector3(target[0], target[1], target[2]).Ros2Unity()), baseT.up);
                if (req["fov"] != null) cam.fieldOfView = (float)req["fov"];
            }
            var rt = RenderTexture.GetTemporary(w, h, 24, RenderTextureFormat.ARGB32);
            var prevTarget = cam.targetTexture;
            var prevActive = RenderTexture.active;
            try
            {
                cam.targetTexture = rt;
                cam.Render();
                RenderTexture.active = rt;
                var tex = new Texture2D(w, h, TextureFormat.RGB24, false);
                tex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
                tex.Apply();
                var png = tex.EncodeToPNG();
                Destroy(tex);
                return png;
            }
            finally
            {
                cam.targetTexture = prevTarget;
                RenderTexture.active = prevActive;
                RenderTexture.ReleaseTemporary(rt);
                if (cam != main) Destroy(cam.gameObject);
            }
        }

        void DoStep()
        {
            robot.Step(dt);
            stepCount++;
        }

        JObject State()
        {
            var flange = robot.FlangePose;
            var tcp = robot.TcpPose;
            return new JObject
            {
                ["step"] = stepCount,
                ["time"] = stepCount * (double)dt,
                ["q"] = new JArray(robot.JointPositions),
                ["dq"] = new JArray(robot.JointVelocities),
                ["effort"] = new JArray(robot.JointEfforts),
                ["goal"] = new JArray(robot.Goal),
                ["at_goal"] = robot.AtGoal(0.002f),
                ["gripper"] = robot.GripperPosition,
                ["gripper_goal"] = robot.GripperGoal,
                ["flange_pose"] = Pose(flange.pos, flange.rot),
                ["tcp_pose"] = Pose(tcp.pos, tcp.rot),
            };
        }

        JArray Objects()
        {
            var arr = new JArray();
            foreach (var o in objects)
            {
                var (p, r) = robot.PoseInBase(o.transform.position, o.transform.rotation);
                var jo = Pose(p, r);
                jo["name"] = o.name;
                arr.Add(jo);
            }
            return arr;
        }

        static JObject Pose(Vector3 p, Quaternion r) => new JObject
        {
            ["position"] = new JArray(p.x, p.y, p.z),
            ["orientation"] = new JArray(r.x, r.y, r.z, r.w), // xyzw
        };

        static T Required<T>(JObject req, string key) =>
            req[key] != null ? req[key].ToObject<T>() : throw new ArgumentException($"missing '{key}'");

        static float[] FloatArray(JObject req, string key, int len)
        {
            var a = Required<float[]>(req, key);
            if (a.Length != len) throw new ArgumentException($"'{key}' must have {len} elements");
            return a;
        }
    }
}
