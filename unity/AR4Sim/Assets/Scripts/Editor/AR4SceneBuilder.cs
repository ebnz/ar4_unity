using System.IO;
using System.Linq;
using Unity.Robotics.UrdfImporter;
using AR4Sim;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace AR4Sim.EditorTools
{
    /// <summary>
    /// Builds Assets/Scenes/AR4.unity from Assets/URDF/ar4/ar4.urdf and the Linux player.
    /// Headless use:
    ///   Unity -batchmode -nographics -quit -projectPath unity/AR4Sim -executeMethod AR4Sim.EditorTools.AR4SceneBuilder.BuildScene
    ///   Unity -batchmode -nographics -quit -projectPath unity/AR4Sim -executeMethod AR4Sim.EditorTools.AR4SceneBuilder.BuildPlayer
    /// </summary>
    public static class AR4SceneBuilder
    {
        const string UrdfPath = "Assets/URDF/ar4/ar4.urdf";
        const string ScenePath = "Assets/Scenes/AR4.unity";
        const string PlayerPath = "../Build/AR4Sim.x86_64";               // Linux
        const string WindowsPlayerPath = "../Build/Windows/AR4Sim.exe";
        const string MacPlayerPath = "../Build/macOS/AR4Sim.app";

        [MenuItem("AR4/Build Scene from URDF")]
        public static void BuildScene()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

            var settings = new ImportSettings
            {
                chosenAxis = ImportSettings.axisType.yAxis,
                convexMethod = ImportSettings.convexDecomposer.vHACD,
                OverwriteExistingPrefabs = true,
            };
            GameObject robot = null;
            var it = UrdfRobotExtensions.Create(Path.GetFullPath(UrdfPath), settings);
            while (it.MoveNext()) robot = it.Current ?? robot;
            if (robot == null) throw new System.Exception($"URDF import failed: {UrdfPath}");
            robot.name = "AR4";
            Object.DestroyImmediate(robot.GetComponent<Unity.Robotics.UrdfImporter.Control.Controller>());
            var controller = robot.AddComponent<AR4Controller>();
            var urdf = new Robot(Path.GetFullPath(UrdfPath));
            controller.maxJointVelocity = AR4Controller.ArmJointNames
                .Select(n => (float)urdf.joints.First(j => j.name == n).limit.velocity).ToArray();

            // Table: top face is the plane z = 0 of base_link (the robot stands on it). A floor below.
            // Keep in sync with rmi/ar4.py (H_TABLE = 0).
            var tableMat = SaveMaterial("Table", new Color(0.62f, 0.55f, 0.45f));
            var legMat = SaveMaterial("TableLeg", new Color(0.3f, 0.3f, 0.32f));
            RosBox("Table", new Vector3(0.0f, -0.15f, -0.02f), new Vector3(1.2f, 1.0f, 0.04f), tableMat);
            foreach (var (x, y) in new[] { (-0.55f, -0.6f), (0.55f, -0.6f), (-0.55f, 0.3f), (0.55f, 0.3f) })
                RosBox("TableLeg", new Vector3(x, y, -0.4f), new Vector3(0.05f, 0.05f, 0.72f), legMat);
            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Floor";
            ground.transform.localScale = new Vector3(0.6f, 1, 0.6f); // 6 m x 6 m
            ground.transform.position = new Vector3(0, -0.76f, 0);
            ground.GetComponent<Renderer>().sharedMaterial = SaveMaterial("Ground", new Color(0.35f, 0.37f, 0.4f));

            // Open bin (no physics role in the tasks, but it collides in physics mode).
            // Keep in sync with rmi/ar4.py: BIN_POS, BIN_INNER_SIZE, BIN_WALL_HEIGHT, BIN_WALL_THICKNESS, BIN_FLOOR_THICKNESS.
            {
                var binMat = SaveMaterial("Bin", new Color(0.9f, 0.9f, 0.88f));
                Vector3 c = new Vector3(0.25f, -0.26f, 0f);
                float ix = 0.12f, iy = 0.12f, h = 0.06f, t = 0.005f, fl = 0.003f;
                var bin = new GameObject("Bin");
                RosBox("BinFloor", c + new Vector3(0, 0, fl / 2), new Vector3(ix + 2 * t, iy + 2 * t, fl), binMat, bin.transform);
                RosBox("BinWallXp", c + new Vector3(ix / 2 + t / 2, 0, h / 2), new Vector3(t, iy + 2 * t, h), binMat, bin.transform);
                RosBox("BinWallXn", c + new Vector3(-ix / 2 - t / 2, 0, h / 2), new Vector3(t, iy + 2 * t, h), binMat, bin.transform);
                RosBox("BinWallYp", c + new Vector3(0, iy / 2 + t / 2, h / 2), new Vector3(ix, t, h), binMat, bin.transform);
                RosBox("BinWallYn", c + new Vector3(0, -iy / 2 - t / 2, h / 2), new Vector3(ix, t, h), binMat, bin.transform);
            }

            // The robot faces ROS -Y, which is Unity +X. Place cubes in its workspace.
            var colors = new[] { new Color(0.85f, 0.2f, 0.2f), new Color(0.2f, 0.6f, 0.25f), new Color(0.2f, 0.35f, 0.85f) };
            var rosPositions = new[] { new Vector3(0.0f, -0.30f, 0.01f), new Vector3(0.10f, -0.28f, 0.01f), new Vector3(-0.10f, -0.28f, 0.01f) };
            var cubeMat = new PhysicsMaterial("cube") { staticFriction = 1.0f, dynamicFriction = 0.8f };
            AssetDatabase.CreateAsset(cubeMat, "Assets/Materials/Cube.physicMaterial");
            string[] names = { "cube_red", "cube_green", "cube_blue" };
            for (int i = 0; i < names.Length; i++)
            {
                var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
                cube.name = names[i];
                cube.transform.localScale = Vector3.one * 0.02f;
                cube.transform.position = rosPositions[i].Ros2Unity();
                cube.GetComponent<Renderer>().sharedMaterial = SaveMaterial(names[i], colors[i]);
                cube.GetComponent<Collider>().sharedMaterial = cubeMat;
                var rb = cube.AddComponent<Rigidbody>();
                rb.mass = 0.03f;
                rb.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
                cube.AddComponent<SimObject>();
            }

            var cam = Camera.main;
            cam.transform.position = new Vector3(1.05f, 0.95f, -0.85f);
            cam.transform.LookAt(new Vector3(0.15f, 0.18f, 0.0f));
            cam.nearClipPlane = 0.01f;
            cam.fieldOfView = 45f;

            var bridge = new GameObject("Bridge").AddComponent<BridgeServer>();
            bridge.robot = robot.GetComponent<AR4Controller>();
            bridge.overlay = new GameObject("Overlay").AddComponent<Overlay>();

            // Lockstep: physics only advances when the bridge calls Physics.Simulate.
            Physics.simulationMode = SimulationMode.Script;

            Directory.CreateDirectory("Assets/Scenes");
            EditorSceneManager.SaveScene(scene, ScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            AssetDatabase.SaveAssets();
            Debug.Log($"[AR4] scene written to {ScenePath}");
        }

        /// <summary>A box given in ROS base_link coordinates (center, size along x, y, z).</summary>
        static GameObject RosBox(string name, Vector3 rosCenter, Vector3 rosSize, Material mat, Transform parent = null)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.position = rosCenter.Ros2Unity();
            go.transform.localScale = new Vector3(rosSize.y, rosSize.z, rosSize.x); // ROS (x, y, z) -> Unity (z, x, y) extents
            go.GetComponent<Renderer>().sharedMaterial = mat;
            return go;
        }

        static Material SaveMaterial(string name, Color color)
        {
            Directory.CreateDirectory("Assets/Materials");
            var mat = new Material(Shader.Find("Standard")) { color = color };
            AssetDatabase.CreateAsset(mat, $"Assets/Materials/{name}.mat");
            return mat;
        }

        [MenuItem("AR4/Build Linux Player")]
        public static void BuildPlayer() => Build(BuildTarget.StandaloneLinux64, PlayerPath);

        // Windows and macOS players need the "Windows Build Support (Mono)" and
        // "Mac Build Support (Mono)" modules of the editor (Unity Hub -> Installs -> Add modules).
        [MenuItem("AR4/Build Windows Player")]
        public static void BuildWindowsPlayer() => Build(BuildTarget.StandaloneWindows64, WindowsPlayerPath);

        [MenuItem("AR4/Build macOS Player")]
        public static void BuildMacPlayer() => Build(BuildTarget.StandaloneOSX, MacPlayerPath);

        static void Build(BuildTarget target, string path)
        {
            PlayerSettings.runInBackground = true;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.defaultScreenWidth = 1280;
            PlayerSettings.defaultScreenHeight = 720;
            PlayerSettings.resizableWindow = true;
            if (!BuildPipeline.IsBuildTargetSupported(BuildPipeline.GetBuildTargetGroup(target), target))
                throw new System.Exception($"build support for {target} is not installed in this editor (Unity Hub -> Installs -> Add modules)");
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = path,
                target = target,
                options = BuildOptions.None,
            });
            if (report.summary.result != UnityEditor.Build.Reporting.BuildResult.Succeeded)
                throw new System.Exception($"build failed: {report.summary.result}");
            Debug.Log($"[AR4] player built: {Path.GetFullPath(path)}");
        }
    }
}
