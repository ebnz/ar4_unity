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
        const string PlayerPath = "../Build/AR4Sim.x86_64";

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

            // URDF uses gray/yellow; give the floor some contrast.
            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.localScale = new Vector3(0.3f, 1, 0.3f); // 3 m x 3 m
            ground.GetComponent<Renderer>().sharedMaterial = SaveMaterial("Ground", new Color(0.35f, 0.37f, 0.4f));

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
            cam.transform.position = new Vector3(0.75f, 0.55f, -0.75f);
            cam.transform.LookAt(new Vector3(0.15f, 0.2f, 0f));
            cam.nearClipPlane = 0.01f;
            cam.fieldOfView = 45f;

            var bridge = new GameObject("Bridge").AddComponent<BridgeServer>();
            bridge.robot = robot.GetComponent<AR4Controller>();

            // Lockstep: physics only advances when the bridge calls Physics.Simulate.
            Physics.simulationMode = SimulationMode.Script;

            Directory.CreateDirectory("Assets/Scenes");
            EditorSceneManager.SaveScene(scene, ScenePath);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath, true) };
            AssetDatabase.SaveAssets();
            Debug.Log($"[AR4] scene written to {ScenePath}");
        }

        static Material SaveMaterial(string name, Color color)
        {
            Directory.CreateDirectory("Assets/Materials");
            var mat = new Material(Shader.Find("Standard")) { color = color };
            AssetDatabase.CreateAsset(mat, $"Assets/Materials/{name}.mat");
            return mat;
        }

        [MenuItem("AR4/Build Linux Player")]
        public static void BuildPlayer()
        {
            PlayerSettings.runInBackground = true;
            PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
            PlayerSettings.defaultScreenWidth = 1280;
            PlayerSettings.defaultScreenHeight = 720;
            PlayerSettings.resizableWindow = true;
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = PlayerPath,
                target = BuildTarget.StandaloneLinux64,
                options = BuildOptions.None,
            });
            if (report.summary.result != UnityEditor.Build.Reporting.BuildResult.Succeeded)
                throw new System.Exception($"build failed: {report.summary.result}");
            Debug.Log($"[AR4] player built: {Path.GetFullPath(PlayerPath)}");
        }
    }
}
