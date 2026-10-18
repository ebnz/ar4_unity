using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

[RequireComponent(typeof(ArticulationBody))]
public class ArticulationBodyDriver : MonoBehaviour
{
    [Header("Joint Control")]
    [SerializeField] private float targetPosition = 0f;
    [SerializeField] private float speed = 10f;
    [SerializeField] private float maxForce = 1000f;
    
    private ArticulationBody articulationBody;
    
    private void Awake()
    {
        InitializeArticulationBody();
    }
    
    private void InitializeArticulationBody()
    {
        if (articulationBody == null)
            articulationBody = GetComponent<ArticulationBody>();
        
        // Initialize the drive
        var drive = articulationBody.xDrive;
        drive.forceLimit = maxForce;
        drive.stiffness = 10000;
        drive.damping = 100;
        articulationBody.xDrive = drive;
    }
    
    private void Update()
    {
        var drive = articulationBody.xDrive;
        drive.target = targetPosition;
        articulationBody.xDrive = drive;
    }
    
#if UNITY_EDITOR
    [CustomEditor(typeof(ArticulationBodyDriver))]
    public class ArticulationBodyDriverEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();
            
            ArticulationBodyDriver driver = (ArticulationBodyDriver)target;
            
            // Make sure the articulation body is initialized
            if (driver.articulationBody == null)
                driver.articulationBody = driver.GetComponent<ArticulationBody>();
                
            if (driver.articulationBody == null)
            {
                EditorGUILayout.HelpBox("ArticulationBody component not found!", MessageType.Error);
                return;
            }
            
            var drive = driver.articulationBody.xDrive;
            
            GUILayout.Space(10);
            GUILayout.Label("Manual Control", EditorStyles.boldLabel);
            
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Move Left"))
            {
                driver.targetPosition -= driver.speed * 0.1f;
                driver.targetPosition = Mathf.Clamp(driver.targetPosition, drive.lowerLimit, drive.upperLimit);
                EditorUtility.SetDirty(target);
            }
            
            if (GUILayout.Button("Move Right"))
            {
                driver.targetPosition += driver.speed * 0.1f;
                driver.targetPosition = Mathf.Clamp(driver.targetPosition, drive.lowerLimit, drive.upperLimit);
                EditorUtility.SetDirty(target);
            }
            GUILayout.EndHorizontal();
            
            GUILayout.Space(5);
            float newPosition = GUILayout.HorizontalSlider(
                driver.targetPosition, 
                drive.lowerLimit, 
                drive.upperLimit
            );
            GUILayout.Space(10);
            
            if (newPosition != driver.targetPosition)
            {
                driver.targetPosition = newPosition;
                EditorUtility.SetDirty(target);
            }
            
            GUILayout.Space(5);
            if (GUILayout.Button("Reset Position"))
            {
                driver.targetPosition = 0f;
                EditorUtility.SetDirty(target);
            }
        }
    }
#endif
}