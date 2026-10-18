using UnityEngine;

public class SimpleRobotController : MonoBehaviour
{
    private MoveToTarget _moveToTarget;
    
    [SerializeField]
    public ArticulationBody[] robotJoints;
    
    [SerializeField]
    public Transform[] gripperPoints;
    
    [SerializeField]
    public Transform target;

    private void Start()
    {
        // Get and configure MoveToTarget
        _moveToTarget = GetComponent<MoveToTarget>();
        
        // Configure initial robot joint settings
        SetUpRobotJoints();

        MoveTo(target.position);
    }

    public void MoveTo(Vector3 target)
    {
        _moveToTarget.SetTarget(target);
    }

    private void SetUpRobotJoints()
    {
        // Configuration values for an AR4 robotic arm
        float[] stiffnessValues = { 15000, 12000, 10000, 8000, 5000, 2000 };
        float[] dampingValues = { 100, 150, 150, 100, 80f, 50 };
        float[] forceLimits = { 1000, 1500, 1000, 800, 500, 300 };
        float[] driveUpperLimits = { 170, 90, 65, 135, 100, 180 };
        float[] driveLowerLimits = { -170, -90, -70, -135, -100, -180 };

        for (int i = 0; i < robotJoints.Length; i++)
        {
            var drive = robotJoints[i].xDrive;
            drive.stiffness = stiffnessValues[i];
            drive.damping = dampingValues[i];
            drive.forceLimit = forceLimits[i];
            drive.upperLimit = driveUpperLimits[i];
            drive.lowerLimit = driveLowerLimits[i];
            robotJoints[i].xDrive = drive;
        }
    }
}