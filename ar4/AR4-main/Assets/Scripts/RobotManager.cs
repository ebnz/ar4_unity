using UnityEngine;

public class RobotManager : MonoBehaviour
{
    private readonly float[] _jointLengths = { 0.45f, 0.75f, 0.0f, 0.55f, 0.1f, 0.11f };
    private TrailRenderer _trailRenderer;
    private MoveToTarget _moveRobot;
    private Vector3 _defaultPosition;
    private bool _robotManagerCommand = false;
    private bool _robotPositioningFinished = true;
    private bool _robotReachedTable = false;
    public ArticulationBody[] robotJoints;
    public Transform[] gripper;
    public Transform target;
    public Transform sensorRotation;
    public float robotSpeed = 0.3f;

    [Header("IK Configuration")]
    [SerializeField, Range(0.01f, 1f)]
    public float dampingFactorLambda = 0.1f;  // Damping for pseudo-inverse stability

    private float[] _initialJointPositions;
    private Vector3 _initialTargetPosition;
    private Vector3 _initialGripperPosition;

    private void SetUpRobot()
    {
        // Ensure robot joints are assigned
        if (robotJoints == null || robotJoints.Length == 0)
        {
            Debug.LogError("No ArticulationBody components assigned. Check your hierarchy.");
            return;
        }

        // Configuration values for an AR4 robotic arm
        float[] stiffnessValues = { 15000, 12000, 10000, 8000, 5000, 2000 };
        float[] dampingValues = { 100, 150, 150, 100, 80, 50 };
        float[] forceLimits = { 1000, 1500, 1000, 800, 500, 300 };
        float[] driveUpperLimits = { 170, 90, 65, 135, 100, 180 };
        float[] driveLowerLimits = { -170, -90, -70, -135, -100, -180 };

        // Configure joint drives
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

        // Store initial positions
        _initialJointPositions = new float[robotJoints.Length];
        for (int i = 0; i < robotJoints.Length; i++)
        {
            _initialJointPositions[i] = robotJoints[i].xDrive.target;
        }
        _initialTargetPosition = target.position;
        _initialGripperPosition = gripper[0].transform.position;
    }

    public void RaiseRobotArm(float height)
    {
        Vector3 positionOffset = Vector3.up * height;
        Vector3 newTarget = gripper[0].transform.position + positionOffset;
        Debug.Log("Raise robot arm to: " + newTarget);

        _robotManagerCommand = true;
        MoveTo(newTarget);
    }

    public void ReturnToDefaultPosition()
    {
        Debug.Log("Returning to default position: " + _defaultPosition);
        
        _robotManagerCommand = true;
        MoveTo(_defaultPosition);
    }

    public void MoveTo(Vector3 target)
    {
        _robotPositioningFinished = false;
        
        _moveRobot.SetTarget(target);
    }

    public void SetTrailRenderer(bool setting)
    {
        _trailRenderer.enabled = setting;
    }

    public void SetRobotManagerCommand(bool setting)
    {
        _robotManagerCommand = setting;
    }

    public bool GetRobotManagerCommand()
    {
        return _robotManagerCommand;
    }

    public void SetRobotPositioningFinished(bool setting)
    {
        _robotPositioningFinished = setting;
    }

    public bool GetRobotPositioningFinished()
    {
        return _robotPositioningFinished;
    }

    public bool GetToleranceReached()
    {
        return _moveRobot.IsToleranceReached();
    }

    public bool HasRobotReachedTable()
    {
        return _robotReachedTable;
    }

    public void SetRobotReachedTable(bool setting)
    {
        _robotReachedTable = setting;
    }

    public Quaternion GetEndEffectorRotation()
    {
        return gripper[0].rotation;
    }

    public float GetRobotSpeed()
    {
        return robotSpeed;
    }

    public void ResetRobotInstantly()
    {
        // Reset target position instantly
        target.position = _initialTargetPosition;

        // Reset all joint positions instantly
        for (int i = 0; i < robotJoints.Length; i++)
        {
            var joint = robotJoints[i];
            
            // Reset joint drive
            var drive = joint.xDrive;
            drive.target = _initialJointPositions[i];
            joint.xDrive = drive;
            
            // Reset velocities and forces
            joint.jointPosition = new ArticulationReducedSpace(0f);
            joint.jointForce = new ArticulationReducedSpace(0f);
            joint.jointVelocity = new ArticulationReducedSpace(0f);
        }

        // Force position update
        _moveRobot.SetTarget(_initialTargetPosition);
        _robotPositioningFinished = true;
    }

    private void Awake()
    {
        _trailRenderer = gripper[0].GetComponent<TrailRenderer>();
        _moveRobot = GetComponent<MoveToTarget>();
        _defaultPosition = gripper[0].transform.position;

        SetUpRobot();
    }

    private void Start()
    {
        // Move initialization to Awake, so Start can be empty or removed
    }

    private void Update() {
        _moveRobot.SetTarget(target.position);
    }
}
