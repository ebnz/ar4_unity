using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using UnityEngine;

public class MoveToTarget : MonoBehaviour
{
    private Vector3 _distanceToTarget;
    private Vector3 _endEffectorPosition;
    private Vector3 _target;
    private RobotManager _robotManagerInstance;
    private float _tolerance = 0.001f;

    /// <summary>
    /// Calculates the distance from the closest end-effector sensor
    /// to the target position.
    /// </summary>
    private void CalculateDistanceToTarget()
    {
        float minDistance = float.MaxValue;

        for (int i = 0; i < _robotManagerInstance.gripper.Length; i++)
        {
            Vector3 gripperPosition = _robotManagerInstance.gripper[i].transform.position;
            Vector3 distanceToGripper = _target - gripperPosition;

            float distanceMagnitude = distanceToGripper.magnitude;
            if (distanceMagnitude < minDistance)
            {
                minDistance = distanceMagnitude;
                _endEffectorPosition = gripperPosition;
                _distanceToTarget = distanceToGripper;
            }
        }
    }

    /// <summary>
    /// Calculates the Jacobian matrix.
    /// </summary>
    private Matrix<double> CalculateJacobian(int numJoints, Vector3 endEffectorPosition)
    {
        Matrix<double> jacobian = DenseMatrix.Build.Dense(6, numJoints);

        for (int i = 0; i < numJoints; i++)
        {
            var joint = _robotManagerInstance.robotJoints[i];
            var jointTransform = joint.transform;
            Vector3 jointPosition = jointTransform.position;

            // Determine the rotation axis in world space
            Quaternion anchorRotation = joint.anchorRotation;
            Vector3 worldRotationAxis = jointTransform.TransformDirection(
                anchorRotation * Vector3.right
            );

            // Compute linear velocity part (same as before)
            Vector3 linearVelocity = Vector3.Cross(
                worldRotationAxis,
                endEffectorPosition - jointPosition
            );

            // Compute angular velocity part (just the rotation axis)
            Vector3 angularVelocity = worldRotationAxis;

            // Assign values to the Jacobian matrix
            jacobian[0, i] = linearVelocity.x;
            jacobian[1, i] = linearVelocity.y;
            jacobian[2, i] = linearVelocity.z;

            jacobian[3, i] = angularVelocity.x;
            jacobian[4, i] = angularVelocity.y;
            jacobian[5, i] = angularVelocity.z;
        }

        return jacobian;
    }

    /// <summary>
    /// Computes the damped least squares pseudo-inverse of the Jacobian matrix.
    /// Returns joint velocities directly instead of the pseudo-inverse matrix.
    /// </summary>
    private Vector<double> ComputeDampedLeastSquares(Matrix<double> jacobian, Vector<double> errorVector)
    {
        // Get damping factor from RobotManager
        float lambda = _robotManagerInstance.dampingFactorLambda;

        // Compute J^T
        var jacobianTranspose = jacobian.Transpose();

        // Compute JJ^T
        var jacobianJacobianT = jacobian * jacobianTranspose;

        // Create identity matrix of appropriate size
        var identity = DenseMatrix.Build.DenseIdentity(jacobianJacobianT.RowCount);

        // Add damping term: JJ^T + λ²I
        var regularized = jacobianJacobianT + (lambda * lambda) * identity;

        // Solve (JJ^T + λ²I)y = error for y, then compute deltaTheta = J^T * y
        var y = regularized.Solve(errorVector);
        var deltaTheta = jacobianTranspose * y;

        return deltaTheta;
    }

    /// <summary>
    /// Calculates the angles for the robot joints by inverse kinematics.
    /// </summary>
    public void InverseKinematics()
    {
        int numJoints = _robotManagerInstance.robotJoints.Length;
        if (numJoints == 0)
        {
            Debug.LogWarning("No robot joints found for inverse kinematics.");
            return;
        }

        // Calculate the 6xN Jacobian matrix
        Matrix<double> jacobian = CalculateJacobian(numJoints, _endEffectorPosition);

        // Position error vector (X, Y, Z)
        Vector<double> positionError = Vector<double>.Build.DenseOfArray(
            new double[] { _distanceToTarget.x, _distanceToTarget.y, _distanceToTarget.z }
        );

        // Compute orientation error (rotation difference)
        Quaternion currentRotation = _robotManagerInstance.sensorRotation.rotation;
        Quaternion rotationDifference = _robotManagerInstance.target.rotation * Quaternion.Inverse(currentRotation);

        // Convert quaternion difference to axis-angle representation
        Vector3 rotationAxis;
        float rotationAngle;
        rotationDifference.ToAngleAxis(out rotationAngle, out rotationAxis);
        Vector3 orientationError = rotationAxis * Mathf.Deg2Rad * rotationAngle; // Convert to radians

        // Combine position and orientation errors into a single 6D error vector
        Vector<double> errorVector = Vector<double>.Build.DenseOfArray(
            new double[]
            {
                positionError[0],
                positionError[1],
                positionError[2],
                orientationError.x,
                orientationError.y,
                orientationError.z,
            }
        );

        float distanceFactor = Mathf.Clamp01(_distanceToTarget.magnitude); // 0 to 1 based on distance
        float adaptiveSpeed = Mathf.Lerp(0.3f, 1.0f, distanceFactor); // Slower as it nears target

        // Compute joint velocities using damped least squares
        Vector<double> deltaTheta = ComputeDampedLeastSquares(jacobian, errorVector) * adaptiveSpeed;

        // Apply the joint adjustments
        float robotSpeed = _robotManagerInstance.GetRobotSpeed();
        for (int i = 0; i < numJoints; i++)
        {
            var joint = _robotManagerInstance.robotJoints[i];
            var drive = joint.xDrive; 

            float newTarget = Mathf.Clamp(
                drive.target + (float)deltaTheta[i] * robotSpeed,
                drive.lowerLimit,
                drive.upperLimit
            );

            if (!Mathf.Approximately(newTarget, drive.target))
            {
                drive.target = newTarget;
                joint.xDrive = drive;
            }
        }
    }

    public bool IsToleranceReached()
    {
        return _distanceToTarget.magnitude < _tolerance;
    }

    /// <summary>
    /// Updates the target vector.
    /// </summary>
    public void SetTarget(Vector3 target)
    {
        _target = target;
    }

    private void UpdateRobotMovement()
    {
        if(_target == null)
            return;

        // Update the distance to the target before any movement logic
        CalculateDistanceToTarget();

        Debug.DrawLine(_endEffectorPosition, _target, Color.red);
        Debug.DrawRay(_endEffectorPosition, Vector3.right * 0.1f, Color.blue);

        InverseKinematics();
    }

    private void Start()
    {
        _robotManagerInstance = GetComponent<RobotManager>();
        if (_robotManagerInstance == null)
        {
            Debug.LogError("MoveToTarget requires a RobotManager component on the same GameObject!");
        }
        _target = _robotManagerInstance.gripper[0].transform.position;
    }

    private void FixedUpdate()
    {
        UpdateRobotMovement();
    }
}
