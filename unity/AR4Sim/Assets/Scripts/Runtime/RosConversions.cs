using UnityEngine;

namespace AR4Sim
{
    /// <summary>
    /// ROS (right-handed, x fwd, y left, z up) &lt;-&gt; Unity (left-handed, x right, y up, z fwd).
    /// Same convention as the URDF Importer, whose own helpers are internal.
    /// </summary>
    public static class RosConversions
    {
        public static Vector3 Ros2Unity(this Vector3 v) => new Vector3(-v.y, v.z, v.x);
        public static Vector3 Unity2Ros(this Vector3 v) => new Vector3(v.z, -v.x, v.y);
        public static Quaternion Ros2Unity(this Quaternion q) => new Quaternion(q.y, -q.z, -q.x, q.w);
        public static Quaternion Unity2Ros(this Quaternion q) => new Quaternion(-q.z, q.x, -q.y, q.w);
    }
}
