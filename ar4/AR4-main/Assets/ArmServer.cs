using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

[System.Serializable] public class Cmd { public float[] q; }   // radians
[System.Serializable] public class State { public float[] q; }

public class ArmServer : MonoBehaviour
{
    public ArticulationBody[] joints;
    TcpListener listener;
    volatile float[] target;
    volatile float[] state;

    void Start()
    {
        target = new float[joints.Length];
        state = new float[joints.Length];
        listener = new TcpListener(IPAddress.Loopback, 5555);
        listener.Start();
        new Thread(Serve) { IsBackground = true }.Start();
    }

    void Serve()
    {
        while (true)
        {
            using var c = listener.AcceptTcpClient();
            using var r = new StreamReader(c.GetStream());
            using var w = new StreamWriter(c.GetStream()) { AutoFlush = true };
            string line;
            while ((line = r.ReadLine()) != null)
            {
                var cmd = JsonUtility.FromJson<Cmd>(line);
                if (cmd.q != null && cmd.q.Length == joints.Length) target = cmd.q;
                w.WriteLine(JsonUtility.ToJson(new State { q = state }));
            }
        }
    }

    void FixedUpdate()
    {
        var s = new float[joints.Length];
        for (int i = 0; i < joints.Length; i++)
        {
            var d = joints[i].xDrive;                    // struct: copy, modify, assign back
            d.target = target[i] * Mathf.Rad2Deg;        // drive targets are in degrees
            joints[i].xDrive = d;
            s[i] = joints[i].jointPosition[0];           // radians
        }
        state = s;
    }
}
