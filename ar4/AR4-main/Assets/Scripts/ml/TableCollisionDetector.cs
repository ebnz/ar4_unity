using UnityEngine;
using Unity.MLAgents;

/// <summary>
/// Attach this to robot parts (joints, end effector, etc.) to detect table collisions
/// and forward them to the main agent.
/// </summary>
public class TableCollisionDetector : MonoBehaviour
{
    private MonoBehaviour agent;

    void Start()
    {
        // Try to find any compatible agent in parent hierarchy
        agent = GetComponentInParent<Agent>() as MonoBehaviour;

        if (agent == null)
        {
            Debug.LogWarning($"TableCollisionDetector on {gameObject.name} couldn't find any compatible agent in parent!");
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (agent != null && IsTable(collision.gameObject))
        {
            // Use reflection to call the method on whichever agent type we found
            agent.GetType().GetMethod("OnTableCollisionEnter")?.Invoke(agent, new object[] { collision });
        }
    }

    private void OnCollisionStay(Collision collision)
    {
        if (agent != null && IsTable(collision.gameObject))
        {
            agent.GetType().GetMethod("OnTableCollisionStay")?.Invoke(agent, new object[] { collision });
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        if (agent != null && IsTable(collision.gameObject))
        {
            agent.GetType().GetMethod("OnTableCollisionExit")?.Invoke(agent, new object[] { collision });
        }
    }

    private bool IsTable(GameObject obj)
    {
        return obj.name.Contains("Table") || obj.name.Contains("table");
    }
}
