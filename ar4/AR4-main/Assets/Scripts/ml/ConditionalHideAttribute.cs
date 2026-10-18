using UnityEngine;
using System;

/// <summary>
/// Attribute to hide a field in the inspector based on a boolean condition
/// </summary>
[AttributeUsage(AttributeTargets.Field, AllowMultiple = false, Inherited = true)]
public class ConditionalHideAttribute : PropertyAttribute
{
    public string ConditionalSourceField { get; private set; }
    public bool HideWhenTrue { get; private set; }

    /// <summary>
    /// Hide this field based on the value of another boolean field
    /// </summary>
    /// <param name="conditionalSourceField">The name of the boolean field to check</param>
    /// <param name="hideWhenTrue">If true, hide when the source field is true. If false, hide when source field is false.</param>
    public ConditionalHideAttribute(string conditionalSourceField, bool hideWhenTrue = true)
    {
        ConditionalSourceField = conditionalSourceField;
        HideWhenTrue = hideWhenTrue;
    }
}
