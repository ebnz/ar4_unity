using UnityEngine;
using UnityEditor;

[CustomPropertyDrawer(typeof(ConditionalHideAttribute))]
public class ConditionalHidePropertyDrawer : PropertyDrawer
{
    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        ConditionalHideAttribute condHide = (ConditionalHideAttribute)attribute;
        bool enabled = GetConditionalHideAttributeResult(condHide, property);

        if (enabled)
        {
            EditorGUI.PropertyField(position, property, label, true);
        }
    }

    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {
        ConditionalHideAttribute condHide = (ConditionalHideAttribute)attribute;
        bool enabled = GetConditionalHideAttributeResult(condHide, property);

        if (enabled)
        {
            return EditorGUI.GetPropertyHeight(property, label);
        }
        else
        {
            return -EditorGUIUtility.standardVerticalSpacing;
        }
    }

    private bool GetConditionalHideAttributeResult(ConditionalHideAttribute condHide, SerializedProperty property)
    {
        bool enabled = true;
        SerializedProperty sourcePropertyValue = property.serializedObject.FindProperty(condHide.ConditionalSourceField);

        if (sourcePropertyValue != null)
        {
            enabled = sourcePropertyValue.boolValue;
            if (condHide.HideWhenTrue)
            {
                enabled = !enabled;
            }
        }
        else
        {
            Debug.LogWarning($"Attempting to use ConditionalHideAttribute but no boolean field named '{condHide.ConditionalSourceField}' found in object");
        }

        return enabled;
    }
}
