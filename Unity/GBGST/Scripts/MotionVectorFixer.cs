using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

namespace GBGST.Scripts
{
    public class MotionVectorFixer : MonoBehaviour
    {
        void Start()
        {
            var cameraData = GetComponent<HDAdditionalCameraData>();
            if (cameraData != null)
            {
                cameraData.customRenderingSettings = true;
                
                var settings = cameraData.renderingPathCustomFrameSettingsOverrideMask;
                settings.mask[(uint)FrameSettingsField.MotionVectors] = true;
                
                GetComponent<Camera>().depthTextureMode |= DepthTextureMode.MotionVectors | DepthTextureMode.Depth;

                Debug.Log("Motion Vectors forced programmatically for: " + gameObject.name);
            }
        }
    }
}