using System.IO;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

namespace GBGST.Scripts
{
    public class EvaluationControls : MonoBehaviour
    {
        #region Properties

        [Header("Custom Pass Settings")] public CustomPassVolume customPassVolume;

        #endregion

        private int currentId = 0;

        void Start()
        {
            Debug.Log($"Screenshot directory: {Application.persistentDataPath}");
        }
        
        void Update()
        {
            if (Input.GetKeyDown(KeyCode.R))
            {
                TakeNumberedScreenshot();
            }

            if (Input.GetKeyDown(KeyCode.F))
            {
                currentId++;
            }

            if (Input.GetKeyDown(KeyCode.C))
            {
                if (customPassVolume != null)
                {
                    customPassVolume.enabled = !customPassVolume.enabled;
                }
            }
        }

        private void TakeNumberedScreenshot()
        {
            bool isPassActive = customPassVolume != null && customPassVolume.enabled;

            string folderPath = $"Evaluation/{currentId}";

            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }

            string filename = $"{folderPath}/{isPassActive}.png";

            ScreenCapture.CaptureScreenshot(filename);
            Debug.Log($"Saved screenshot as: {filename}");
        }
    }
}