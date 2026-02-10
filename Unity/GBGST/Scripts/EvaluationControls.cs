using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;
using Unity.Barracuda;

namespace GBGST.Scripts
{
    public class EvaluationControls : MonoBehaviour
    {
        #region Properties

        [Header("Custom Pass Settings")] public CustomPassVolume customPassVolume;
        [Header("FPS Counter")] public FpsCounter fpsCounter;
        [Header("NN Models")]
        public List<NNModel> models = new List<NNModel>();

        #endregion

        private int currentId = 0;
        private int currentModelIndex = 0;
        private CopyPassStylizationEffects stylizationPass;

        void Start()
        {
            Debug.Log($"Screenshot directory: {Application.persistentDataPath}");

            if (customPassVolume != null)
            {
                foreach (var pass in customPassVolume.customPasses)
                {
                    if (pass is CopyPassStylizationEffects cp)
                    {
                        stylizationPass = cp;
                        break;
                    }
                }
            }

            if (fpsCounter != null)
                fpsCounter.MeasurementCompleted += (result => { Debug.Log($"{result.Display()}"); });
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

            if (Input.GetKeyDown(KeyCode.T))
            {
                fpsCounter.StartMeasurement(5f);
            }

            if (Input.GetKeyDown(KeyCode.N))
            {
                LoadModelAtIndex(currentModelIndex);
                currentModelIndex = ( currentModelIndex + 1 ) % Mathf.Max(1, models.Count);
            }
        }

        public void LoadModelAtIndex(int index)
        {
            if (models == null || models.Count == 0)
            {
                Debug.LogWarning("List of models is empty.");
                return;
            }

            index = ((index % models.Count) + models.Count) % models.Count;
            NNModel model = models[index];

            if (stylizationPass != null)
            {
                stylizationPass.SetModelAndReload(model);
            }
            else
            {
                Debug.LogWarning("CopyPassStylizationEffects not found in Custom Pass Volume.");
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