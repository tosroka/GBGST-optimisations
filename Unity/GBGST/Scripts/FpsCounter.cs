using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace GBGST.Scripts
{
    public class FpsCounter : MonoBehaviour
    {
        [Tooltip("Measurement duration in seconds")] [Min(0.1f)] [SerializeField]
        private float measurementDuration = 5f;

        public event Action<FpsHistogramResult> MeasurementCompleted;

        private readonly List<float> frameTimes = new List<float>();
        private float measurementStartTime;
        private bool isMeasuring;

        public void StartMeasurement(float? durationSeconds = null)
        {
            float duration = durationSeconds ?? measurementDuration;
            if (duration <= 0f)
            {
                duration = measurementDuration;
            }

            frameTimes.Clear();
            measurementStartTime = Time.realtimeSinceStartup;
            isMeasuring = true;
            measurementDuration = duration;
        }

        private void Update()
        {
            if (!isMeasuring)
            {
                return;
            }

            frameTimes.Add(Time.unscaledDeltaTime);

            if (Time.realtimeSinceStartup - measurementStartTime < measurementDuration)
            {
                return;
            }

            isMeasuring = false;
            MeasurementCompleted?.Invoke(ComputeHistogram());
        }

        private FpsHistogramResult ComputeHistogram()
        {
            if (frameTimes.Count == 0)
            {
                return default;
            }

            var sorted = new List<float>(frameTimes);
            sorted.Sort();

            float minFt = sorted[0];
            float maxFt = sorted[^1];
            int i99 = Mathf.Clamp((int)Math.Ceiling(sorted.Count * 0.99) - 1, 0, sorted.Count - 1);
            float top99Ft = sorted[i99];

            float ToFps(float ft) => ft > 0.0001f ? 1f / ft : 0f;

            return new FpsHistogramResult
            {
                sampleCount = frameTimes.Count,
                minFps = ToFps(maxFt),
                maxFps = ToFps(minFt),
                top99Fps = ToFps(top99Ft),
                minFrameTimeMs = minFt * 1000f,
                maxFrameTimeMs = maxFt * 1000f,
                top99FrameTimeMs = top99Ft * 1000f,
                averageFrameTimeMs = sorted.Sum() * 1000f / sorted.Count,
            };
        }

        [Serializable]
        public struct FpsHistogramResult
        {
            public int sampleCount;
            public float minFps;
            public float maxFps;
            public float top99Fps;
            public float minFrameTimeMs;
            public float maxFrameTimeMs;
            public float top99FrameTimeMs;
            public float averageFrameTimeMs;

            public string Display()
            {
                return
                    $"Sample count: {sampleCount}, Min FPS: {minFps}, Max FPS: {maxFps}, Top 99 FPS: {top99Fps}, Min frame time: {minFrameTimeMs}ms, Max frame time: {maxFrameTimeMs}ms, Top 99 frame time: {top99FrameTimeMs}ms, Avg frame time: {averageFrameTimeMs}";
            }
        }
    }
}