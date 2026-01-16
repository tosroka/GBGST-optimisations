using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;
using Unity.Barracuda;
using UnityEditor;
using System;

namespace GBGST.Scripts
{
    public class CopyPassStylizationEffects : CustomPass
    {
        #region Properties

        [Tooltip("Reference to the Barracuda neural network asset")]
        public NNModel modelAsset;

        [Tooltip(
            "Reference to the compute shader responsible for preparing of finalizing the image for style transfer")]
        public ComputeShader styleTransferShader;

        [Tooltip("Reference to the computer shader responsible for @TODO")]
        public ComputeShader styleTransferTemporalShader;

        [Tooltip("The backend used when performing inference")]
        public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

        [SerializeField] [Tooltip("Reference to the compute shader responsible for GBuffer")]
        public ComputeShader gbufferShader;

        #endregion

        private Model styleTransferModel;

        private IWorker styleTransferWorker;

        private IWorker inferenceEngine;

        protected override bool executeInSceneView => true;

        int normalPass;
        int roughnessPass;
        int depthPass;
        private RenderTexture diffuseTexture;
        Material gbufferMaterial;

        private RenderTexture motionVectorsTexture;
        private RenderTexture normalsTexture;
        private RenderTexture previousNormalsTexture;
        private RenderTexture depthTexture;
        private RenderTexture ambientOcclusionTexture;

        RenderTexture previousStylized;
        bool hasPreviousStylizedBeenSet;

        #region Setup & Cleanup

        protected override void Setup(ScriptableRenderContext renderContext, CommandBuffer cmd)
        {
            if (modelAsset != null)
            {
                styleTransferModel = ModelLoader.Load(modelAsset);
            }

            if (styleTransferModel != null)
            {
                styleTransferWorker = WorkerFactory.CreateWorker(styleTransferModel);
            }

            inferenceEngine = WorkerFactory.CreateWorker(workerType, styleTransferModel);

            if (gbufferShader == null)
            {
                gbufferShader = Resources.Load<ComputeShader>("GBGST/Shaders/GbufferShader");
            }

            previousStylized = new RenderTexture(1920, 1080, 0, RenderTextureFormat.ARGB32)
            {
                enableRandomWrite = true
            };
            previousStylized.Create();

            previousNormalsTexture = new RenderTexture(1920, 1080, 0, RenderTextureFormat.ARGB32)
            {
                enableRandomWrite = true
            };
            previousNormalsTexture.Create();

            hasPreviousStylizedBeenSet = false;
        }

        protected override void Cleanup()
        {
            base.Cleanup();

            styleTransferWorker?.Dispose();
            previousStylized?.Release();
            previousNormalsTexture?.Release();
            motionVectorsTexture?.Release();
            normalsTexture?.Release();
            depthTexture?.Release();
            ambientOcclusionTexture?.Release();

            styleTransferWorker = null;
            styleTransferModel = null;
            previousStylized = null;
            previousNormalsTexture = null;

            GC.Collect();
            Resources.UnloadUnusedAssets();
            EditorUtility.UnloadUnusedAssetsImmediate();
        }

        #endregion

        #region Execute

        private static readonly int GBufferShaderCameraBuffer = Shader.PropertyToID("CameraBuffer");
        private static readonly int GBufferShaderStylizedTexture = Shader.PropertyToID("StylizedTexture");
        
        protected override void Execute(CustomPassContext ctx)
        {
            var scale = RTHandles.rtHandleProperties.rtHandleScale;

            diffuseTexture = ctx.hdCamera.GetCurrentFrameRT((int)HDCameraFrameHistoryType.ColorBufferMipChain);
            SyncRenderTextureAspect(diffuseTexture, ctx.hdCamera.camera);

            if (!ctx.hdCamera.frameSettings.IsEnabled(FrameSettingsField.ObjectMotionVectors) ||
                !ctx.hdCamera.frameSettings.IsEnabled(FrameSettingsField.OpaqueObjects))
            {
                Debug.Log(
                    $"Motion Vectors: {ctx.hdCamera.frameSettings.IsEnabled(FrameSettingsField.ObjectMotionVectors)}");
                Debug.Log($"Opaque Objects: {ctx.hdCamera.frameSettings.IsEnabled(FrameSettingsField.OpaqueObjects)}");
                Debug.Log("Motion Vectors are disabled on the camera!");
                return;
            }

            ClearRenderTexture(normalsTexture);
            ClearRenderTexture(previousNormalsTexture);
            ClearRenderTexture(motionVectorsTexture);
            ClearRenderTexture(depthTexture);
            ClearRenderTexture(ambientOcclusionTexture);

            // MotionVectors
            ctx.cmd.Blit(ctx.cameraMotionVectorsBuffer, motionVectorsTexture, new Vector2(scale.x, scale.y),
                Vector2.zero, 0, 0);

            // Normals
            normalsTexture = ctx.cameraNormalBuffer;

            // Depth
            depthTexture = ctx.cameraDepthBuffer;

            // AO
            ambientOcclusionTexture = ctx.hdCamera.GetCurrentFrameRT((int)HDCameraFrameHistoryType.AmbientOcclusion);

            SyncRenderTextureAspect(depthTexture, ctx.hdCamera.camera);

            StylizeImage(diffuseTexture, previousStylized, motionVectorsTexture, normalsTexture, depthTexture,
                ambientOcclusionTexture, hasPreviousStylizedBeenSet);

            // Save previously stylized & normals
            Graphics.Blit(diffuseTexture, previousStylized);
            Graphics.Blit(normalsTexture, previousNormalsTexture);

            Texture2D texture2D =
                new Texture2D(diffuseTexture.width, diffuseTexture.height, TextureFormat.RGBA32, false);
            RenderTexture previousActive = RenderTexture.active;
            RenderTexture.active = diffuseTexture;
            texture2D.ReadPixels(new Rect(0, 0, diffuseTexture.width, diffuseTexture.height), 0, 0);
            texture2D.Apply();
            RenderTexture.active = previousActive;

            // Copy the StylizedTexture to camera
            gbufferShader.SetTexture(0, GBufferShaderCameraBuffer, ctx.cameraColorBuffer.rt);
            gbufferShader.SetTexture(0, GBufferShaderStylizedTexture, texture2D);
            ctx.cmd.DispatchCompute(gbufferShader, 0, 64, 64, 1);
        }

        #endregion

        #region Stylization

        private void StylizeImage(RenderTexture src, RenderTexture previous, RenderTexture motionVectors,
            RenderTexture normals, RenderTexture depth, RenderTexture ao, bool hasPreviousStylizedSet)
        {
            RenderTexture source = RenderTexture.GetTemporary(src.width, src.height, 24, src.format);
            Graphics.Blit(src, source);
            
            RenderTexture rTex = RenderTexture.GetTemporary(src.width, src.height, 24, src.format);
            Graphics.Blit(src, rTex);
            
            ExecuteStyleTransferShader(rTex, StyleTransferShaderStep.PrepareInput);
            
            Tensor input = new Tensor(rTex, channels: 3);
            inferenceEngine.Execute(input);
            Tensor prediction = inferenceEngine.PeekOutput();
            input.Dispose();


            RenderTexture.active = null;
            prediction.ToRenderTexture(rTex);
            prediction.Dispose();

            if (hasPreviousStylizedSet)
            {
                ProcessImageMotionVectors(previous, rTex, motionVectors, normals, depth, ao,
                    StyleTransferTemporalShaderStep.ProcessOutput);
            }
            else ExecuteStyleTransferShader(rTex, StyleTransferShaderStep.FinalizeOutput);
            
            hasPreviousStylizedBeenSet = true;
            Graphics.Blit(rTex, src);

            // Release the temporary RenderTexture
            RenderTexture.ReleaseTemporary(rTex);
            RenderTexture.ReleaseTemporary(source);
        }

        #endregion

        #region StyleTransferEffectsShader

        private enum StyleTransferTemporalShaderStep
        {
            ProcessOutput
        }
        
        private static readonly int StyleTransferTemporalShaderResult = Shader.PropertyToID("Result");
        private static readonly int StyleTransferTemporalShaderStylizedImage = Shader.PropertyToID("StylizedImage");
        private static readonly int StyleTransferTemporalShaderPreviousStylizedImage = Shader.PropertyToID("PreviousStylizedImage");
        private static readonly int StyleTransferTemporalShaderMotionVectors = Shader.PropertyToID("MotionVectors");
        private static readonly int StyleTransferTemporalShaderNormalMap = Shader.PropertyToID("NormalMap");
        private static readonly int StyleTransferTemporalShaderDepthMap = Shader.PropertyToID("DepthMap");
        private static readonly int StyleTransferTemporalShaderOcclusion = Shader.PropertyToID("AmbientOcclusion");

        private void ProcessImageMotionVectors(RenderTexture image, RenderTexture imageStylized,
            RenderTexture motionVectors, RenderTexture normals, RenderTexture depth, RenderTexture ao,
            StyleTransferTemporalShaderStep step)
        {
            string kernelName = step.ToString();

            int kernelHandle = styleTransferTemporalShader.FindKernel(kernelName);

            RenderTexture temporary = RenderTexture.GetTemporary(imageStylized.width, imageStylized.height, 24,
                RenderTextureFormat.ARGBHalf);
            temporary.enableRandomWrite = true;
            temporary.Create();

            RenderTexture inputTexture =
                RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
            inputTexture.enableRandomWrite = true;
            inputTexture.Create();
            Graphics.Blit(image, inputTexture);

            RenderTexture inputMotionVectors =
                RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
            inputMotionVectors.enableRandomWrite = true;
            inputMotionVectors.Create();
            Graphics.Blit(motionVectors, inputMotionVectors);

            RenderTexture inputNormals =
                RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
            inputNormals.enableRandomWrite = true;
            inputNormals.Create();
            Graphics.Blit(normals, inputNormals);

            RenderTexture inputDepth =
                RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
            inputDepth.enableRandomWrite = true;
            inputDepth.Create();
            Graphics.Blit(depth, inputDepth);

            RenderTexture inputAO =
                RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
            inputAO.enableRandomWrite = true;
            inputAO.Create();
            Graphics.Blit(ao, inputAO);

            styleTransferTemporalShader.SetTexture(kernelHandle, StyleTransferTemporalShaderResult, temporary);
            styleTransferTemporalShader.SetTexture(kernelHandle, StyleTransferTemporalShaderStylizedImage, imageStylized);
            styleTransferTemporalShader.SetTexture(kernelHandle, StyleTransferTemporalShaderPreviousStylizedImage, inputTexture);
            styleTransferTemporalShader.SetTexture(kernelHandle, StyleTransferTemporalShaderMotionVectors, inputMotionVectors);
            styleTransferTemporalShader.SetTexture(kernelHandle, StyleTransferTemporalShaderNormalMap, inputNormals);
            styleTransferTemporalShader.SetTexture(kernelHandle, StyleTransferTemporalShaderDepthMap, inputDepth);
            styleTransferTemporalShader.SetTexture(kernelHandle, StyleTransferTemporalShaderOcclusion, inputAO);
            
            styleTransferTemporalShader.Dispatch(kernelHandle, 1920, 1080, 1);

            Graphics.Blit(temporary, imageStylized);

            RenderTexture.ReleaseTemporary(temporary);
            RenderTexture.ReleaseTemporary(inputTexture);
            RenderTexture.ReleaseTemporary(inputMotionVectors);
            RenderTexture.ReleaseTemporary(inputNormals);
            RenderTexture.ReleaseTemporary(inputDepth);
            RenderTexture.ReleaseTemporary(inputAO);
        }

        #endregion

        # region StyleTransferShader

        private enum StyleTransferShaderStep
        {
            PrepareInput,
            FinalizeOutput
        }

        private static readonly int StyleTransferShaderInputImage = Shader.PropertyToID("InputImage");
        private static readonly int StyleTransferShaderOutputImage = Shader.PropertyToID("OutputImage");
        
        // Prepares or finalizes the image for style transfer using the selected shader step.
        private void ExecuteStyleTransferShader(RenderTexture inputImage, StyleTransferShaderStep step)
        {
            const int threads = 8;
            string kernelName = step.ToString();

            RenderTexture temporary =
                RenderTexture.GetTemporary(inputImage.width, inputImage.height, 24, RenderTextureFormat.ARGBHalf);
            temporary.enableRandomWrite = true;
            temporary.Create();

            int kernelHandle = styleTransferShader.FindKernel(kernelName);
            styleTransferShader.SetTexture(kernelHandle, StyleTransferShaderInputImage, inputImage);
            styleTransferShader.SetTexture(kernelHandle, StyleTransferShaderOutputImage, temporary);
            styleTransferShader.Dispatch(kernelHandle, temporary.width / threads, temporary.width / threads, 1);

            Graphics.Blit(temporary, inputImage);
            RenderTexture.ReleaseTemporary(temporary);
        }

        # endregion

        #region Utilities

        private void ClearRenderTexture(RenderTexture renderTexture)
        {
            RenderTexture rt = RenderTexture.active;
            RenderTexture.active = renderTexture;
            GL.Clear(true, true, Color.clear);
            RenderTexture.active = rt;
        }

        void SyncRenderTextureAspect(RenderTexture rt, Camera camera)
        {
            if (rt == null)
            {
                return;
            }

            float aspect = rt.width / (float)rt.height;

            if (!Mathf.Approximately(aspect, camera.aspect))
            {
                rt.Release();
                rt.width = camera.pixelWidth;
                rt.height = camera.pixelHeight;
                rt.Create();
            }
        }

        #endregion
    }
}