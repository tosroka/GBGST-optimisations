using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;
using Unity.Barracuda;
using UnityEditor;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering.RendererUtils;
using UnityEngine.Rendering;
using System;

/*#if UNITY_EDITOR
    using UnityEditor;
    using UnityEditor.Rendering.HighDefinition;

    [CustomPassDrawer(typeof(CopyPass))]
    public class CopyPassDrawer : CustomPassDrawer
    {
        protected override PassUIFlag commonPassUIFlags => PassUIFlag.Name;
    }
#endif*/

public class CopyPassStylizationEffects : CustomPass
{

    public NNModel modelAsset; // Reference to the Barracuda neural network asset
    public ComputeShader styleTransferShader; // Reference to the compute shader responsible for applying the style transfer
    public ComputeShader styleTransferOutputShader;

    [Tooltip("The height og the image being fed to the model")]
    public int targetHeight;
    private Model styleTransferModel;
    private IWorker styleTransferWorker;
    // The inference used to execute the neural network
    private IWorker engine;
    [Tooltip("The backend used when performing inference")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;


    public enum BufferType
    {
        Color,
        Normal,
        Roughness,
        Depth,
        MotionVectors,
    }

    public RenderTexture outputRenderTexture;

    public BufferType bufferType;

    Shader fullscreenShader;
    public Material fullscreenMaterial;
    RenderTargetIdentifier diffuseTextureIdentifier = new RenderTargetIdentifier("_GBufferTexture0");

    protected override bool executeInSceneView => true;

    int normalPass;
    int roughnessPass;
    int depthPass;
    private RenderTexture diffuseTexture;
    public ComputeShader gbufferShader;
    Material gbufferMaterial;

    //private Texture diffuseTxt;
    private RenderTexture motionVectorsTexture;
    private RenderTexture normalsTexture;
    private RenderTexture previousNormalsTexture;
    private RenderTexture depthTexture;
    private RenderTexture ambientOcclusionTexture;
    public LayerMask renderingMask = -1;

    RenderTexture previousStylized;
    bool hasPreviousStylizedBeenSet = false;

    public void ClearRenderTexture(RenderTexture renderTexture)
    {
        RenderTexture rt = RenderTexture.active;
        RenderTexture.active = renderTexture;
        GL.Clear(true, true, Color.clear);
        RenderTexture.active = rt;
    }

    protected override void Setup(ScriptableRenderContext renderContext, CommandBuffer cmd)
    {
        if (modelAsset != null)
            styleTransferModel = ModelLoader.Load(modelAsset);

        if (styleTransferModel != null)
            styleTransferWorker = WorkerFactory.CreateWorker(styleTransferModel);


        // initialize inference engine
        engine = WorkerFactory.CreateWorker(workerType, styleTransferModel);

        // if (gbufferShader == null)
        gbufferShader = Resources.Load<ComputeShader>("GbufferShader");

        if (fullscreenShader == null)
            fullscreenShader = Shader.Find("FullScreen/Fullscreen_NST");
        // fullscreenMaterial = CoreUtils.CreateEngineMaterial(fullscreenShader);

        previousStylized = new RenderTexture(1920, 1080, 0, RenderTextureFormat.ARGB32);
        previousStylized.enableRandomWrite = true;
        previousStylized.Create();

        previousNormalsTexture = new RenderTexture(1920, 1080, 0, RenderTextureFormat.ARGB32);
        previousNormalsTexture.enableRandomWrite = true;
        previousStylized.Create();

        hasPreviousStylizedBeenSet = false;
    }

    protected override void Execute(CustomPassContext ctx)
    {
        if (outputRenderTexture == null)
            return;

        SyncRenderTextureAspect(outputRenderTexture, ctx.hdCamera.camera);

        var scale = RTHandles.rtHandleProperties.rtHandleScale;

        diffuseTexture = ctx.hdCamera.GetCurrentFrameRT((int)HDCameraFrameHistoryType.ColorBufferMipChain);
        // diffuseTexture = ctx.cameraColorBuffer.rt;
        // int kernelHandle = gbufferShader.FindKernel("CSMain");
        // gbufferShader.SetTextureFromGlobal(kernelHandle, "_GBufferTexture0", "_GBufferTexture0");
        //ctx.cmd.DispatchCompute(gbufferShader, kernelHandle, (int)scale.x,(int) scale.y, 1);

        
        // Texture diffuseTxt = Shader.GetGlobalTexture("_GBufferTexture0");
        // //diffuseTexture = RenderTexture.GetTemporary(1920, 1080, 16, RenderTextureFormat.ARGB32);

        // Graphics.Blit(diffuseTxt, diffuseTexture);
        SyncRenderTextureAspect(diffuseTexture, ctx.hdCamera.camera);

        if (!ctx.hdCamera.frameSettings.IsEnabled(FrameSettingsField.ObjectMotionVectors) ||
            !ctx.hdCamera.frameSettings.IsEnabled(FrameSettingsField.OpaqueObjects))
        {
            Debug.Log("Motion Vectors are disabled on the camera!");
            return;
        }
        // clean up textures
        ClearRenderTexture(normalsTexture);
        ClearRenderTexture(previousNormalsTexture);
        ClearRenderTexture(motionVectorsTexture);
        ClearRenderTexture(depthTexture);
        ClearRenderTexture(ambientOcclusionTexture);
        

        // motion vector texture
        ctx.cmd.Blit(ctx.cameraMotionVectorsBuffer, motionVectorsTexture, new Vector2(scale.x, scale.y), Vector2.zero, 0, 0);
        normalsTexture = ctx.cameraNormalBuffer;
        depthTexture = ctx.cameraDepthBuffer; // ctx.hdCamera.GetCurrentFrameRT((int)HDCameraFrameHistoryType.Depth);
        ambientOcclusionTexture = ctx.hdCamera.GetCurrentFrameRT((int)HDCameraFrameHistoryType.AmbientOcclusion);

        SyncRenderTextureAspect(depthTexture, ctx.hdCamera.camera);
        // get the motion vectors texture
        // motionVectorsTexture = RenderTexture.GetTemporary(1920, 1080, 24, RenderTextureFormat.ARGB32);
        // motionVectorsTexture = ctx.cameraNormalBuffer; //  ctx.hdCamera.GetCurrentFrameRT((int)HDCameraFrameHistoryType.Normal);
        // motionVectorsTexture = ctx.hdCamera.GetCurrentFrameRT((int)HDCameraFrameHistoryType.AmbientOcclusion);
        // ctx.cmd.Blit(ctx.cameraMotionVectorsBuffer, motionVectorsTexture, new Vector2(scale.x, scale.y), Vector2.zero, 0, 0);
        // motionVectorsTexture = ctx.cameraDepthBuffer.rt;

        /* Texture2D motionVectorsTexture2D = new Texture2D(motionVectorsTexture.width, motionVectorsTexture.height, TextureFormat.RGBA32, false);
         RenderTexture motionVectorsTexture2DpreviousActive = RenderTexture.active;
         RenderTexture.active = (RenderTexture)motionVectorsTexture;
         motionVectorsTexture2D.ReadPixels(new Rect(0, 0, motionVectorsTexture.width, motionVectorsTexture.height), 0, 0);
         motionVectorsTexture2D.Apply();
         RenderTexture.active = motionVectorsTexture2DpreviousActive;

         // // Save the texture as an asset in the project's Assets folder
         byte[] bytes = motionVectorsTexture2D.EncodeToPNG();
         string path = "Assets/Stylized/motionVectorsTexture.png";
         System.IO.File.WriteAllBytes(path, bytes);
         AssetDatabase.ImportAsset(path);
         TextureImporter importer = (TextureImporter)AssetImporter.GetAtPath(path);
         importer.sRGBTexture = true;
         importer.alphaSource = TextureImporterAlphaSource.None;
         importer.alphaIsTransparency = false;
         importer.SaveAndReimport();*/

        // SyncRenderTextureAspect(diffuseTexture, ctx.hdCamera.camera);
        //RenderTexture diffuseTexture = new RenderTexture(1920, 1080, 16, RenderTextureFormat.ARGB32);
        // ctx.cmd.Blit(diffuseTexture, outputRenderTexture, new Vector2(scale.x, scale.y), Vector2.zero, 0, 0);
        // ctx.cmd.Blit(ctx.cameraColorBuffer, diffuseTexture, new Vector2(scale.x, scale.y), Vector2.zero, 0, 0);

        //SyncRenderTextureAspect(diffuseTexture, ctx.hdCamera.camera);
        StylizeImage(diffuseTexture, previousStylized, motionVectorsTexture, normalsTexture, depthTexture, ambientOcclusionTexture,  hasPreviousStylizedBeenSet);
        Graphics.Blit(diffuseTexture, previousStylized); // , new Vector2(scale.x, scale.y), Vector2.zero, 0, 0);
        Graphics.Blit(normalsTexture, previousNormalsTexture);
        //Graphics.Blit(diffuseTexture, outputRenderTexture);
        ctx.cmd.Blit(diffuseTexture, outputRenderTexture, new Vector2(scale.x, scale.y), Vector2.zero, 0, 0);
        // int kernelHandle = gbufferShader.FindKernel("CSMain");
        // Debug.Log(kernelHandle);
        Texture2D texture2D = new Texture2D(diffuseTexture.width, diffuseTexture.height, TextureFormat.RGBA32, false);
        RenderTexture previousActive = RenderTexture.active;
        RenderTexture.active = (RenderTexture)diffuseTexture;
        texture2D.ReadPixels(new Rect(0, 0, diffuseTexture.width, diffuseTexture.height), 0, 0);
        texture2D.Apply();
        RenderTexture.active = previousActive;
        //gbufferShader.SetTextureFromGlobal(0, "_GBufferTexture0", "_GBufferTexture0");
        gbufferShader.SetTexture(0, "_GBufferTexture0", ctx.cameraColorBuffer.rt);
        gbufferShader.SetTexture(0, "_StylizedTexture", texture2D);
        ctx.cmd.DispatchCompute(gbufferShader, 0, 64, 64, 1);
        // ctx.cmd.Blit(outputRenderTexture, ctx.cameraColorBuffer);

      
    }


    void SyncRenderTextureAspect(RenderTexture rt, Camera camera)
    {
        float aspect = rt.width / (float)rt.height;

        if (!Mathf.Approximately(aspect, camera.aspect))
        {
            rt.Release();
            rt.width = camera.pixelWidth;
            rt.height = camera.pixelHeight;
            rt.Create();
        }
    }

    protected override void Cleanup()
    {
        base.Cleanup();
        
        if (styleTransferWorker != null)
            styleTransferWorker.Dispose();


        if (previousStylized != null)
            previousStylized.Release();

        if (motionVectorsTexture != null)
        {
            motionVectorsTexture.Release();
            normalsTexture.Release();
            depthTexture.Release();
            ambientOcclusionTexture.Release();
        }

        // destroy things
        CoreUtils.Destroy(fullscreenMaterial);
        if (styleTransferWorker != null)
        {
            styleTransferWorker.Dispose();
            styleTransferWorker = null;
        }

        if (styleTransferModel != null)
        {
            styleTransferModel = null;
        }

        if (previousStylized != null)
        {
            previousStylized.Release();
            previousStylized = null;
        }

        if (previousNormalsTexture != null)
        {
            previousNormalsTexture.Release();
            previousNormalsTexture = null;
        }

        GC.Collect();
        Resources.UnloadUnusedAssets();
        EditorUtility.UnloadUnusedAssetsImmediate();



    }


    private void StylizeImage(RenderTexture src, RenderTexture previousStylized, RenderTexture mVectorsTexture, RenderTexture nTexture, RenderTexture dTexture, RenderTexture aoTexture, bool hasPreviousStylizedSet)
    {
        RenderTexture source;
        source = RenderTexture.GetTemporary(src.width, src.height, 24, src.format);

        // Copy the src RenderTexture to the new rTex RenderTexture
        Graphics.Blit(src, source);
        // Apply preprocessing steps
        // ProcessImage(source, "ProcessInput");

        // create a new RenderTexture variable 
        RenderTexture rTex;

        // Assign a temporary RenderTexture with the src dimensions
        rTex = RenderTexture.GetTemporary(src.width, src.height, 24, src.format);

        // Copy the src RenderTexture to the new rTex RenderTexture
        Graphics.Blit(src, rTex);

        // Apply preprocessing steps
        ProcessImage(rTex, "ProcessInput");

        // Create a Tensor of shape [1, rTex.height, rTex.width, 3]
        Tensor input = new Tensor(rTex, channels: 3);

        engine.Execute(input);

        // Get the raw model output
        Tensor prediction = engine.PeekOutput();
        // Release GPU resources allocated for the Tensor
        input.Dispose();

        // Make sure rTex is not the active RenderTexture
        RenderTexture.active = null;
        // Copy the model output to rTex
        prediction.ToRenderTexture(rTex);
        // Release GPU resources allocated for the Tensor
        prediction.Dispose();

        if (hasPreviousStylizedSet)
            ProcessImageMotionVectors(previousStylized, rTex, mVectorsTexture, nTexture, dTexture, source, "ProcessOutput");
        else ProcessImage(rTex, "ProcessOutput");

        // Copy rTex into src
        // Graphics.Blit(rTex, previousStylized);
        hasPreviousStylizedBeenSet = true;

        // Copy rTex into src
        Graphics.Blit(rTex, source);

        // Copy rTex into src
        Graphics.Blit(source, src);

        // Release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(rTex);
        RenderTexture.ReleaseTemporary(source);
    }


    private void ProcessImageMotionVectors(RenderTexture image, RenderTexture imageStylized, RenderTexture motionVectors, RenderTexture normals, RenderTexture depth, RenderTexture ao, string functionName)
    {
        // number of threads on the GPU
        int numthreads = 8;
        // Get the index of the specified function in the ComputeShader
        int kernelHandle = styleTransferOutputShader.FindKernel(functionName);
        //temporary HDR RenderTexture
        RenderTexture result = RenderTexture.GetTemporary(imageStylized.width, imageStylized.height, 24, RenderTextureFormat.ARGBHalf);// RenderTextureFormat.ARGBHalf);
        // Enable random write access
        result.enableRandomWrite = true;
        // create the HDR RenderTexture
        result.Create();

        //temporary HDR RenderTexture
        RenderTexture input = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
        input.enableRandomWrite = true;
        input.Create();
        Graphics.Blit(image, input);

        //temporary HDR RenderTexture
        RenderTexture inputmotionVectors = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);// RenderTextureFormat.ARGBHalf);
        inputmotionVectors.enableRandomWrite = true;
        inputmotionVectors.Create();
        Graphics.Blit(motionVectors, inputmotionVectors);

        //temporary HDR RenderTexture
        RenderTexture inputNormals = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);// RenderTextureFormat.ARGBHalf);
        inputNormals.enableRandomWrite = true;
        inputNormals.Create();
        Graphics.Blit(normals, inputNormals);

        //temporary HDR RenderTexture
        RenderTexture inputDepth = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);// RenderTextureFormat.ARGBHalf);
        inputDepth.enableRandomWrite = true;
        inputDepth.Create();
        Graphics.Blit(depth, inputDepth);

        //temporary HDR RenderTexture
        RenderTexture inputAO = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);// RenderTextureFormat.ARGBHalf);
        inputAO.enableRandomWrite = true;
        inputAO.Create();
        Graphics.Blit(ao, inputAO);

        // set the value for the Result variable in the ComputeShader
        styleTransferOutputShader.SetTexture(kernelHandle, "Result", result);
        // set the value for the InputImage variable in the ComputeShader
        styleTransferOutputShader.SetTexture(kernelHandle, "StylizedImage", imageStylized);
        // set the value for the InputImage variable in the ComputeShader
        styleTransferOutputShader.SetTexture(kernelHandle, "PreviousStylizedImage", input);
        styleTransferOutputShader.SetTexture(kernelHandle, "MotionVectors", inputmotionVectors);
        styleTransferOutputShader.SetTexture(kernelHandle, "NormalMap", inputNormals);
        styleTransferOutputShader.SetTexture(kernelHandle, "DepthMap", inputDepth);
        styleTransferOutputShader.SetTexture(kernelHandle, "AmbientOcclusion", inputAO);

        // execute the ComputeShader
        styleTransferOutputShader.Dispatch(kernelHandle, 1920, 1080, 1);
        // copy the result into the source RenderTexture
        Graphics.Blit(result, imageStylized);
        // release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(input);
        RenderTexture.ReleaseTemporary(inputmotionVectors);
        RenderTexture.ReleaseTemporary(inputNormals);
        RenderTexture.ReleaseTemporary(inputDepth);
        RenderTexture.ReleaseTemporary(inputAO);

    }

    /// <summary>]/// Process the provided image using the specified function on the GPU
    /// </summary>
    /// <param name="image"></param>
    /// <param name="functionName"></param>
    /// <returns>The processed image</returns>
    private void ProcessImage(RenderTexture image, string functionName)
    {
        // number of threads on the GPU
        int numthreads = 8;
        // Get the index of the specified function in the ComputeShader
        int kernelHandle = styleTransferShader.FindKernel(functionName);
        //temporary HDR RenderTexture
        RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);// RenderTextureFormat.ARGBHalf);
        // Enable random write access
        result.enableRandomWrite = true;
        // create the HDR RenderTexture
        result.Create();

        // set the value for the Result variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "Result", result);
        // set the value for the InputImage variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "InputImage", image);

        // execute the ComputeShader
        styleTransferShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);
        // copy the result into the source RenderTexture
        Graphics.Blit(result, image);
        // release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(result);

    }


    void PrintTensorValues(Tensor tensor)
    {
        float[] data = tensor.ToReadOnlyArray();

        int width = tensor.width;
        int height = tensor.height;
        int channels = tensor.channels;

        for (int c = 0; c < channels/3; c++)
        {
            for (int y = 0; y < height/100; y++)
            {
                for (int x = 0; x < width/120; x++)
                {
                    float value = data[x + y * width + c * width * height];
                    Debug.Log($"Channel: {c}, X: {x}, Y: {y}, Value: {value}");
                }
            }
        }
    }
}