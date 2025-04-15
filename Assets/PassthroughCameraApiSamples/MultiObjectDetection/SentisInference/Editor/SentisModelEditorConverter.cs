// Copyright (c) Meta Platforms, Inc. and affiliates.
/*
using Meta.XR.Samples;
using Unity.Sentis;
using UnityEditor;
using UnityEngine;
using FF = Unity.Sentis.Functional;

namespace PassthroughCameraSamples.MultiObjectDetection.Editor
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    [CustomEditor(typeof(SentisInferenceRunManager))]
    public class SentisModelEditorConverter : UnityEditor.Editor
    {
        private const string FILEPATH = "Assets/PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model/sentis-brf.sentis";
        private SentisInferenceRunManager m_targetClass;
        private float m_iouThreshold;
        private float m_scoreThreshold;

        public void OnEnable()
        {
            m_targetClass = (SentisInferenceRunManager)target;
            m_iouThreshold = serializedObject.FindProperty("m_iouThreshold").floatValue;
            m_scoreThreshold = serializedObject.FindProperty("m_scoreThreshold").floatValue;
        }

        public override void OnInspectorGUI()
        {
            _ = DrawDefaultInspector();

            if (GUILayout.Button("Generate Yolov9 Sentis model with Non-Max-Supression layer"))
            {
                OnEnable(); // Get the latest values from the serialized object
                // ConvertModel(); // convert the ONNX model to sentis
                ConvertModelReshape(); // convert the ONNX model to sentis
            }
        }

        private void ConvertModelReshape()
        {
            //Load model
            var model = ModelLoader.Load(m_targetClass.OnnxModel);

            //Here we transform the output of the model by feeding it through a Non-Max-Suppression layer.
            var graph = new FunctionalGraph();
            var input = graph.AddInput(model, 0);

            var centersToCornersData = new[]
            {
                        1,      0,      1,      0,
                        0,      1,      0,      1,
                        -0.5f,  0,      0.5f,   0,
                        0,      -0.5f,  0,      0.5f
            };
            var centersToCorners = FF.Constant(new TensorShape(4, 4), centersToCornersData);
            var modelOutput = FF.Forward(model, input)[0];  //shape(1,N,85)

            // Following for yolo model. in (1, 84, N) out put shape

            var boxCoords = modelOutput[0, ..4, ..].Transpose(0, 1);
            var allScores = modelOutput[0, 4.., ..].Transpose(0, 1);

            var scores = FF.ReduceMax(allScores, 1);    //shape=(N)
            var classIDs = FF.ArgMax(allScores, 1); //shape=(N)


            //
            // matmul error
            //

            int[] shape = new int[] { -1, 4 };
            boxCoords = boxCoords.Reshape(shape); // fuerza shape

            //
            //
            //

            var boxCorners = FF.MatMul(boxCoords, centersToCorners);    //shape=(N,4)

            var indices = FF.NMS(boxCorners, scores, m_iouThreshold, m_scoreThreshold); //shape=(N)
            indices = FF.Reshape(indices, new[] { -1 }); // Forzar a que sea 1D



            var labelIDs = FF.Gather(classIDs, 0, indices); //shape=(N)


            var indices2 = indices.Unsqueeze(-1).BroadcastTo(new[] { 4 });  //shape=(N,4)

            var coords = FF.Gather(boxCoords, 0, indices2); //shape=(N,4)

            var modelFinal = graph.Compile(coords, labelIDs);

            //Export the model to Sentis format
            ModelQuantizer.QuantizeWeights(QuantizationType.Uint8, ref modelFinal);
            ModelWriter.Save(FILEPATH, modelFinal);

            // refresh assets
            AssetDatabase.Refresh();
        }

        //
        // ORIGINAL
        //

        private void ConvertModel()
        {
            //Load model
            var model = ModelLoader.Load(m_targetClass.OnnxModel);

            //Here we transform the output of the model by feeding it through a Non-Max-Suppression layer.
            var graph = new FunctionalGraph();
            var input = graph.AddInput(model, 0);

            var centersToCornersData = new[]
            {
                        1,      0,      1,      0,
                        0,      1,      0,      1,
                        -0.5f,  0,      0.5f,   0,
                        0,      -0.5f,  0,      0.5f
            };
            var centersToCorners = FF.Constant(new TensorShape(4, 4), centersToCornersData);
            var modelOutput = FF.Forward(model, input)[0];  //shape(1,N,85)

            // Following for yolo model. in (1, 84, N) out put shape

            var boxCoords = modelOutput[0, ..4, ..].Transpose(0, 1);
            var allScores = modelOutput[0, 4.., ..].Transpose(0, 1);

            var scores = FF.ReduceMax(allScores, 1);    //shape=(N)
            var classIDs = FF.ArgMax(allScores, 1); //shape=(N)

            var boxCorners = FF.MatMul(boxCoords, centersToCorners);    //shape=(N,4)



            var indices = FF.NMS(boxCorners, scores, m_iouThreshold, m_scoreThreshold); //shape=(N)
            var indices2 = indices.Unsqueeze(-1).BroadcastTo(new[] { 4 });  //shape=(N,4)
            var labelIDs = FF.Gather(classIDs, 0, indices); //shape=(N)
            var coords = FF.Gather(boxCoords, 0, indices2); //shape=(N,4)

            var modelFinal = graph.Compile(coords, labelIDs);

            //Export the model to Sentis format
            ModelQuantizer.QuantizeWeights(QuantizationType.Uint8, ref modelFinal);
            ModelWriter.Save(FILEPATH, modelFinal);

            // refresh assets
            AssetDatabase.Refresh();
        }


    }

    //
    // YOLO BGR
    //

    public static class YoloInputConverter
    {
        public static Tensor<float> ConvertToBGRTensor(Texture2D texture)
        {
            int width = 416;
            int height = 416;

            TextureTransform textureTransform = new TextureTransform();

            textureTransform.SetDimensions(width, height);
            textureTransform.SetChannelSwizzle(ChannelSwizzle.BGRA);

            Tensor<float> tensorTBGRA = TextureConverter.ToTensor(texture, textureTransform);


            return tensorTBGRA;


        }

    }


}
*/


using Meta.XR.Samples;
using Unity.Sentis;
using UnityEditor;
using UnityEngine;
using FF = Unity.Sentis.Functional;

namespace PassthroughCameraSamples.MultiObjectDetection.Editor
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    [CustomEditor(typeof(SentisInferenceRunManager))]
    public class SentisModelEditorConverter : UnityEditor.Editor
    {
        private const string FILEPATH = "Assets/PassthroughCameraApiSamples/MultiObjectDetection/SentisInference/Model/sentis-best-decode.sentis";
        private SentisInferenceRunManager m_targetClass;
        private float m_iouThreshold;
        private float m_scoreThreshold;

        public void OnEnable()
        {
            m_targetClass = (SentisInferenceRunManager)target;
            m_iouThreshold = serializedObject.FindProperty("m_iouThreshold").floatValue;
            m_scoreThreshold = serializedObject.FindProperty("m_scoreThreshold").floatValue;
        }

        public override void OnInspectorGUI()
        {
            _ = DrawDefaultInspector();

            if (GUILayout.Button("Generate Yolov9 Sentis model with Non-Max-Supression layer"))
            {
                OnEnable(); // Get the latest values from the serialized object
                ConvertModel(); // convert the ONNX model to sentis
                // ConvertModelReshape(); // convert the ONNX model to sentis
                // ConvertModelYoloV3(); // convert the ONNX model to sentis
            }
        }

        private void ConvertModelReshape()
        {
            //Load model
            var model = ModelLoader.Load(m_targetClass.OnnxModel);

            //Here we transform the output of the model by feeding it through a Non-Max-Suppression layer.
            var graph = new FunctionalGraph();
            var input = graph.AddInput(model, 0);

            var centersToCornersData = new[]
            {
                        1,      0,      1,      0,
                        0,      1,      0,      1,
                        -0.5f,  0,      0.5f,   0,
                        0,      -0.5f,  0,      0.5f
            };
            var centersToCorners = FF.Constant(new TensorShape(4, 4), centersToCornersData);
            var modelOutput = FF.Forward(model, input)[0];  //shape(1,N,85)

           

            // Following for yolo model. in (1, 84, N) out put shape

            var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1);
            var allScores = modelOutput[0, 4.., ..].Transpose(0, 1);

            var scores = FF.ReduceMax(allScores, 1);    //shape=(N)
            var classIDs = FF.ArgMax(allScores, 1); //shape=(N)

            //
            // matmul error - reshape para solucionarlo
            //

            int[] shape = new int[] { -1, 4 };
            boxCoords = boxCoords.Reshape(shape); // fuerza shape

            //
            // continuar con operaciones
            //

            var boxCorners = FF.MatMul(boxCoords, centersToCorners);    //shape=(N,4)

            var indices = FF.NMS(boxCorners, scores, m_iouThreshold, m_scoreThreshold); //shape=(N)

            // Aseguramos que indices sea 1D
            indices = FF.Reshape(indices, new[] { -1 }); // Forzar a que sea 1D

            // Para la operación de Gather con classIDs
            var labelIDs = FF.Gather(classIDs, 0, indices); //shape=(N)

            // Para la operación que necesita coordenadas 2D, usar método alternativo para el broadcast
            // En lugar de BroadcastTo directo, usamos Tile y luego Reshape
            var indices2 = FF.Reshape(FF.Tile(indices.Unsqueeze(-1), new[] { 1, 4 }), new[] { -1, 4 });
            var coords = FF.Gather(boxCoords, 0, indices2); //shape=(N,4)

            var modelFinal = graph.Compile(coords, labelIDs);

            //Export the model to Sentis format
            ModelQuantizer.QuantizeWeights(QuantizationType.Uint8, ref modelFinal);
            ModelWriter.Save(FILEPATH, modelFinal);

            // refresh assets
            AssetDatabase.Refresh();
        }

        private void ConvertModel()
        {
            //Load model
            var model = ModelLoader.Load(m_targetClass.OnnxModel);

            //Here we transform the output of the model by feeding it through a Non-Max-Suppression layer.
            var graph = new FunctionalGraph();
            var input = graph.AddInput(model, 0);

            var centersToCornersData = new[]
            {
                        1,      0,      1,      0,
                        0,      1,      0,      1,
                        -0.5f,  0,      0.5f,   0,
                        0,      -0.5f,  0,      0.5f
            };
            var centersToCorners = FF.Constant(new TensorShape(4, 4), centersToCornersData);
            var modelOutput = FF.Forward(model, input)[0];  //shape(1,N,85)

            // Following for yolo model. in (1, 84, N) out put shape

            var boxCoords = modelOutput[0, ..4, ..].Transpose(0, 1);
            var allScores = modelOutput[0, 4.., ..].Transpose(0, 1);

            var scores = FF.ReduceMax(allScores, 1);    //shape=(N)
            var classIDs = FF.ArgMax(allScores, 1); //shape=(N)

            var boxCorners = FF.MatMul(boxCoords, centersToCorners);    //shape=(N,4)

            var indices = FF.NMS(boxCorners, scores, m_iouThreshold, m_scoreThreshold); //shape=(N)
            var indices2 = indices.Unsqueeze(-1).BroadcastTo(new[] { 4 });  //shape=(N,4)
            var labelIDs = FF.Gather(classIDs, 0, indices); //shape=(N)
            var coords = FF.Gather(boxCoords, 0, indices2); //shape=(N,4)

            var modelFinal = graph.Compile(coords, labelIDs);

            //Export the model to Sentis format
            ModelQuantizer.QuantizeWeights(QuantizationType.Uint8, ref modelFinal);
            ModelWriter.Save(FILEPATH, modelFinal);

            // refresh assets
            AssetDatabase.Refresh();
        }



        private void ConvertModelYoloV3()
        {



            var rawModel = ModelLoader.Load(m_targetClass.OnnxModel);
            var finalModel = YoloLite.Decode(rawModel);

            ModelWriter.Save(FILEPATH, finalModel);
            AssetDatabase.Refresh();



        }


    }

    /*
    public static class YoloInputConverter
    {
        public static Tensor<float> ConvertToBGRTensor(Texture2D texture)
        {
            int width = 416;
            int height = 416;

            TextureTransform textureTransform = new TextureTransform();

            textureTransform.SetDimensions(width, height);
            textureTransform.SetChannelSwizzle(ChannelSwizzle.BGRA);

            Tensor<float> tensorTBGRA = TextureConverter.ToTensor(texture, textureTransform);

            return tensorTBGRA;
        }
    }
    */


  




    public static class YoloLite
    {
        public static Model Decode(Model rawModel, int gridSize = 13, int numAnchors = 3, int numClasses = 80, float iouThreshold = 0.5f, float scoreThreshold = 0.4f)
        {
            var g = new FunctionalGraph();

            int totalPreds = gridSize * gridSize * numAnchors;  // 13x13x3 = 507
            int bboxData = 5 + numClasses; // 4 bbox + 1 objectness + N classes = 85

            var input = g.AddInput(rawModel, 0); // output del modelo: (1, 255, 13, 13)

            // Reorganizar: (1, 3, 85, 169)
            var reshaped1 = input.Reshape(new[] { 1, numAnchors, bboxData, gridSize * gridSize }); // [1,3,85,169]

            // Transpose(1,2): [1,85,3,169]
            var transposed1 = reshaped1.Transpose(1, 2);

            // Reshape: [1, 85, 507]
            var reshaped2 = transposed1.Reshape(new[] { 1, bboxData, totalPreds });

            // Transpose(1,2): [1,507,85]
            var final = reshaped2.Transpose(1, 2);

            // Cortamos los datos
            var coords = final[0, .., 0..4];        // (N, 4)
            var conf = final[0, .., 4];             // (N)
            var classScores = final[0, .., 5..];    // (N, numClasses)

            var scores = FF.ReduceMax(classScores, dim: 1);    // (N)
            var labels = FF.ArgMax(classScores, dim: 1);       // (N)
            var finalScores = FF.Mul(conf, scores);             // (N)

            // Aplicar NMS
            var indices = FF.NMS(coords, finalScores, iouThreshold, scoreThreshold);
            indices = FF.Reshape(indices, new[] { -1 });

            var coordsGathered = FF.Gather(coords, 0, indices.Unsqueeze(-1).BroadcastTo(new[] { 4 }));
            var labelsGathered = FF.Gather(labels, 0, indices);

            return g.Compile(coordsGathered, labelsGathered);
        }
    }





}