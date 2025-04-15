using Unity.Sentis;
using UnityEngine;

public class OnnxRunner : MonoBehaviour
{

    /*
        public NNModel modelAsset; // Arrastra aquí el asset .onnx importado desde el inspector

        IWorker worker;
        Model model;

        void Start()
        {
            model = ModelLoader.Load(modelAsset);
            worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);

            // Crear un tensor de entrada según lo que espera tu modelo
            // Por ejemplo: un tensor de 1x3x224x224 para una imagen RGB 224x224
            Tensor input = new Tensor(1, 3, 224, 224);

            // Llenar el tensor con datos reales aquí...

            worker.Execute(input);

            Tensor output = worker.PeekOutput();

            Debug.Log("Resultado del modelo: " + output[0]);

            // Libera recursos
            input.Dispose();
            output.Dispose();
        }

        void OnDestroy()
        {
            worker.Dispose();
        }

    */

    
}
