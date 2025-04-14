using UnityEngine;
using Unity.Sentis;

public class AzureModelRunner : MonoBehaviour
{
    /*
    
    public NNModel modelAsset;
    private Model model;
    private IWorker worker;

    void Start()
    {
        // Cargar modelo
        model = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);

        // CREAR INPUT SEGÚN MODELO DE AZURE (normalmente 1x3x224x224 para imagenes)
        // Por ahora solo una imagen vacía para probar
        TensorFloat input = new TensorFloat(new TensorShape(1, 3, 224, 224));

        // Ejecutar
        worker.Execute(input);

        // Obtener la salida
        TensorFloat output = worker.PeekOutput() as TensorFloat;
        Debug.Log("Output length: " + output.length);
        Debug.Log("Primer valor: " + output[0]);

        // Liberar recursos
        input.Dispose();
        output.Dispose();
    }

    void OnDestroy()
    {
        worker.Dispose();
    }
    */
}