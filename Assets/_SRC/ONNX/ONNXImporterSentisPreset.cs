#if UNITY_EDITOR
using Unity.Sentis;
using UnityEditor;
using UnityEditor.AssetImporters;
using UnityEngine;

public class ONNXImporterSentisPreset : AssetPostprocessor
{
    static void OnPostprocessAllAssets(string[] importedAssets, string[] deletedAssets, string[] movedAssets, string[] movedFromPath)
    {
        /*
        foreach (string assetPath in importedAssets)
        {
            if (assetPath.EndsWith(".onnx"))
            {
                Debug.Log($"[Sentis Preset] Procesando modelo ONNX: {assetPath}");

                var asset = AssetDatabase.LoadAssetAtPath<NNModel>(assetPath);
                if (asset == null)
                {
                    Debug.LogWarning($"[Sentis Preset] El archivo no fue importado correctamente como NNModel: {assetPath}");
                    continue;
                }

                // Opcional: Forzar reimportación para regenerar el asset .sentis
                AssetDatabase.ImportAsset(assetPath, ImportAssetOptions.ForceUpdate);

                Debug.Log($"[Sentis Preset] Modelo ONNX convertido y disponible como NNModel.");
            }
        }
        */
    }
}
#endif