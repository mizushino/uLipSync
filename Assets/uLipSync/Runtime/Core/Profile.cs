using UnityEngine;
using Unity.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.Mathematics;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace uLipSync
{

[System.Serializable]
public struct MfccCalibrationData
{
    public float[] array;
    public float this[int i] => array[i];
    public int length => array.Length;
}

[System.Serializable]
public class MfccData
{
    public string name;
    public List<MfccCalibrationData> mfccCalibrationDataList = new List<MfccCalibrationData>();
    public NativeArray<float> mfccNativeArray;

    public MfccData(string name)
    {
        this.name = name;
    }

    ~MfccData()
    {
        Deallocate();
    }

    public void Allocate(int length)
    {
        if (IsAllocated())
        {
            if (mfccNativeArray.Length == length)
            {
                return;
            }
            Deallocate();
        }

        mfccNativeArray = new NativeArray<float>(length, Allocator.Persistent);
    }

    public void Deallocate()
    {
        if (!IsAllocated()) return;

        mfccNativeArray.Dispose();
    }

    bool IsAllocated()
    {
        return mfccNativeArray.IsCreated;
    }

    public void AddCalibrationData(float[] mfcc)
    {
        if (mfcc.Length != 12 && mfcc.Length != 24 && mfcc.Length != 36)
        {
            Debug.LogError("The length of MFCC array should be 12 or 24 or 36.");
            return;
        }

        if (mfccCalibrationDataList.Count > 0 && mfcc.Length != mfccCalibrationDataList[0].length)
        {
            mfccCalibrationDataList.Clear();
        }

        mfccCalibrationDataList.Add(new MfccCalibrationData() { array = mfcc });
    }

    public void RemoveOldCalibrationData(int dataCount)
    {
        while (mfccCalibrationDataList.Count > dataCount) mfccCalibrationDataList.RemoveAt(0);
    }
    
    public void UpdateNativeArray()
    {
        if (mfccCalibrationDataList.Count == 0) return;
        if (mfccNativeArray.Length != mfccCalibrationDataList[0].length) return;

        for (int i = 0; i < mfccNativeArray.Length; ++i)
        {
            mfccNativeArray[i] = 0f;
            foreach (var mfcc in mfccCalibrationDataList)
            {
                mfccNativeArray[i] += mfcc[i];
            }
            mfccNativeArray[i] /= mfccCalibrationDataList.Count;
        }
    }

    public float GetAverage(int i)
    {
        return mfccNativeArray[i];
    }
}

[CreateAssetMenu(menuName = Common.AssetName + "/Profile")]
public class Profile : ScriptableObject
{
    [HideInInspector] public string jsonPath = "";

    [Tooltip("The number of MFCC")]
    public int mfccNum = 12;
    [Tooltip("The number of MFCC data to calculate the average MFCC values")]
    public int mfccDataCount = 16;
    [Tooltip("The number of Mel Filter Bank channels")]
    public int melFilterBankChannels = 30;
    [Tooltip("Target sampling rate to apply downsampling")]
    public int targetSampleRate = 16000;
    [Tooltip("Number of audio samples after downsampling is applied")]
    public int sampleCount = 1024;
    [Tooltip("Whether to perform standardization of each coefficient of MFCC")] 
    public bool useStandardization = false;
    [Tooltip("The comparison method for MFCC")]
    public CompareMethod compareMethod = CompareMethod.L2Norm;
    [Tooltip("The number of delta MFCC")]
    public int deltaMfccNum = 0;
    [Tooltip("The number of delta delta MFCC")]
    public int deltaDeltaMfccNum = 0;

    public List<MfccData> mfccs = new List<MfccData>();
    
    float[] _means;
    float[] _stdDevs;
    public float[] means => _means; 
    public float[] standardDeviation => _stdDevs; 

    public int mfccLength => mfccNum * (1 + ((deltaMfccNum > 0) ? 1 : 0) + ((deltaDeltaMfccNum > 0) ? 1 : 0));

    void OnEnable()
    {
        UpdateMeansAndStandardization();

        foreach (var data in mfccs)
        {
            data.Allocate(mfccLength);
            data.RemoveOldCalibrationData(mfccDataCount);
            data.UpdateNativeArray();
        }
    }

    void OnDisable()
    {
        foreach (var data in mfccs)
        {
            data.Deallocate();
        }
    }

    public string GetPhoneme(int index)
    {
        if (index < 0 || index >= mfccs.Count) return "";
        
        return mfccs[index].name;
    }

    public void AddMfcc(string name)
    {
        var data = new MfccData(name);
        data.Allocate(mfccLength);
        for (int i = 0; i < mfccDataCount; ++i)
        {
            data.AddCalibrationData(new float[mfccLength]);
        }
        mfccs.Add(data);
    }

    public void RemoveMfcc(int index)
    {
        if (index < 0 || index >= mfccs.Count) return;
        
        var data = mfccs[index];
        data.Deallocate();
        mfccs.RemoveAt(index);
        
        UpdateMeansAndStandardization();
    }

    public void UpdateMfcc(int index, NativeArray<float> mfcc, bool calib)
    {
        if (index < 0 || index >= mfccs.Count) return;

        var array = new float[mfcc.Length];
        mfcc.CopyTo(array);

        var data = mfccs[index];
        data.AddCalibrationData(array);
        data.RemoveOldCalibrationData(mfccDataCount);

        if (calib)
        {
            data.UpdateNativeArray();
            UpdateMeansAndStandardization();
        }
    }

    public NativeArray<float> GetAverages(int index)
    {
        return mfccs[index].mfccNativeArray;
    }

    public bool Export(string path)
    {
        var json = JsonUtility.ToJson(this);

        try
        {
            File.WriteAllText(path, json);
        }
        catch (System.Exception e)
        {
            Debug.LogError(e.Message);
            return false;
        }

        return true;
    }

    public bool Import(string path)
    {
        string json = "";

        try
        {
            json = File.ReadAllText(path);
        }
        catch (System.Exception e)
        {
            Debug.LogError(e.Message);
            return false;
        }

        JsonUtility.FromJsonOverwrite(json, this);
        OnEnable();

        return true;
    }

    public string[] GetPhonemeNames()
    {
        return mfccs.Select(x => x.name).Distinct().ToArray();
    }

    public void UpdateMeansAndStandardization()
    {
        UpdateMeans();
        UpdateStandardizations();
    }
    
    void UpdateMeans()
    {
        if (_means == null || _means.Length != mfccLength)
        {
            _means = new float[mfccLength];
        }

        for (int i = 0; i < _means.Length; ++i)
        {
            _means[i] = 0f;
        }

        if (!useStandardization) return;

        int n = 0;
        foreach (var mfccData in mfccs)
        {
            var list = mfccData.mfccCalibrationDataList;
            foreach (var mfcc in list)
            {
                for (int i = 0; i < mfcc.length; ++i)
                {
                    _means[i] += mfcc[i];
                }
                ++n;
            }
        }

        for (int i = 0; i < _means.Length; ++i)
        {
            _means[i] /= n;
        }
    }

    void UpdateStandardizations()
    {
        if (_stdDevs == null || _stdDevs.Length != mfccLength)
        {
            _stdDevs = new float[mfccLength];
        }

        if (!useStandardization)
        {
            for (int i = 0; i < _stdDevs.Length; ++i)
            {
                _stdDevs[i] = 1f;
            }
            return;
        }

        for (int i = 0; i < _stdDevs.Length; ++i)
        {
            _stdDevs[i] = 0f;
        }
        
        int n = 0;
        foreach (var mfccData in mfccs)
        {
            var list = mfccData.mfccCalibrationDataList;
            foreach (var mfcc in list)
            {
                for (int i = 0; i < mfcc.length; ++i)
                {
                    _stdDevs[i] += math.pow(mfcc[i] - _means[i], 2f);
                }
                ++n;
            }
        }
        
        for (int i = 0; i < _stdDevs.Length; ++i)
        {
            _stdDevs[i] = math.sqrt(_stdDevs[i] / n);
        }
    }

    public void CalcMinMax(out float min, out float max)
    {
        max = float.MinValue;
        min = float.MaxValue;
        foreach (var data in mfccs)
        {
            for (int j = 0; j < data.mfccCalibrationDataList.Count; ++j)
            {
                var array = data.mfccCalibrationDataList[j].array;
                foreach (var x in array)
                {
                    max = Mathf.Max(max, x);
                    min = Mathf.Min(min, x);
                }
            }
        }
    }

    public void Save()
    {
#if UNITY_EDITOR
        EditorUtility.SetDirty(this);
        AssetDatabase.SaveAssets();
        Debug.Log($"{name} saved.");
#endif
    }

    public static Profile Create()
    {
        return ScriptableObject.CreateInstance<Profile>();
    }
}

}
