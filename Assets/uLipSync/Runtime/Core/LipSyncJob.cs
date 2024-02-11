using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Burst;

namespace uLipSync
{

[BurstCompile]
public struct LipSyncJob : IJob
{
    public struct Info
    {
        public float volume;
        public int mainPhonemeIndex;
    }

    [ReadOnly] public NativeArray<float> input;
    [ReadOnly] public int startIndex;
    [ReadOnly] public int outputSampleRate;
    [ReadOnly] public int targetSampleRate;
    [ReadOnly] public int melFilterBankChannels;
    [ReadOnly] public int mfccNum;
    [ReadOnly] public int deltaMfccNum;
    [ReadOnly] public CompareMethod compareMethod;
    [ReadOnly] public NativeArray<float> means;
    [ReadOnly] public NativeArray<float> standardDeviations;
    [ReadOnly] public NativeArray<float> phonemes;
    [ReadOnly] public int mfccBufferIndex;
    public NativeArray<float> mfcc;
    public NativeArray<float> mfccBuffer;
    public NativeArray<float> scores;
    public NativeArray<Info> info;
    
#if ULIPSYNC_DEBUG
    public NativeArray<float> debugData;
    public NativeArray<float> debugSpectrum;
    public NativeArray<float> debugMelSpectrum;
    public NativeArray<float> debugMelCepstrum;
#endif

    int cutoff => targetSampleRate / 2;
    int range => 500;

    public void Execute()
    {
        float volume = Algorithm.GetRMSVolume(input);

        Algorithm.CopyRingBuffer(input, out var buffer, startIndex);
        Algorithm.LowPassFilter(ref buffer, outputSampleRate, cutoff, range);
        Algorithm.DownSample(buffer, out var data, outputSampleRate, targetSampleRate);
        Algorithm.PreEmphasis(ref data, 0.97f);
        Algorithm.HammingWindow(ref data);
        Algorithm.Normalize(ref data, 1f);
        Algorithm.FFT(data, out var spectrum);
        Algorithm.MelFilterBank(spectrum, out var melSpectrum, targetSampleRate, melFilterBankChannels);
        Algorithm.PowerToDb(ref melSpectrum);
        Algorithm.DCT(melSpectrum, out var melCepstrum);

        NativeArray<float>.Copy(melCepstrum, 1, mfcc, 0, mfccNum);

        if (deltaMfccNum > 0) {
            Algorithm.Delta(mfcc, mfccNum, mfccBuffer, mfccBufferIndex, out var deltaMfcc, deltaMfccNum);
            NativeArray<float>.Copy(deltaMfcc, 0, mfcc, mfccNum, mfccNum);
            deltaMfcc.Dispose();
        }

        CalcScores();

        info[0] = new Info()
        {
            volume = volume,
            mainPhonemeIndex = GetVowel(),
        };
        
#if ULIPSYNC_DEBUG
        data.CopyTo(debugData);
        spectrum.CopyTo(debugSpectrum);
        melSpectrum.CopyTo(debugMelSpectrum);
        melCepstrum.CopyTo(debugMelCepstrum);
#endif

        buffer.Dispose();
        data.Dispose();
        spectrum.Dispose();
        melSpectrum.Dispose();
        melCepstrum.Dispose();
    }

    void CalcScores()
    {
        float sum = 0f;
        
        for (int i = 0; i < scores.Length; ++i)
        {
            float score = CalcScore(i);
            scores[i] = score;
            sum += score;
        }
        
        for (int i = 0; i < scores.Length; ++i)
        {
            scores[i] = sum > 0 ? scores[i] / sum : 0f;
        }
    }

    float CalcScore(int index)
    {
        switch (compareMethod)
        {
            case CompareMethod.L1Norm:
                return CalcL1NormScore(index);
            case CompareMethod.L2Norm:
                return CalcL2NormScore(index);
            case CompareMethod.CosineSimilarity:
                return CalcCosineSimilarityScore(index);
        }
        return 0f;
    }

    float CalcL1NormScore(int index)
    {
        int n = mfcc.Length;
        var phoneme = new NativeSlice<float>(phonemes, index * n, n);
        
        var distance = 0f;
        for (int i = 0; i < n; ++i)
        {
            float x = (mfcc[i] - means[i]) / standardDeviations[i];
            float y = (phoneme[i] - means[i]) / standardDeviations[i];
            distance += math.abs(x - y);
        }
        distance /= n;

        return math.pow(10f, -distance);
    }

    float CalcL2NormScore(int index)
    {
        int n = mfcc.Length;
        var phoneme = new NativeSlice<float>(phonemes, index * n, n);
        
        var distance = 0f;
        for (int i = 0; i < n; ++i)
        {
            float x = (mfcc[i] - means[i]) / standardDeviations[i];
            float y = (phoneme[i] - means[i]) / standardDeviations[i];
            distance += math.pow(x - y, 2f);
        }
        distance = math.sqrt(distance / n);

        return math.pow(10f, -distance);
    }

    float CalcCosineSimilarityScore(int index)
    {
        int n = mfcc.Length;
        var phoneme = new NativeSlice<float>(phonemes, index * n, n);
        float mfccNorm = 0f;
        float phonemeNorm = 0f;
        
        float prod = 0f;
        for (int i = 0; i < n; ++i)
        {
            float x = (mfcc[i] - means[i]) / standardDeviations[i];
            float y = (phoneme[i] - means[i]) / standardDeviations[i];
            mfccNorm += x * x;
            phonemeNorm += y * y;
            prod += x * y;
        }
        mfccNorm = math.sqrt(mfccNorm);
        phonemeNorm = math.sqrt(phonemeNorm);
        float similarity = prod / (mfccNorm * phonemeNorm);
        similarity = math.max(similarity, 0f);

        return math.pow(similarity, 100f);
    }

    int GetVowel()
    {
        int index = -1;
        float maxScore = -1f;
        for (int i = 0; i < scores.Length; ++i)
        {
            var score = scores[i];
            if (score > maxScore)
            {
                index = i;
                maxScore = score;
            }
        }
        return index;
    }
}

}
