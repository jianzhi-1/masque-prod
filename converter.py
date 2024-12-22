from sparc import load_model
from model import TransformerEmotionModel
import torch
import numpy as np
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.inference.TTS import Tacotron2
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import librosa
import argparse

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

coder = load_model("en", device=device)

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")

def transcript_to_audio(sentence, save_file_name):
    
    mel_output, _, _ = tacotron2.encode_text(sentence)
    # 1. Mel spectrogram with properties in the Tacotron paper (or see get_reconstructed_sample)
    #    Shape = (batch_size, n_mels=80, Mel_length + 1); Mel_length proportional to length of sequence
    # 2. Mel_length = mel_output.shape[2] - 1
    # 3. Alignment
    #    Shape = (batch_size, Mel_length, Token_length) where Token_length is from tacotron2.text_to_seq(txt)

    waveforms = hifi_gan.decode_batch(mel_output) # spectrogram to waveform

    torchaudio.save(save_file_name, waveforms.squeeze(1), 22050)

def predict_unknown(model, text, emotion, filename):

    label_decoder = {
        'confused': 0,
        'default': 1,
        'emphasis': 2,
        'enunciated': 3,
        'essentials': 4,
        'happy': 5,
        'laughing': 6,
        'longform': 7,
        'sad': 8,
        'singing': 9,
        'whisper': 10
    }

    model.eval()
    transcript_to_audio(text, f"{filename}_temp.wav")
    
    waveform, sr = librosa.load(f"{filename}_temp.wav", sr=None)
    assert sr == 22050

    ai_sparc = coder.encode(
        librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    )

    item = {
        "ai_sparc": torch.tensor(ai_sparc["ema"]), 
        "ai_sparc_loudness": torch.tensor(ai_sparc["loudness"]),
        "ai_sparc_pitch": torch.tensor(ai_sparc["pitch"]),
        "ai_sparc_spk_emb": torch.tensor(ai_sparc["spk_emb"])
    }

    ai_mel = pad_sequence(
        [
            torch.cat((
                item["ai_sparc"], 
                (item["ai_sparc_pitch"][:len(item["ai_sparc"])] - 225)/25, 
                torch.log(item["ai_sparc_loudness"][:len(item["ai_sparc"])]) + 2
            ), dim=1)
        ],
        batch_first=True, padding_value=np.nan
    )
    
    mask = torch.all(torch.where(torch.isnan(ai_mel), torch.full(ai_mel.shape, True), torch.full(ai_mel.shape, False)), 2)

    ai_mel = pad_sequence(
        [
            torch.cat((
                item["ai_sparc"], 
                (item["ai_sparc_pitch"][:len(item["ai_sparc"])] - 225)/25, 
                torch.log(item["ai_sparc_loudness"][:len(item["ai_sparc"])]) + 2
            ), dim=1)
        ],
        batch_first=True, padding_value=0.0
    )
    
    batch = {
        "ai_sparc": ai_mel.to(device), 
        "mask": mask.to(device),
        "labels": torch.tensor([label_decoder[emotion]]).to(device)
    }
    
    pred = model.transform(batch)

    pred = pred.squeeze().detach().cpu().numpy()

    pred_ema = pred[:,:12]
    pred_pitch = pred[:, 12]*25 + 225
    pred_loudness = np.exp(pred[:, 13] - 2)

    # Unseen speaker embedding
    spk_emb = np.array([
        -0.59513223, -0.24693543, -0.91295105, -0.08613814,  0.07910099,
        0.453186  ,  1.1357052 , -1.2295437 ,  0.54718304,  0.3908419 ,
       -0.86906415,  1.2277517 , -0.12485051, -1.1065365 , -0.09333476,
       -1.3072228 ,  0.0208655 , -0.7234351 , -0.18774654, -0.9365419 ,
        1.4554019 , -0.75150466,  0.57694477,  0.33537802, -0.59501827,
       -0.13084492, -0.5040275 ,  0.7690312 , -0.23731217,  0.64804363,
       -0.46272534, -0.5939316 , -0.11866839,  0.24177563, -0.29979146,
        1.0454851 , -0.4311453 ,  0.02153815, -0.13319929,  0.8301219 ,
       -0.6667359 , -0.29259   ,  1.0002614 , -0.16082573, -0.74691886,
        0.51362294,  1.0062574 ,  0.04126044, -0.13968004, -0.9094023 ,
        0.83920056,  0.38661447, -0.2562605 , -0.14365867,  0.87786126,
        0.13121217,  0.6104253 ,  0.45328456,  0.12193693,  1.072044  ,
       -1.2453098 , -0.61824876,  0.7683093 , -0.63364744
    ])

    wav_pred = coder.decode(
        ema=pred_ema, 
        pitch=pred_pitch, 
        loudness=pred_loudness, 
        spk_emb=spk_emb
    )

    torchaudio.save(f"{filename}.wav", torch.tensor(wav_pred).unsqueeze(0), coder.sr)

def main():
    
    transformer_encoder_model = TransformerEmotionModel(d_model=512, num_encoder_layers=6, dropout=0.1)
    transformer_encoder_model.to(device)

    transformer_encoder_model.load_state_dict(torch.load("./sparc-model-weights.pt", map_location=device, weights_only=True))

    parser = argparse.ArgumentParser()
    parser.add_argument("transcript", type=str, help="transcript")
    parser.add_argument("label", type=str, help="label")
    parser.add_argument("filename", type=str, help="filename")
    args = parser.parse_args()

    predict_unknown(transformer_encoder_model, args.transcript, args.label, args.filename)


if __name__ == "__main__":
    main()
