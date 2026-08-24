import csv
import glob
import os

import librosa
import numpy as np
import torch
import soundfile as sf

try:
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    HAS_WAVLM = True
except ImportError:
    HAS_WAVLM = False
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    HAS_RESEMBLYZER = True
except ImportError:
    HAS_RESEMBLYZER = False


WAVLM_MODEL_ID = "microsoft/wavlm-base-sv"
TARGET_SAMPLE_RATE = 16_000
MIN_DUR = 1.0  # minimum duration for an utterance to be considered valid

def get_duration_seconds(wav_path: str) -> float:
    info = sf.info(wav_path)
    return info.frames / info.samplerate

def l2_normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Cannot normalize an invalid or zero embedding.")

    return vector / norm


class SpeakerLabelizer:
    def __init__(self, root_data_path, model_type="wavlm", threshold=None):
        """
        Args:
            root_data_path:
                Path to the IViE root directory.
            model_type:
                Either "wavlm" or "resemblyzer".
            threshold:
                Cosine-similarity threshold below which a candidate is labelled
                as unknown. Thresholds should be calibrated on held-out data.
        """
        self.root_path = os.path.abspath(root_data_path)
        self.model_type = model_type.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        default_thresholds = {
            "wavlm": 0.86,
            "resemblyzer": 0.75,
        }

        if self.model_type not in default_thresholds:
            raise ValueError(f"Unknown model_type {model_type!r}; expected 'wavlm' or 'resemblyzer'.")
        
        self.threshold = (default_thresholds[self.model_type] if threshold is None else float(threshold))

        self.model = None
        self.feature_extractor = None
        self.encoder = None

        self._init_model()

    def _init_model(self):
        """Initialise the selected speaker-embedding model."""
        if self.model_type == "wavlm":
            if not HAS_WAVLM:
                raise ImportError(
                    "WavLM requires transformers. Install it with: "
                    "pip install transformers"
                )

            self.feature_extractor = (
                Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_ID)
            )
            self.model = WavLMForXVector.from_pretrained(
                WAVLM_MODEL_ID
            ).to(self.device)
            self.model.eval()

        elif self.model_type == "resemblyzer":
            if not HAS_RESEMBLYZER:
                raise ImportError(
                    "Resemblyzer is not installed. Install it with: "
                    "pip install resemblyzer"
                )

            self.encoder = VoiceEncoder(device=str(self.device))

        print(
            f"Loaded {self.model_type} on {self.device}; "
            f"threshold={self.threshold:.3f}"
        )

    @staticmethod
    def _load_mono_16khz(wav_path):
        """Load audio as a mono float32 waveform resampled to 16 kHz."""
        waveform, sample_rate = librosa.load(
            wav_path,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
            dtype=np.float32,
        )

        waveform = np.asarray(waveform, dtype=np.float32)

        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Audio is empty or has an invalid shape.")

        if not np.isfinite(waveform).all():
            raise ValueError("Audio contains NaN or infinite samples.")

        if np.max(np.abs(waveform)) == 0:
            raise ValueError("Audio contains only silence.")

        return np.ascontiguousarray(waveform)

    def extract_embedding(self, wav_path):
        """Extract one unit-normalised speaker embedding."""
        try:
            if self.model_type == "wavlm":
                waveform = self._load_mono_16khz(wav_path)

                inputs = self.feature_extractor(
                    waveform,
                    sampling_rate=TARGET_SAMPLE_RATE,
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                }

                with torch.inference_mode():
                    outputs = self.model(**inputs)
                    embedding = torch.nn.functional.normalize(
                        outputs.embeddings,
                        p=2,
                        dim=-1,
                    )

                return embedding[0].cpu().numpy().astype(
                    np.float32,
                    copy=False,
                )

            if self.model_type == "resemblyzer":
                # Resemblyzer's preprocessing performs mono conversion,
                # 16-kHz resampling, volume normalisation and silence trimming.
                waveform = preprocess_wav(wav_path)

                if waveform.size == 0 or not np.isfinite(waveform).all():
                    raise ValueError(
                        "Resemblyzer preprocessing produced invalid audio."
                    )

                embedding = self.encoder.embed_utterance(waveform)
                return l2_normalize(embedding)

            raise RuntimeError(f"Unsupported model type: {self.model_type}")

        except Exception as error:
            print(f"Error processing {wav_path}: {error}")
            return None

    def build_centroid_embeddings(self, task="retold_passages"):
        """Build one unit-normalised centroid per speaker."""

        # scan for all accents
        embeddings_by_accent = {}
        reference_directory = os.path.join(self.root_path, "wav")
        accents = sorted(entry.name for entry in os.scandir(reference_directory) if entry.is_dir())

        for accent in accents:
            print(f"Processing accent: {accent}")
            embeddings_by_accent[accent] = {}
            task_directory = os.path.join(reference_directory, accent, task)
            wav_files = sorted(glob.glob(os.path.join(task_directory, f"{accent}-{task}-*.wav")))

            # group utterances by speaker
            speaker_utterances = {}
            for wav_file in wav_files:
                parts = os.path.basename(wav_file).split("-")
                if task in ["free_conversation", "map_task"]:
                    speaker_id = "-".join(parts[1:4])
                elif task in ["retold_passages"]:
                    speaker_id = parts[-2]
                else:
                    raise ValueError(f"Unexpected task in filename: {wav_file}")
                if get_duration_seconds(wav_file) < MIN_DUR:
                    print(f"Skipping {wav_file} (duration < {MIN_DUR}s)")
                    continue
                speaker_utterances.setdefault(speaker_id, []).append(wav_file)
            
            # calculate the centroid embedding for each speaker
            for speaker_id, utterances in sorted(speaker_utterances.items()):
                utterance_embeddings = []
                for utterance_path in utterances:
                    embedding = self.extract_embedding(utterance_path)
                    if embedding is not None:
                        utterance_embeddings.append(embedding)
                if not utterance_embeddings:
                    continue
                speaker_centroid = l2_normalize(np.mean(np.stack(utterance_embeddings, axis=0),axis=0,))
                embeddings_by_accent[accent][speaker_id] = (speaker_centroid)
                print(f"{speaker_id}: {len(utterance_embeddings)} utterances")
        
        return embeddings_by_accent
    
    def label_candidate_speech(self, reference_embeddings, candidate_embeddings, output_file):
        """Label candidate speech with the closest reference speaker."""
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["accent", "candidate_speaker", "closest_reference_speaker", "cosine_similarity"])
            
            for accent in sorted(candidate_embeddings.keys()):
                print(f"Labeling candidate speech for accent: {accent}")
                for candidate_speaker, candidate_embedding in sorted(candidate_embeddings[accent].items()):
                    best_match = None
                    best_similarity = -1.0
                    for reference_speaker, reference_embedding in sorted(reference_embeddings[accent].items()):
                        if reference_speaker not in candidate_speaker.split("-")[1].split("_"):
                            continue
                        similarity = np.dot(candidate_embedding, reference_embedding)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = reference_speaker
                    if best_similarity < self.threshold:
                        best_match = "unknown"
                    writer.writerow([accent, candidate_speaker, best_match, f"{best_similarity:.4f}"])


def run_model(
    root_data,
    output_directory,
    model_type,
    threshold=None,
):
    print("=" * 50)
    print(f"Using {model_type} speaker-embedding model")
    print("=" * 50)

    labelizer = SpeakerLabelizer(
        root_data_path=root_data,
        model_type=model_type,
        threshold=threshold,
    )

    print("\nBuilding reference speaker embeddings...")
    reference_embeddings = labelizer.build_centroid_embeddings(task="retold_passages")
    
    print("\nExtract candidate speaker embeddings...")
    candidate_embeddings = labelizer.build_centroid_embeddings(task="free_conversation")
    candidate_embeddings_map_task = labelizer.build_centroid_embeddings(task="map_task")
    # merge two dictionary
    for accent in candidate_embeddings:
        assert accent in candidate_embeddings_map_task, f"Accent {accent} not found in map_task embeddings"
        candidate_embeddings[accent].update(candidate_embeddings_map_task[accent])
    print("\nLabeling candidate speech...")
    output_file = os.path.join(output_directory, f"speaker_labels_{model_type}.tsv")
    labelizer.label_candidate_speech(reference_embeddings, candidate_embeddings, output_file)


def main():
    root_data = "/home/s2526235/data/IViE/segmented"
    output_directory = os.path.join(root_data, "speaker_labeling_results")
    os.makedirs(output_directory, exist_ok=True)

    for model_type, threshold in (
        ("wavlm", 0.86),
        # ("resemblyzer", 0.75),
    ):
        try:
            run_model(
                root_data=root_data,
                output_directory=output_directory,
                model_type=model_type,
                threshold=threshold,
            )
        except Exception as error:
            print(f"Error with {model_type}: {error}")


if __name__ == "__main__":
    main()