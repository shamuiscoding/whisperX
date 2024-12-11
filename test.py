import os
import whisperx
import torch
import gc
from tqdm import tqdm

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16  # Reduce if low on GPU memory
compute_type = "float16" if torch.cuda.is_available() else "int8"
input_dir = "./Input"  # Input directory containing .mp3 files
output_dir = "./Results/"  # Output directory for results
hf_token = (
    "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"  # Replace with your Hugging Face token
)
word_level_dia = True

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the Whisper model
model = whisperx.load_model("large-v3", device, compute_type=compute_type)

# Process each file
mp3_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
for filename in tqdm(mp3_files, desc="Processing audio files"):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(
        output_dir, f"{os.path.splitext(filename)[0]}_results.txt"
    )

    # Load audio file
    audio = whisperx.load_audio(input_path)

    # Step 1: Transcription
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        prompt="Transcribe disfluencies such as: um, uh, like, haha, heh, hmm, you know, I mean, er, ah, eh, okay, right, so, well, yeah, uh huh, mm hmm, oh, uh oh, huh, yeah no, etc.",
    )

    # Clear model from memory if resources are low
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Alignment
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Clear alignment model from memory
    gc.collect()
    torch.cuda.empty_cache()
    del model_a

    # Step 3: Diarization
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(
        audio, min_speakers=1, max_speakers=2, return_logits=True
    )

    # Assign speaker labels
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Save results to a text file
    with open(output_path, "w") as f:
        f.write("Segments with Speaker Labels:\n")
        for segment in result["segments"]:
            f.write(
                f"start={segment['start']:.1f}s stop={segment['end']:.1f}s speaker={segment.get('speaker', 'unknown')}\n"
            )
            f.write(f"text={segment['text']}\n")

            # Add word-level information
            if word_level_dia and "words" in segment:
                f.write("Word-level breakdown:\n")
                for word in segment["words"]:
                    f.write(
                        f"  {word['word']}: {word.get('start', '?'):.1f}s - {word.get('end', '?'):.1f}s "
                        f"[speaker={word.get('speaker', 'unknown')}]\n"
                    )
            f.write("\n")

    print(f"Processed {filename} and saved results to {output_path}")

# Clean up resources
del model
gc.collect()
torch.cuda.empty_cache()

print("All files have been processed and results saved.")
