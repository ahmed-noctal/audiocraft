import gradio as gr
import torchaudio
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import os
import torch

# Load the pre-trained model
print("Loading pre-trained AudioGen model...")
model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(duration=5)  # Generate 5 seconds.
print("Model loaded successfully.")

# Fixed file path in the current working directory
fixed_audio_path = os.path.join(os.getcwd(), "generated_audio")
print(f"Fixed audio path: {fixed_audio_path}")

def generate_audio(descriptions):
    try:
        print("Received descriptions:")
        print(descriptions)

        # Split the input descriptions by line and remove any empty lines
        descriptions_list = [desc.strip() for desc in descriptions.strip().split("\n") if desc.strip()]
        print(f"Processed descriptions list: {descriptions_list}")

        if not descriptions_list:
            raise ValueError("Please provide at least one valid description.")

        # Generate audio for each description
        print("Generating audio...")
        wavs = model.generate(descriptions_list)
        print(f"Generated wavs: {wavs}")

        # Check if wavs is a tensor and has data
        if isinstance(wavs, torch.Tensor) and wavs.numel() > 0:
            # Use the first generated audio
            audio_tensor = wavs[0][0].cpu()  # Assuming shape (batch, channels, samples)
            print(f"Audio tensor shape: {audio_tensor.shape}, dtype: {audio_tensor.dtype}")

            # Attempt to write the audio using audiocraft's audio_write
            try:
                print(f"Attempting to write audio to {fixed_audio_path} using audio_write...")
                audio_write(fixed_audio_path, audio_tensor, model.sample_rate, strategy="loudness", loudness_compressor=True)
                print("audio_write succeeded.")
            except Exception as e:
                print(f"audio_write failed with error: {e}")
                print("Attempting to write audio using torchaudio as a fallback...")
                try:
                    torchaudio.save(fixed_audio_path, audio_tensor.unsqueeze(0), sample_rate=model.sample_rate)
                    print("torchaudio.save succeeded.")
                except Exception as te:
                    print(f"torchaudio.save failed with error: {te}")
                    raise RuntimeError(f"Failed to write audio file: {te}")

            # Verify that the file was created successfully
            if os.path.isfile(fixed_audio_path + ".wav"):
                print(f"Audio file exists: {fixed_audio_path}")
                return fixed_audio_path + ".wav"
            else:
                print(f"Audio file was not created: {fixed_audio_path}")
                raise FileNotFoundError("Audio file was not created successfully.")
        else:
            print("No wavs generated or wavs is not a tensor.")
            raise ValueError("No audio was generated. Please check your input descriptions.")
    except Exception as e:
        print(f"Exception in generate_audio: {e}")
        raise e

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("""
    # AudioGen Demo
    Enter descriptions for audio generation (one description per line). Only the first description will produce a fixed-named audio file.
    """)

    with gr.Row():
        description_input = gr.Textbox(
            label="Descriptions (one per line)",
            placeholder="dog barking\nsirens of an emergency vehicle\nfootsteps in a corridor",
            lines=5
        )

    with gr.Row():
        generate_button = gr.Button("Generate Audio")

    with gr.Row():
        audio_output = gr.Audio(
            label="Generated Audio",
            type="filepath",
            interactive=False
        )

    # Connect the button to the generate_audio function
    generate_button.click(generate_audio, inputs=[description_input], outputs=[audio_output])

# Launch the Gradio app
demo.launch()
